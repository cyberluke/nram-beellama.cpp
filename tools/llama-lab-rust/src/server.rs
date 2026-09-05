use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Read a model file's GGUF metadata and extract MoE expert counts.
/// A MoE model only reads the *used* experts per token, so decode speed scales
/// with (used/total) weight mass — this is why the old full-file estimate said
/// 49 tok/s while a Qwen3-27B-A3B actually generates at ~81.
#[derive(Clone, Copy, Default)]
pub struct GgufMoE {
    pub expert_count: u64,
    pub expert_used: u64,
}

pub fn gguf_moe_info(path: &str) -> Option<GgufMoE> {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else { return None };
    let mut hdr = [0u8; 26];
    if f.read_exact(&mut hdr).is_err() {
        return None;
    }
    // "GGUF" magic + version u32 + tensor_count u64 = 4+4+8 bytes.
    if &hdr[0..4] != b"GGUF" {
        return None;
    }
    let mut rd = |b: &mut [u8]| -> bool { f.read_exact(b).is_ok() };
    // meta_kv_count: u64 at offset 20.
    let mut kv_buf = [0u8; 8];
    if !rd(&mut kv_buf) {
        return None;
    }
    let kv_count = u64::from_le_bytes(kv_buf);
    let mut expert_count: u64 = 0;
    let mut expert_used: u64 = 0;
    for _ in 0..kv_count.min(2000) {
        // key: u64 len + bytes
        if !rd(&mut kv_buf) {
            return None;
        }
        let klen = u64::from_le_bytes(kv_buf) as usize;
        if klen > 4096 {
            return None;
        }
        let mut key = vec![0u8; klen];
        if !rd(&mut key) {
            return None;
        }
        let key = String::from_utf8_lossy(&key).into_owned();
        // value type: u32
        let mut vt = [0u8; 4];
        if !rd(&mut vt) {
            return None;
        }
        let vtype = u32::from_le_bytes(vt);
        // Values of interest are scalars UINT32(4)/INT32(5)/UINT64(10)/INT64(11).
        let read_scalar = |rd: &mut dyn FnMut(&mut [u8]) -> bool| -> Option<u64> {
            let mut buf = [0u8; 8];
            match vtype {
                4 | 5 => {
                    let mut b4 = [0u8; 4];
                    if !rd(&mut b4) {
                        return None;
                    }
                    Some((u32::from_le_bytes(b4) as i32) as i64 as u64)
                }
                10 | 11 => {
                    if !rd(&mut buf) {
                        return None;
                    }
                    Some((i64::from_le_bytes(buf)) as u64)
                }
                _ => None,
            }
        };
        // Value by type; skip unknown types properly by parsing lengths so we
        // keep the cursor aligned through the whole metadata blob.
        let val: Option<u64> = match vtype {
            0..=7 => read_scalar(&mut rd),
            8 => {
                let mut lb = [0u8; 8];
                let l = if rd(&mut lb) { u64::from_le_bytes(lb) as usize } else { return None };
                let mut s = vec![0u8; l.min(4096)];
                let want = l.min(4096);
                if !rd(&mut s[..want]) {
                    return None;
                }
                // skip remainder of oversized strings (shouldn't happen)
                let mut skip = [0u8; 1024];
                let mut remaining = l - want;
                while remaining > 0 {
                    let n = remaining.min(1024);
                    if !rd(&mut skip[..n]) {
                        return None;
                    }
                    remaining -= n;
                }
                None
            }
            9 => {
                // array: u32 elem_type + u64 count + elems
                let mut et = [0u8; 4];
                let mut cnt = [0u8; 8];
                if !rd(&mut et) || !rd(&mut cnt) {
                    return None;
                }
                let etype = u32::from_le_bytes(et);
                let ecount = u64::from_le_bytes(cnt);
                for _ in 0..ecount.min(1_000_000) {
                    // element value — just skip by the same scalar rules
                    let elen = match etype {
                        0..=7 => 1u64, // approximate: u8/i8/u16/i16/u32/i32/f32/bool ≤ 4B — see below
                        _ => 0,
                    };
                    let _ = elen;
                    // Actually consume 4 or 8 bytes based on type.
                    let mut b4 = [0u8; 4];
                    let mut b8 = [0u8; 8];
                    match etype {
                        0 | 1 => { if !rd(&mut b4[..1]) { return None; } }
                        2 | 3 => { if !rd(&mut b4[..2]) { return None; } }
                        4 | 5 | 6 | 7 => { if !rd(&mut b4) { return None; } }
                        10 | 11 | 12 => { if !rd(&mut b8) { return None; } }
                        8 => {
                            let mut lb = [0u8; 8];
                            let l = if rd(&mut lb) { u64::from_le_bytes(lb) as usize } else { return None };
                            let mut skip = [0u8; 1024];
                            let mut remaining = l;
                            while remaining > 0 {
                                let n = remaining.min(1024);
                                if !rd(&mut skip[..n]) { return None; }
                                remaining -= n;
                            }
                        }
                        9 => return None,
                        _ => return None,
                    }
                }
                None
            }
            _ => return None,
        };
        if let Some(v) = val {
            if key == "llama.expert_count" || key == "llama.experts.count" {
                expert_count = v;
            } else if key == "llama.expert_used_count" || key == "llama.experts.used_count" {
                expert_used = v;
            }
        }
        if expert_count > 0 && expert_used > 0 {
            break;
        }
    }
    if expert_count > 0 {
        return Some(GgufMoE { expert_count, expert_used });
    }
    None
}

/// Hardware queries spawn sub-processes (nvidia-smi / vulkaninfo / PowerShell).
/// Running them on every render frame froze the TUI, so each query is computed
/// once and cached for the session (hardware doesn't change under us).
use std::sync::OnceLock;

static VRAM_CACHE: OnceLock<f64> = OnceLock::new();
static RAM_CACHE: OnceLock<f64> = OnceLock::new();
static GPU_INFO_CACHE: OnceLock<(String, f64)> = OnceLock::new();
static BANDWIDTH_CACHE: OnceLock<f64> = OnceLock::new();
static CUDA_OK_CACHE: OnceLock<bool> = OnceLock::new();

/// Whether the CUDA backend is usable: only when nvidia-smi reports an NVIDIA
/// GPU (name + used amount parse). AMD/Intel machines get `false` so CUDA is
/// hidden instead of being offered and failing at launch.
pub fn supports_cuda() -> bool {
    *CUDA_OK_CACHE.get_or_init(|| {
        let (name, used) = gpu_info_raw();
        let ok = nvidia_smi_ok();
        !name.is_empty() && used >= 0.0 && ok
    })
}

/// True if `nvidia-smi` executes and succeeds (NVIDIA driver present).
fn nvidia_smi_ok() -> bool {
    run_output_cmd("nvidia-smi", &["-L"], Duration::from_millis(1500))
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a command with a hard timeout, returning its Output or None. Every
/// hardware probe goes through here so a hanging/hung probe (stale driver,
/// slow Vulkan ICD) can NEVER freeze the TUI.
pub fn run_output_cmd(cmd: &str, args: &[&str], timeout: Duration) -> Option<std::process::Output> {
    let mut child = Command::new(cmd).args(args).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().ok()?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => return None,
        }
    }
}

/// Query total VRAM (MiB), computed once.
pub fn detect_vram_mib_cached() -> f64 {
    *VRAM_CACHE.get_or_init(detect_vram_mib_raw)
}

/// Query total physical RAM (MiB), computed once.
pub fn detect_ram_mib_cached() -> f64 {
    *RAM_CACHE.get_or_init(detect_ram_mib_raw)
}

/// GPU name + VRAM used (MiB), computed once.
pub fn gpu_info_cached() -> (String, f64) {
    GPU_INFO_CACHE.get_or_init(gpu_info_raw).clone()
}

/// GPU memory bandwidth (GB/s), computed once. Decode tok/s is memory-bound
/// (each token reads the whole weight file), so this is the driving number for
/// the CUDA↔Vulkan comparison. Derived from the detected GPU name's real
/// specification (bus width × effective clock). Unknown → 0 (shown as "?").
pub fn gpu_bandwidth_gbps() -> f64 {
    *BANDWIDTH_CACHE.get_or_init(|| {
        let name = gpu_info_cached().0.to_lowercase();
        // Real specs of common cards (GB/s) — NVIDIA, AMD and Intel.
        let known: &[(&str, f64)] = &[
            // NVIDIA consumer / workstation
            ("rtx 4090", 1007.8),
            ("rtx 4080", 716.8),
            ("rtx 4070 ti super", 672.0),
            ("rtx 4070 super", 504.2),
            ("rtx 4070", 504.2),
            ("rtx 4060 ti", 288.0),
            ("rtx 4060", 272.0),
            ("rtx 3090", 936.2),
            ("rtx 3080", 760.3),
            ("rtx 3070", 448.0),
            ("rtx 3060 ti", 448.0),
            ("rtx 3060", 360.0),
            ("rtx 2080 ti", 616.0),
            ("rtx 2070 super", 448.0),
            ("rtx 2070", 448.0),
            ("rtx 2060", 336.0),
            ("rtx 5080", 960.0),
            ("rtx 5070 ti", 896.0),
            ("rtx 5070", 672.0),
            ("rtx 5060 ti", 448.0),
            ("a100", 2039.0),
            ("h100", 3350.0),
            ("l40s", 3456.0),
            // NVIDIA professional Quadra / RTX A-series
            ("rtx a6000", 768.0),
            ("rtx a5000", 768.0),
            ("rtx a4000", 448.0),
            ("rtx 6000 ada", 960.0),
            // AMD RDNA2 / RDNA3 (GPU/VRAM)
            ("rx 7900 xtx", 960.0),
            ("rx 7900 xt", 800.0),
            ("rx 7800 xt", 624.0),
            ("rx 7700 xt", 432.0),
            ("rx 7600 xt", 288.0),
            ("rx 7600", 288.0),
            ("rx 6950 xt", 576.0),
            ("rx 6900 xt", 512.0),
            ("rx 6800 xt", 512.0),
            ("rx 6800", 512.0),
            ("rx 6700 xt", 384.0),
            ("rx 6600 xt", 256.0),
            ("rx 6600", 224.0),
            // AMD CDNA datacenter
            ("mi300x", 5300.0),
            ("mi250x", 3276.0),
            ("mi210", 3276.0),
            ("mi100", 1228.8),
            // Intel Arc
            ("arc b580", 456.0),
            ("arc b570", 380.0),
            ("arc a770", 560.0),
            ("arc a750", 512.0),
            ("arc a380", 186.0),
        ];
        for (key, gbps) in known {
            if name.contains(key) {
                return *gbps;
            }
        }
        // Unlisted GPU: return 0 so tok/s shows "?" instead of a fabricated
        // number. The plan (ctx/KV/VRAM) still works fine.
        0.0
    })
}

/// Backend-agnostic VRAM probe (called once, result cached).
fn detect_vram_mib_raw() -> f64 {
    // NVIDIA first (accurate, cheap).
    let out = run_output_cmd("nvidia-smi", &["--query-gpu=memory.total", "--format=csv,noheader,nounits"], Duration::from_millis(1500));
    if let Some(o) = out {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if let Ok(mib) = s.parse::<f64>() {
                return mib;
            }
        }
    }
    // Cross-vendor via vulkaninfo: read `size   = <bytes>` lines and keep the
    // largest device heap ≤ 48 GiB (discrete-GPU class; skips Intel iGPU
    // shared-RAM heaps that can report 72+ GiB).
    if let Some(o) = run_output_cmd("vulkaninfo", &[], Duration::from_secs(3)) {
        let text = String::from_utf8_lossy(&o.stdout).to_string()
            + &String::from_utf8_lossy(&o.stderr);
        let mut best: f64 = 0.0;
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with("size") {
                if let Some(eq) = t.find('=') {
                    let val = t[eq + 1..].trim();
                    if let Ok(by) = val.split_whitespace().next().unwrap_or("").parse::<f64>() {
                        let gib = by / 1e9;
                        if gib > 0.0 && gib <= 48.0 && by > best {
                            best = by;
                        }
                    }
                }
            }
        }
        if best > 0.0 {
            return best / (1024.0 * 1024.0);
        }
    }
    24576.0 // final fallback only when no GPU can be probed
}

/// Memory pool for the backend: VRAM for CUDA/Vulkan, system RAM for CPU.
/// Both are cached (OnceLock) so per-frame calls are free.
pub fn detect_mem_mib(backend: crate::optimizer::Backend) -> f64 {
    match backend {
        crate::optimizer::Backend::Cpu => detect_ram_mib_cached(),
        crate::optimizer::Backend::Cuda | crate::optimizer::Backend::Vulkan => detect_vram_mib_cached(),
    }
}

/// Total physical RAM (MiB), probe once then cache.
fn detect_ram_mib_raw() -> f64 {
    // Total physical RAM (MiB) via CIM. wmic is REMOVED on Windows 11,
    // so the modern PowerShell/Get-CimInstance path is the one that
    // returns real values (verified: 136.7 GB on a 128 GiB box).
    let ps = "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory";
    if let Some(o) = run_output_cmd("powershell", &["-NoProfile", "-Command", ps], Duration::from_secs(4)) {
        let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if let Ok(b) = s.parse::<f64>() {
            if b > 0.0 {
                return b / (1024.0 * 1024.0);
            }
        }
    }
    // Win32_PhysicalMemory capacity sum as a second shape.
    let ps2 = "Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum | Select-Object -ExpandProperty Sum";
    if let Some(o) = run_output_cmd("powershell", &["-NoProfile", "-Command", ps2], Duration::from_secs(4)) {
        let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if let Ok(b) = s.parse::<f64>() {
            if b > 0.0 {
                return b / (1024.0 * 1024.0);
            }
        }
    }
    -1.0 // unknown — callers show "?" instead of a fabricated number
}

/// GPU display name + current VRAM (used MiB) — probe once, cached.
/// nvidia-smi gives the authoritative name; on non-NVIDIA machines (AMD/Intel)
/// it falls back to vulkaninfo `deviceName` (may list several devices; we take
/// the last — the primary discrete one typically appears last).
fn gpu_info_raw() -> (String, f64) {
    let out = run_output_cmd("nvidia-smi", &["--query-gpu=name,memory.used", "--format=csv,noheader"], Duration::from_millis(1500));
    if let Some(o) = out {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout);
            if let Some(line) = s.lines().next() {
                // "NVIDIA GeForce RTX 4090, 12345 MiB"
                let mut parts = line.rsplitn(2, ',').map(|p| p.trim().to_string()).collect::<Vec<_>>();
                parts.reverse();
                let name = parts.get(0).cloned().unwrap_or_default();
                let used_mib = parts.get(1).cloned().and_then(|v| {
                    v.trim_end_matches("MiB").trim().parse::<f64>().ok()
                }).unwrap_or(-1.0);
                return (name, used_mib);
            }
        }
    }
    // Cross-vendor: parse vulkaninfo deviceName lines.
    if let Some(o) = run_output_cmd("vulkaninfo", &[], Duration::from_secs(3)) {
        let text = String::from_utf8_lossy(&o.stdout).to_string()
            + &String::from_utf8_lossy(&o.stderr);
        let mut last = String::new();
        for line in text.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("deviceName") {
                if let Some(eq) = rest.find('=') {
                    last = rest[eq + 1..].trim().to_string();
                }
            }
        }
        if !last.is_empty() {
            return (last, -1.0); // no used-amount from vulkaninfo
        }
    }
    (String::new(), -1.0)
}

/// A model discovered on disk.
pub struct Model {
    pub path: String,
    pub name: String,
    pub size_gb: f64,
    /// Optional multimodal projector (mmproj-*.gguf) paired with this model.
    pub mmproj: Option<String>,
}

/// Run `llama-server.exe --help` once and collect every long flag it advertises
/// (e.g. "--reasoning-loop-guard"). Empty on failure — the caller then passes
/// everything (best-effort).
pub fn supported_flags(exe: &str) -> Vec<String> {
    let out = run_output_cmd(exe, &["--help"], Duration::from_secs(5));
    let Some(o) = out else { return Vec::new() };
    if !o.status.success() {
        return Vec::new();
    }
    let mut flags = Vec::new();
    let s = String::from_utf8_lossy(&o.stdout);
    let text = s.to_string() + &String::from_utf8_lossy(&o.stderr);
    for tok in text.split_whitespace() {
        if tok.starts_with("--") {
            let flag = tok.trim_start_matches(',').trim_end_matches(',').trim_end_matches('=');
            let flag = flag.to_string();
            if !flags.contains(&flag) {
                flags.push(flag);
            }
        }
    }
    flags
}

/// Discover *.gguf models under common roots (LM Studio, HF cache, env override).
/// Resolve authoritative model directories from each app's CONFIG file, not
/// guesses. Sources:
///   - LM Studio home: ~/.lmstudio-home-pointer (relocated home), then
///     <home>/settings.json -> "downloadsFolder".
///   - Ollama: OLLAMA_MODELS env (its config is env driven on Windows).
///   - ~/.ollama default.
fn load_config_model_dirs() -> Vec<String> {
    let mut dirs = Vec::new();
    let home = dirs::home_dir();
    if let Some(h) = home {
        let hp = h.as_path().to_string_lossy().into_owned();

        // 1) LM Studio home pointer -> may relocate the whole .lmstudio dir.
        let ptr = format!(r"{}\.lmstudio-home-pointer", hp);
        let lms_home = if let Ok(p) = std::fs::read_to_string(&ptr) {
            let t = p.trim().to_string();
            if !t.is_empty() {
                t
            } else {
                format!(r"{}\.lmstudio", hp)
            }
        } else {
            format!(r"{}\.lmstudio", hp)
        };

        // 2) LM Studio settings.json -> downloadsFolder (the real models root).
        let settings = format!(r"{lms_home}\settings.json");
        if let Ok(json) = std::fs::read_to_string(&settings) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(df) = v.get("downloadsFolder").and_then(|x| x.as_str()) {
                    if !df.is_empty() {
                        dirs.push(df.to_string());
                    }
                }
            }
        }
        // LM Studio default models location too.
        dirs.push(format!(r"{lms_home}\models"));

        // 3) Ollama default.
        dirs.push(format!(r"{hp}\.ollama"));
    }
    dirs
}

pub fn discover_models() -> Vec<Model> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut roots: Vec<String> = Vec::new();

    // Authoritative app-config dirs first (LM Studio downloadsFolder, Ollama).
    for r in load_config_model_dirs() {
        if !roots.contains(&r) {
            roots.push(r);
        }
    }

    if let Some(home) = dirs::home_dir() {
        let h = home.to_string_lossy().into_owned();
        roots.push(format!(r"{}\.cache\lm-studio\models", h));
        roots.push(format!(r"{}\.lmstudio\models", h));
        // HF hub cache (honor the OS override first, default ~/.cache else).
        let hf_default = format!(r"{}\.cache\huggingface\hub", h);
        let hf_home = std::env::var("HF_HOME")
            .map(|v| format!(r"{v}\hub"))
            .unwrap_or_else(|_| hf_default.clone());
        roots.push(hf_home);
        let hf_cache = std::env::var("HF_HUB_CACHE").unwrap_or_else(|_| hf_default.clone());
        roots.push(hf_cache);
        // Ollama model blobs live under .ollama/models/blobs (or OLLAMA_MODELS).
        let ollama = format!(r"{}\.ollama\models\blobs", h);
        if std::path::Path::new(&ollama).is_dir() {
            roots.push(ollama);
        }
    }
    // Honor the real, OS-overridden model roots the user has configured:
    // OLLAMA_MODELS points Ollama at e.g. C:\_MODELS; also scan the LM Studio /
    // HF style roots that actually exist. Env override wins.
    for ev in ["OLLAMA_MODELS", "LLAMA_MODEL_DIRS", "MY_MODELS"] {
        if let Ok(v) = std::env::var(ev) {
            for part in v.split(';') {
                let p = part.trim();
                if !p.is_empty() {
                    roots.push(p.to_string());
                }
            }
        }
    }
    // No drive-root heuristic fallback: discovery is 100% config/env-driven.
    for root in roots {
        // Ollama roots: a dir containing a `manifests` subdir (e.g. C:\_MODELS
        // via OLLAMA_MODELS, or ~/.ollama) OR the blobs dir itself.
        let is_ollama = std::path::Path::new(&root).join("manifests").is_dir()
            || root.to_lowercase().contains("ollama")
            || root.to_lowercase().ends_with("\\blobs");
        if is_ollama {
            walk_ollama(&root, &mut out);
        } else {
            walk_gguf(&root, &mut seen, &mut out);
        }
    }
    // de-dup by path
    let mut dedup: Vec<Model> = Vec::new();
    for m in out {
        if !dedup.iter().any(|x| x.path == m.path) {
            dedup.push(m);
        }
    }
    dedup.sort_by(|a, b| b.size_gb.partial_cmp(&a.size_gb).unwrap_or(std::cmp::Ordering::Equal));
    dedup
}

/// Discover *.gguf models under a single explicit directory (recursive).
pub fn discover_dir(dir: &str) -> Vec<Model> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    walk_gguf(dir, &mut seen, &mut out);
    out
}

/// Resolve Ollama models from `manifests` (registry.ollama.ai/.../latest) so
/// the blobs (which lack a .gguf extension) get a friendly name + real size.
/// Returns model records with path = the resolved blob file.
fn walk_ollama(root: &str, out: &mut Vec<Model>) {
    if root.is_empty() {
        return;
    }
    // manifests live at <root>\manifests\registry.ollama.ai\...
    let manifests_dir = std::path::Path::new(root)
        .join("manifests")
        .join("registry.ollama.ai");
    let blobs_dir = std::path::Path::new(root).join("blobs");
    // If `root` IS C:\_MODELS (OLLAMA_MODELS), the structure has manifests/ +
    // blobs/. For ~/.ollama it's .ollama\models\manifests + .ollama\models\blobs.
    let mut search_roots = vec![manifests_dir.clone()];
    let alt = std::path::Path::new(&root).parent().map(|p| p.join("models"));
    if let Some(a) = alt {
        search_roots.push(a.join("manifests").join("registry.ollama.ai"));
        search_roots.push(a.join("blobs"));
    }
    for mroot in search_roots {
        if !mroot.exists() {
            continue;
        }
        // walk registry.ollama.ai/{namespace}/{model}/{tag-or-md5}/latest
        let mut stack = vec![mroot];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().map(|f| f.eq_ignore_ascii_case("latest")).unwrap_or(false)
                    || p.extension().map(|e| e.eq_ignore_ascii_case("json")).unwrap_or(false)
                {
                    if let Ok(meta) = std::fs::read_to_string(&p) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&meta) {
                            // manifest has "layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:..."}]
                            if let Some(layers) = v.get("layers").and_then(|l| l.as_array()) {
                                for lyr in layers {
                                    let is_model = lyr
                                        .get("mediaType")
                                        .and_then(|m| m.as_str())
                                        .map(|m| m.contains("model"))
                                        .unwrap_or(false);
                                    if is_model {
                                        let digest = lyr.get("digest").and_then(|d| d.as_str()).unwrap_or("");
                                        let size = lyr.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
                                        let blob = blobs_dir.join(digest);
                                        if blob.exists() {
                                            let name = p
                                                .parent()
                                                .and_then(|pp| pp.file_name())
                                                .map(|n| n.to_string_lossy().into_owned())
                                                .unwrap_or_else(|| "ollama-model".into());
                                            out.push(Model {
                                                path: blob.to_string_lossy().into_owned(),
                                                name: format!("ollama/{name}"),
                                                size_gb: size as f64 / 1e9,
                                                mmproj: None,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Distributed/interrupted download marker sidecars we must reject.
fn has_incomplete_sidecar(p: &std::path::Path) -> bool {
    let parent = p.parent().map(|d| d.to_path_buf());
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some(par) = parent {
        // Direct sibling markers: <file>.incomplete / .partial / .part / ...
        for suffix in [".incomplete", ".partial", ".download", ".tmp", ".crdownload"] {
            if par.join(format!("{name}{suffix}")).exists() {
                return true;
            }
            let base = name.trim_end_matches(".gguf");
            if par.join(format!("{base}{suffix}")).exists() {
                return true;
            }
        }
        // HF / LM Studio prefixed partial: `downloading_<file>.gguf.part` means
        // <file>.gguf is not yet complete → treat as interrupted.
        let bare = name.trim_end_matches(".gguf");
        let partial = format!("downloading_{bare}.gguf.part");
        if par.join(&partial).exists() {
            return true;
        }
    }
    false
}

/// Minimal GGUF validity: header magic "GGUF" (0x46554747) at byte 0.
fn gguf_header_valid(p: &std::path::Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(p) else { return false };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return false;
    }
    magic == *b"GGUF"
}

/// A docker/ollama manifest (application/vnd.ollama.image.model) found in a dir.
fn ollama_manifest_from_dir(p: &std::path::Path) -> Option<Model> {
    // Look for a sibling `*.json`-ish docker manifest referencing a blob.
    let Ok(rd) = std::fs::read_dir(p) else { return None };
    // candidate manifests = files (not sha256 blobs) that are valid JSON with layers
    for entry in rd.flatten() {
        let f = entry.path();
        if !f.is_file() {
            continue;
        }
        let fname = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if fname.starts_with("sha256-") {
            continue; // blob, not manifest
        }
        if fname == "latest" || f.extension().map(|e| e.eq_ignore_ascii_case("json")).unwrap_or(false)
            || fname.ends_with(".1") || fname.ends_with(".0") {
            if let Ok(txt) = std::fs::read_to_string(&f) {
                if txt.contains("application/vnd.ollama.image.model") {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                        if let Some(layers) = v.get("layers").and_then(|l| l.as_array()) {
                            for lyr in layers {
                                let is_model = lyr.get("mediaType").and_then(|m| m.as_str())
                                    .map(|m| m.contains("ollama.image.model")).unwrap_or(false);
                                if is_model {
                                    let digest = lyr.get("digest").and_then(|d| d.as_str()).unwrap_or("");
                                    let size = lyr.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
                                    let from = lyr.get("from").and_then(|s| s.as_str()).unwrap_or("ollama");
                                    let blob = p.join(digest);
                                    if blob.exists() {
                                        return Some(Model {
                                            path: blob.to_string_lossy().into_owned(),
                                            name: format!("ollama/{from}"),
                                            size_gb: size as f64 / 1e9,
                                            mmproj: None,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn walk_gguf(root: &str, seen: &mut std::collections::HashSet<String>, out: &mut Vec<Model>) {
    if !Path::new(root).is_dir() {
        return;
    }
    let mut stack = vec![root.to_string()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        // Detect nested Ollama-style model dirs (docker manifest + sha256 blobs).
        if let Some(om) = ollama_manifest_from_dir(Path::new(&dir)) {
            if seen.insert(om.path.clone()) {
                out.push(om);
            }
            continue; // don't also walk this as gguf (blobs have no .gguf)
        }
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p.to_string_lossy().into_owned());
            } else if p.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
                // Reject interrupted / invalid downloads.
                if has_incomplete_sidecar(&p) || !gguf_header_valid(&p) {
                    continue;
                }
                let abs = p.to_string_lossy().into_owned();
                if seen.insert(abs.clone()) {
                    let size_gb = std::fs::metadata(&p).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
                    // Pair with an mmproj-*.gguf sibling in the same dir (vision).
                    let dir = p.parent().map(|d| d.to_path_buf());
                    let mmproj = dir.and_then(|d| {
                        std::fs::read_dir(&d).ok().and_then(|rd| {
                            for e in rd.flatten() {
                                let n = e.file_name().to_string_lossy().to_lowercase();
                                if (n.starts_with("mmproj-") || n.starts_with("mmproj_")) && n.ends_with(".gguf") {
                                    return Some(e.path().to_string_lossy().into_owned());
                                }
                            }
                            None
                        })
                    });
                    out.push(Model {
                        path: abs,
                        name: p.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                        size_gb,
                        mmproj,
                    });
                }
            }
        }
    }
}

/// Events flowing from the server manager to the UI thread.
pub enum ServerEvent {
    LogLine(String),
    Healthy(bool),
    Exit(u32),
}

/// Kill any process already bound to `port` on 127.0.0.1 so a fresh spawn can
/// actually take the socket. Guards against a stale llama-server left over from
/// a crashed run holding the port while the TUI livelocks in "spouštím...".
fn reap_stale_port(port: u16) {
    if let Ok(out) = std::process::Command::new("netstat")
        .args(["-ano", "-p", "tcp"])
        .output()
    {
        let s = String::from_utf8_lossy(&out.stdout);
        for line in s.lines() {
            //  TCP    127.0.0.1:8888    0.0.0.0:0    LISTENING    12345
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() >= 5 {
                let local = toks[1];
                if local.rfind(':').map(|p| &local[p + 1..]).map_or(false, |prt| prt == port.to_string())
                    && toks[3].eq_ignore_ascii_case("LISTENING")
                {
                    if let Ok(pid) = toks[4].parse::<u32>() {
                        let _ = std::process::Command::new("taskkill")
                            .args(["/PID", &pid.to_string(), "/F", "/T"])
                            .status();
                    }
                }
            }
        }
    }
}

/// Manages the llama-server subprocess. Spawn + log-pipe + health poll.
pub struct ServerProcess {
    receiver: Receiver<ServerEvent>,
    // Shared child so Drop can terminate it. Mutex<Option<Child>>.
    child: Arc<Mutex<Option<Child>>>,
    request_stop: Arc<AtomicBool>,
    healthy: Arc<Mutex<bool>>,
}

impl ServerProcess {
    /// Spawn llama-server.exe `exe_name` (path) serving `model_path` on `port`,
    /// using the given pre-computed launch `plan` (already reflects the selected
    /// preset + vision choice). When `mmproj` is Some, vision is passed.
    pub fn start(
        model_path: &str,
        port: u16,
        exe_name: &str,
        mmproj: Option<&str>,
        plan: crate::optimizer::LaunchPlan,
    ) -> anyhow::Result<ServerProcess> {
        let (tx, rx) = mpsc::channel::<ServerEvent>();
        let healthy = Arc::new(Mutex::new(false));
        let request_stop = Arc::new(AtomicBool::new(false));

        let server_exe = exe_name.to_string();
        if !Path::new(&server_exe).exists() {
            return Err(anyhow::anyhow!("llama-server.exe not found at {}", server_exe));
        }

        // Capability probe: only pass flags the resolved binary actually
        // supports. Stock llama.cpp and the BeeLlama fork differ (e.g.
        // --reasoning-loop-guard), and the all-backend build is stock. We run
        // --help once and build the supported set, then drop any plan flag the
        // binary doesn't know — universally safe across builds.
        let supported = supported_flags(&server_exe);
        let plan_args = plan
            .args
            .iter()
            .filter(|f| {
                let f: &str = f.as_str();
                !f.starts_with('-')
                    || f == "--model"
                    || f == "--host"
                    || f == "--port"
                    || supported.iter().any(|s| s == f || s.starts_with(&format!("{f},")))
            })
            .cloned()
            .collect::<Vec<_>>();

        // Assemble: base model/host/port + optimizer args (skip its --model).
        let mut final_args: Vec<String> = Vec::new();
        final_args.push("--model".into());
        final_args.push(model_path.into());
        final_args.push("--host".into());
        final_args.push("127.0.0.1".into());
        final_args.push("--port".into());
        final_args.push(port.to_string());
        let mut i = 0;
        let pa = &plan_args;
        while i < pa.len() {
            let flag = &pa[i];
            i += 1;
            if flag == "--model" || flag == "--host" || flag == "--port" {
                i += 1; // skip paired value
                continue;
            }
            final_args.push(flag.clone());
            if i < pa.len() && !pa[i].starts_with('-') {
                final_args.push(pa[i].clone());
                i += 1;
            }
        }

        if let Some(mm) = mmproj {
            final_args.push("--mmproj".into());
            final_args.push(mm.to_owned());
        }

        // Prepend the server's own directory (its DLLs live there) and any
        // NVIDIA CUDA toolkit bin so ggml-cuda.dll can resolve cudart, matching
        // how the reference script launches the BeeLlama build.
        let mut path = String::new();
        if let Some(dir) = std::path::Path::new(&server_exe).parent() {
            path.push_str(&dir.to_string_lossy());
            path.push(';');
        }
        if let Ok(cuda) = std::env::var("CUDA_PATH") {
            path.push_str(&format!("{}\\bin;", cuda));
        }
        path.push_str(&std::env::var("PATH").unwrap_or_default());

        let mut cmd = Command::new(&server_exe);
        cmd.args(&final_args)
            .env("PATH", path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // A stale llama-server from a crashed run may still hold the port and
        // would (a) make our spawn exit instantly and (b) answer /health as a
        // foreign process. Reap it before we spawn so the socket is really ours.
        reap_stale_port(port);

        let child = cmd.spawn().map_err(|e| anyhow::anyhow!("spawn failed: {e}"))?;
        let child_shared = Arc::new(Mutex::new(Some(child)));
        let tx_log = tx.clone();
        let tx_err = tx.clone();

        if let Some(stdout) = child_shared.lock().unwrap().as_mut().and_then(|c| c.stdout.take()) {
            let tx = tx_log.clone();
            thread::spawn(move || {
                let rdr = BufReader::new(stdout);
                for line in rdr.lines() {
                    if let Ok(l) = line {
                        let _ = tx.send(ServerEvent::LogLine(l));
                    }
                }
            });
        }
        if let Some(stderr) = child_shared.lock().unwrap().as_mut().and_then(|c| c.stderr.take()) {
            let tx = tx_err.clone();
            thread::spawn(move || {
                let rdr = BufReader::new(stderr);
                for line in rdr.lines() {
                    if let Ok(l) = line {
                        let _ = tx.send(ServerEvent::LogLine(l));
                    }
                }
            });
        }

        // health poller
        let ok_h = healthy.clone();
        let stop = request_stop.clone();
        let tx_health = tx.clone();
        thread::spawn(move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_millis(1000))
                .build()
                .unwrap();
            let mut last = false;
            while !stop.load(Ordering::Relaxed) {
                let now = client
                    .get(format!("http://127.0.0.1:{}/health", port))
                    .send()
                    .map(|r| r.status().is_success())
                    .unwrap_or(false);
                {
                    let mut h = ok_h.lock().unwrap();
                    *h = now;
                }
                if now != last {
                    let _ = tx_health.send(ServerEvent::Healthy(now));
                    last = now;
                }
                thread::sleep(Duration::from_millis(500));
            }
        });

        // reaper: wait for child exit, notify UI, clear slot
        let tc = tx.clone();
        let child_shared2 = child_shared.clone();
        thread::spawn(move || {
            if let Some(mut c) = child_shared2.lock().unwrap().take() {
                let status = c.wait();
                let code = status.map(|s| s.code().unwrap_or(0) as u32).unwrap_or(u32::MAX);
                let _ = tc.send(ServerEvent::Exit(code));
            }
        });

        Ok(ServerProcess {
            receiver: rx,
            child: child_shared,
            request_stop,
            healthy,
        })
    }

    pub fn try_recv(&self) -> Option<ServerEvent> {
        self.receiver.try_recv().ok()
    }

    pub fn is_healthy(&self) -> bool {
        *self.healthy.lock().unwrap()
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        self.request_stop.store(true, Ordering::Relaxed);
        if let Some(mut c) = self.child.lock().unwrap().take() {
            let _ = c.kill();
        }
    }
}