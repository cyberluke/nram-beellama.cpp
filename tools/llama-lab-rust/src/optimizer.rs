/// Deterministic VRAM-based launch-parameter optimizer for llama-server.
///
/// Universal across model architectures and GPU cards: the KV-cache footprint is
/// estimated from the model file size with a single architecture-agnostic ratio
/// (no hard-coded dense/MoE split), and every capture of VRAM, context and slot
/// count flows from the card actually detected via nvidia-smi. Three preset
/// plans are offered — a conservative "safe" setup, a single-user max, and a
/// two-user concurrent split — so the driver picks the bar that fits the job.

/// Result of the optimizer: a full set of server args plus an explainability
/// chain (so the UI can show *why* each value was chosen).
#[derive(Clone)]
pub struct LaunchPlan {
    pub args: Vec<String>,
    pub summary: Vec<(String, String)>,   // "flag" -> "value"
    pub evidence: Vec<String>,            // human-readable reasoning lines
    pub ctx: i64,
    pub parallel: i64,
    pub vr_used_gb: f64,                  // predicted VRAM committed (weights + KV + draft)
    /// Estimated decode tokens/second (memory-bandwidth-bound model).
    pub tok_s: f64,
    /// Raw KV-cache quant code that this plan uses (e.g. "kvarn4", "q5_0").
    /// Mirrors the "--cache-type-k/v" args so the UI can tweak it in place.
    pub kv_quant: String,
}

/// The three selectable performance bars.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Conservative, // safe, low-risk headroom
    Solo,         // maximum context / speed for a single user
    Twin,         // two concurrent users sharing one unified pool
}

/// Compute backend selected before the model picker.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Vulkan,
}

impl Backend {
    pub const ALL: [Backend; 3] = [Backend::Cpu, Backend::Cuda, Backend::Vulkan];

    pub fn title_key(&self) -> &'static str {
        match self {
            Backend::Cpu => "be_cpu",
            Backend::Cuda => "be_cuda",
            Backend::Vulkan => "be_vulkan",
        }
    }

    /// Feature bullets shown on the selection slide (comma-separated keys).
    pub fn reasons(&self) -> &'static [&'static str] {
        match self {
            Backend::Cpu => &["be_cpu_r1", "be_cpu_r2", "be_cpu_r3"],
            Backend::Cuda => &["be_cuda_r1", "be_cuda_r2", "be_cuda_r3"],
            Backend::Vulkan => &["be_vulkan_r1", "be_vulkan_r2", "be_vulkan_r3"],
        }
    }

    /// n-gpu-layers: CPU keeps everything in RAM; GPU backends offload all.
    pub fn gpu_layers(&self) -> i64 {
        match self {
            Backend::Cpu => 0,
            Backend::Cuda | Backend::Vulkan => 999,
        }
    }

    /// Flash attention: on for GPU compute (CUDA/Vulkan), off for pure CPU.
    pub fn flash_attn(&self) -> bool {
        matches!(self, Backend::Cuda | Backend::Vulkan)
    }
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Conservative, Preset::Solo, Preset::Twin];

    /// Localization key for the card title (nice, non-technical).
    pub fn title_key(&self) -> &'static str {
        match self {
            Preset::Conservative => "preset_safe",
            Preset::Solo => "preset_solo",
            Preset::Twin => "preset_twin",
        }
    }

    /// Localization key for the one-line subtitle under the title.
    pub fn desc_key(&self) -> &'static str {
        match self {
            Preset::Conservative => "preset_safe_desc",
            Preset::Solo => "preset_solo_desc",
            Preset::Twin => "preset_twin_desc",
        }
    }
}

/// KV-cache bytes per token at f16, estimated from the model weight size.
/// Architecture-agnostic: dense and MoE models land within the same broad
/// band (a larger file generally means more layers / KV heads). We use a single
/// mid-band factor so a dense 70B isn't under-provisioned but a 27B-A3B MoE
/// still gets a generous pool instead of the overly-cautious legacy ratio.
const KV_F16_RATIO: f64 = 4.5e-6;

/// KV-quant bytes vs f16 (for the cache, NOT model weights).
/// Ratios from BeeLlama's benchmark ladder (README "Size vs bf16"):
/// kvarn8 52.3% · kvarn6 39.8% · kvarn5 33.6% · kvarn4 27.3% ·
/// kvarn3 21.1% · kvarn2 14.8% · q8_0 53.1% · q6_0 40.6% · q5_0 34.4% ·
/// q4_1 31.3% · q4_0/iq4_nl 28.1%.
fn quant_ratio(q: &str) -> f64 {
    match q {
        "f16" | "bf16" => 1.0,
        "q8_0" => 0.531,
        "kvarn8" => 0.523,
        "q6_0" | "q6_1" => 0.406,
        "kvarn6" => 0.398,
        "q5_0" | "q5_1" => 0.344,
        "kvarn5" => 0.336,
        "q4_1" => 0.313,
        "q4_0" | "iq4_nl" => 0.281,
        "kvarn4" => 0.273,
        "q3_0" | "q3_1" => 0.219,
        "kvarn3" => 0.211,
        "q2_0" | "q2_1" => 0.156,
        "kvarn2" => 0.148,
        _ => 1.0,
    }
}

fn kv_bytes_per_token(cache_quant: &str, weights_bytes: f64) -> f64 {
    weights_bytes * KV_F16_RATIO * quant_ratio(cache_quant)
}

/// BeeLlama precision-tail cost: `--kv-tail-tokens T` keeps the newest T tokens
/// exact (f16) while the body stays quantized. Cost ≈ T × full-f16 per-token.
fn kv_tail_bytes(tail_tokens: i64, weights_bytes: f64) -> f64 {
    (tail_tokens as f64) * weights_bytes * KV_F16_RATIO
}

/// GGUF weights map ~1:1 onto the GPU: the file IS the weight tensor data
/// (plus a few-MB header). No scratch markup — llama.cpp loads offload tensors
/// at their stored size; a +4% fudge overstated VRAM by ~0.7 GB on a 16.8 GB
/// model and the cards lied vs nvidia-smi.
fn weights_bytes(model_file_size_bytes: u64) -> f64 {
    model_file_size_bytes as f64
}

/// True VRAM we leave free (MiB), per-card:
///   min(CAP_MIB, BASELINE_MIB + vram_mib * SURPLUS_FRAC)
/// Constant baseline covers CUDA context / fragmentation on any card; the small
/// surplus share protects small GPUs; the cap stops big cards wasting proportion.
/// Headroom reserve (MiB): covers CUDA/Vulkan context, prefill compute buffers,
/// and the auxiliary MTP draft context above the KV pool. This scales a little
/// with the card (bigger GPU → bigger prefill buffers) but is floored so even a
/// small GPU leaves the ~1.5 GB that llama.cpp genuinely needs to allocate
/// compute graphs. Old 1 GB cap was exactly what OOM'd.
const HEADROOM_CAP_MIB: f64 = 3072.0;
const HEADROOM_BASELINE_MIB: f64 = 1024.0;
const HEADROOM_SURPLUS_FRAC: f64 = 0.05;

fn headroom_mib(vram_mib: f64) -> f64 {
    (HEADROOM_BASELINE_MIB + vram_mib * HEADROOM_SURPLUS_FRAC).min(HEADROOM_CAP_MIB)
}

/// Hard safety margin the final plan must leave free (bytes): after the KV and
/// buffers there must still be this much of the card unclaimed or we shrink ctx.
/// This is what makes "must fit" a hard rule instead of an estimate.
const HARD_MARGIN_BYTES: f64 = 256.0 * 1024.0 * 1024.0;

pub const DRAFT_QUANT: &str = "q4_1"; // MTP draft: denser aux context (KVarN is target-only); q4_1 keeps the aux VRAM budget lean so the main KV pool stays max'd.

/// Speculative-decoding (MTP) gain over the pure bandwidth-bound decode rate.
/// MTP verifies multiple tokens per model step and its draft shares compute,
/// so a running server lands above the naive bytes/token estimate. Set from
/// observed serving (measured ~1.6–1.7× on MoE and dense alike).
const MTP_SPEEDUP: f64 = 1.65;

/// Round a context pool down to a multiple of these tokens.
pub const CTX_STEP: i64 = 16384; // +/– 16k nudges the user-tenable knobs by a full step
const ROUND: i64 = CTX_STEP;
pub const MAX_CTX: i64 = 262144;
/// A pool of at least this many tokens enables the "two users" split.
const TWIN_POOL: i64 = 65536;

/// Inputs to the optimizer.
#[derive(Clone, Copy)]
pub struct Inputs {
    pub backend: Backend,
    /// Memory available for weights+KV. For GPU backends this is VRAM (MiB);
    /// for CPU it is system RAM (MiB).
    pub mem_mib: f64,
    pub model_file_bytes: u64,
    pub vision_mib: f64,
    /// Weight mass actually read per token (1.0 = dense loads everything;
    /// < 1.0 = MoE where only the used experts contribute each step).
    pub moe_active: f64,
}

impl Inputs {
    /// Physical CPU core count for thread tuning, computed once (probe is a
    /// PowerShell spawn — actually running it per-plan was freezing the TUI).
    pub fn cpu_threads(&self) -> i64 {
        static CACHED: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            let ps = "Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum | Select-Object -ExpandProperty Sum";
            if let Some(o) = crate::server::run_output_cmd("powershell", &["-NoProfile", "-Command", ps], std::time::Duration::from_secs(4)) {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if let Ok(n) = s.parse::<i64>() {
                    if n > 0 {
                        return n;
                    }
                }
            }
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(1) as i64
        })
    }
}

/// Manual profile tweaks applied by the user in the Optimize screen. `None`
/// means "let the optimizer pick (its default) for that dimension."
#[derive(Clone, Copy, Default)]
pub struct Overrides {
    /// Forced context pool size (rounded to a 16k multiple, clamped to the
    /// backend max). `None` = optimizer's memory-derived default.
    pub ctx: Option<i64>,
    /// Forced raw KV-cache quant code (e.g. "kvarn4", "q5_0"). Must come from
    /// `quant_ladder`. `None` = optimizer default.
    pub quant: Option<&'static str>,
    /// Forced reasoning effort level (e.g. "high", "xhigh", "max"). Must come
    /// from `REASONING_EFFORTS`. `None` = optimizer default.
    pub effort: Option<&'static str>,
}

impl Overrides {
    pub const NONE: Overrides = Overrides { ctx: None, quant: None, effort: None };
}

/// Supported `reasoning_effort` levels (ascending) accepted by llama-server's
/// chat-template kwarg handling. A model's template either honors them (mapping
/// each to a reasoning-token budget) or silently ignores them, so the TUI shows
/// all of these and the default is picked per model family.
pub const REASONING_EFFORTS: [&'static str; 6] =
    ["minimal", "low", "medium", "high", "xhigh", "max"];

/// Default reasoning effort when the user has not touched the knob. The TUI
/// normally passes a model-aware pick via `Overrides.effort`; this is the
/// fallback for `Overrides::NONE` (reference cards / CLI fallback).
pub const DEFAULT_REASONING_EFFORT: &str = "high";

/// The KV-cache quality ladder for a backend, densest-last. Shared by the
/// optimizer (which descends it) and the tweak knob (which walks it left/right).
pub fn quant_ladder(backend: Backend) -> &'static [&'static str] {
    match backend {
        Backend::Cpu => &["q8_0", "q5_0", "q4_1", "iq4_nl"],
        Backend::Cuda | Backend::Vulkan => &[
            "kvarn8", "kvarn6", "kvarn5", "kvarn4", "kvarn3", "kvarn2",
            "q5_0", "q4_1", "iq4_nl",
        ],
    }
}

/// Build launch params for one preset. `over` lets the user override the two
/// tweakable dimensions (context size, KV quant); pass `Overrides::NONE` for
/// the pure optimizer default.
pub fn plan_for(preset: Preset, ip: &Inputs, over: &Overrides) -> LaunchPlan {
    let mut evidence: Vec<String> = Vec::new();
    // Guard unknown memory (probe failure): use 16 GB so the plan stays sane
    // instead of clamping to a 16k ctx — never a fabricated "32 GB".
    let mem_mib = if ip.mem_mib > 0.0 { ip.mem_mib } else { 16384.0 };
    // Model weights: GPU backends offload to VRAM (file ≈ size + small scratch);
    // CPU keeps them in RAM (mmap, file size directly).
    let weights = match ip.backend {
        Backend::Cpu => ip.model_file_bytes as f64,
        Backend::Cuda | Backend::Vulkan => weights_bytes(ip.model_file_bytes),
    };
    let vision_cost = ip.vision_mib * 1024.0 * 1024.0;
    let total = mem_mib * 1024.0 * 1024.0;
    let headroom = headroom_mib(mem_mib) * 1024.0 * 1024.0;
    let draft_per = kv_bytes_per_token(DRAFT_QUANT, weights);
    let draft_kv = draft_per * 8192.0; // ~8k draft window
    // Conservative keeps more headroom.
    let extra_reserve = if preset == Preset::Conservative { 768.0 * 1024.0 * 1024.0 } else { 0.0 };
    let avail = (total - headroom - weights - vision_cost - draft_kv - extra_reserve).max(0.0);

    let mem_label = match ip.backend {
        Backend::Cpu => "RAM",
        Backend::Cuda | Backend::Vulkan => "VRAM",
    };
    if ip.vision_mib > 0.0 {
        evidence.push(format!("Vision (mmproj) reserves {:.0} MiB of {mem_label}.", ip.vision_mib));
    }
    evidence.push(format!(
        "{mem_label} {:.0} MiB − weights ~{:.1} GB − headroom {:.0} MiB − draft {:.0} MiB − vision {:.0} MiB → {:.1} GB for KV pool",
        mem_mib, weights / 1e9, headroom / 1e6, draft_kv / 1e6, ip.vision_mib, avail / 1e9
    ));

    // Quant ladder: quality-first on GPU (KVarN); CPU prefers standard quants at
    // q8/q5/q4 to keep decode fast (KVarN is supported but heavier on CPU).
    let quants = quant_ladder(ip.backend);
    // CPU has no VRAM cliff: cap context much lower to keep resp. times sane.
    let max_ctx_backend = match ip.backend {
        Backend::Cpu => 131072,
        Backend::Cuda | Backend::Vulkan => MAX_CTX,
    };
    // Performance presets maximize context: they descend the ladder to the
    // densest quant that is still quality-acceptable (KVarN 3 on GPU, q4_1 on
    // CPU), never stopping early like Conservative does. Conservative keeps the
    // quality-first pick that only needs a safe ~32k pool.
    let min_pool = match preset {
        Preset::Conservative => 32768,
        Preset::Solo => TWIN_POOL,
        Preset::Twin => TWIN_POOL,
    };
    let mut kv_quant = quants[quants.len() - 1]; // densest fallback
    let mut ctx = 16384i64;
    let mut best_ctx = 0i64;

    if preset == Preset::Conservative {
        // Quality-first: first quant from the top that still yields a bulletpool
        // (≥ 32k) wins — the SAFE pick.
        for q in quants {
            let per = kv_bytes_per_token(q, weights);
            let c = ((avail / per) as i64 / ROUND) * ROUND;
            let c = c.clamp(ROUND, max_ctx_backend);
            if c >= min_pool {
                kv_quant = q;
                ctx = c;
                break;
            }
            if c > best_ctx {
                best_ctx = c;
                ctx = c;
                kv_quant = q;
            }
        }
    } else {
        // Solo / Twin: maximize context by descending to the densest
        // quality-acceptable quant. GPU floor = kvarn3, CPU floor = q4_1.
        // If even the floor misses a usable window, keep going denser.
        let floor_quant: &str = match ip.backend {
            Backend::Cpu => "q4_1",
            Backend::Cuda | Backend::Vulkan => "kvarn3",
        };
        let mut candidates: Vec<&str> = quants.to_vec();
        if let Some(pos) = candidates.iter().position(|q| *q == floor_quant) {
            candidates.truncate(pos + 1); // drop quants below the floor
        }
        // Descend from the densest acceptable (last) up to a usable window.
        let mut chosen = candidates[candidates.len() - 1];
        for q in candidates.iter().rev() {
            let per = kv_bytes_per_token(q, weights);
            let c = ((avail / per) as i64 / ROUND) * ROUND;
            let c = c.clamp(ROUND, max_ctx_backend);
            if c >= min_pool {
                chosen = *q;
                ctx = c;
                break;
            }
            if c > best_ctx {
                best_ctx = c;
                ctx = c;
                chosen = *q;
            }
        }
        kv_quant = chosen;
    }
    // Precision tail: keep the newest 1024 tokens exact. GPU only — the tail
    // kernel path is for the quantized-cache (KVarN/standard) GPU decode.
    let mut tail_tokens: i64 = if kv_quant.starts_with("kvarn") && matches!(ip.backend, Backend::Cuda | Backend::Vulkan) {
        1024
    } else {
        0
    };
    let mut kv_tail = kv_tail_bytes(tail_tokens, weights);
    let mut kv_per = kv_bytes_per_token(kv_quant, weights);
    if kv_tail > 0.0 {
        let c = (((avail - kv_tail) / kv_per) as i64 / ROUND) * ROUND;
        ctx = c.clamp(ROUND, max_ctx_backend);
    }

    // ---- manual tweaks (profile knobs) ----
    // User overrides win over the fit-driven default: swap the quant and/or
    // context pool, then re-derive the per-token cost so the hard clamp and the
    // VRAM estimate below both reflect the tweaked plan.
    let mut tweaked = false;
    if let Some(q) = over.quant {
        kv_quant = q;
        kv_per = kv_bytes_per_token(kv_quant, weights);
        tail_tokens = if kv_quant.starts_with("kvarn") && matches!(ip.backend, Backend::Cuda | Backend::Vulkan) { 1024 } else { 0 };
        kv_tail = kv_tail_bytes(tail_tokens, weights);
        tweaked = true;
    }
    if let Some(c) = over.ctx {
        let c = (c / ROUND) * ROUND;
        ctx = c.clamp(ROUND, max_ctx_backend);
        tweaked = true;
    }
    if tweaked {
        evidence.push("Manual profile tweak overrides the default context/quant choice.".into());
    }

    // HARD CLAMP: the plan (weights + headroom + KV + tail + draft + vision)
    // must fit inside real memory minus a hard margin. If the mathematical
    // selection over-claimed (e.g. shared iGPU heaps fooling VRAM, or a model
    // right at the edge), shrink ctx down until it genuinely fits — never ship
    // a plan that OOMs like the old KVarN-8 run did.
    let mut back_off_ctx = ctx;
    loop {
        let check = weights + headroom + kv_per * back_off_ctx as f64 + kv_tail + draft_kv + vision_cost;
        if check <= total - HARD_MARGIN_BYTES || back_off_ctx <= ROUND {
            break;
        }
        back_off_ctx -= ROUND;
    }
    if back_off_ctx < ctx {
        evidence.push(format!(
            "Hard fit: {:.1} GB committed ≥ card budget − 256 MiB; context reduced {ctx} → {back_off_ctx} so the run cannot OOM.",
            (weights + kv_per * ctx as f64 + kv_tail + draft_kv + vision_cost) / 1e9
        ));
        ctx = back_off_ctx;
    }
    evidence.push(format!(
        "Pool quant: {kv_quant} ({:.1} bytes/token) — {ctx} total ctx{}.",
        kv_per,
        if tail_tokens > 0 { format!(", +{tail_tokens}-token precision tail") } else { String::new() }
    ));

    // Slot split per preset.
    let (parallel, per_slot) = match preset {
        Preset::Conservative => (1, ctx),
        Preset::Solo => (1, ctx),
        Preset::Twin => {
            if ctx >= TWIN_POOL {
                (2, ctx / 2)
            } else {
                (1, ctx)
            }
        }
    };
    let ctx_used = (parallel * per_slot).max(RUNROUND);
    evidence.push(format!(
        "Slots: {parallel}{} sharing the unified pool (~{per_slot}/slot).",
        if parallel > 1 { " concurrent" } else { "" }
    ));

// Batch / ubatch: scaled by the detected memory pool (bigger cards → bigger
// prefill batches), with a backend flavor. Base batch = pool/4GiB rounded down
// to a power-of-two bucket, times a backend coefficient.
    let mem_gb = (mem_mib / 1024.0).max(4.0);
    let bucket = |g: f64| -> i64 {
        let b = (mem_gb / g).floor() as i64;
        match b {
            0 => 256,
            1 => 512,
            2 => 1024,
            3 => 2048,
            4 => 2048,
            _ => 4096,
        }
    };
    let base_batch = match ip.backend {
        Backend::Cuda => bucket(4.0),   // CUDA prefill is cheapest → biggest
        Backend::Vulkan => bucket(6.0), // Vulkan mid
        Backend::Cpu => bucket(8.0),    // CPU smallest, avoids RAM thrash
    };
    let (batch, ubatch) = if preset == Preset::Conservative {
        (base_batch, (base_batch / 8).max(128))
    } else {
        // Sprint presets: batch high, but ubatch capped at 256 so the prefill
        // compute buffer doesn't itself become a VRAM hog (that, plus MTP, was
        // what OOM'd a 22.5 GB card at ubatch 512).
        (base_batch * 2, (base_batch / 4).clamp(128, 256))
    };
// Threads: always the runtime physical/logical core count. llama-server uses
// the CPU for prompt/offload work even on GPU backends, so oversaturating at a
// guessed "24" would tank tokens/s on any other machine (beellama perf tips).
    let threads = ip.cpu_threads();
    let use_mtp = true; // MTP speculation on everywhere — it's pure win, off only slows

    let mut args: Vec<String> = Vec::new();
    push(&mut args, "--ctx-size", ctx_used);
    push(&mut args, "--n-gpu-layers", ip.backend.gpu_layers());
    push(&mut args, "--parallel", parallel);
    push(&mut args, "--flash-attn", if ip.backend.flash_attn() { "on" } else { "off" });
    push(&mut args, "--fit", "off");
    args.push("--kv-unified".into());
    push(&mut args, "--cache-type-k", kv_quant);
    push(&mut args, "--cache-type-v", kv_quant);
    // BeeLlama precision tail: newest tokens kept exact while the body is
    // quantized. Filtered by server.rs when the binary doesn't support it.
    if tail_tokens > 0 {
        push(&mut args, "--kv-tail-tokens", tail_tokens);
    }
    push(&mut args, "--n-predict", 16384);

    match preset {
        Preset::Conservative => {
            push(&mut args, "--reasoning", "on");
            // Safety-first reasoning: a small per-block budget plus tight loop
            // detection so a runaway thinking loop is force-closed, then stopped,
            // long before it eats the whole context. The budget re-arms on each
            // new <think> block, so `--n-predict` remains the total backstop.
            push(&mut args, "--reasoning-budget", 2048);
            push(&mut args, "--reasoning-loop-guard", "force-close");
            push(&mut args, "--reasoning-loop-min-tokens", 96);
            push(&mut args, "--reasoning-loop-window", 512);
            push(&mut args, "--reasoning-loop-min-coverage", 48);
            push(&mut args, "--reasoning-loop-check-interval", 24);
            push(&mut args, "--reasoning-loop-interventions", 2);
            push(&mut args, "--temp", "1.0");
            push(&mut args, "--top-p", "0.9");
            push(&mut args, "--top-k", 40);
            push(&mut args, "--min-p", "0.0");
            push(&mut args, "--presence-penalty", "0.0");
            push(&mut args, "--repeat-penalty", "1.1");
        }
        Preset::Solo | Preset::Twin => {
            // Qwen3.8-flavored quality setup, but runaway-guarded: a moderate
            // per-block budget and aggressive loop detection so long thinking
            // either concludes or is force-closed/stopped well short of the
            // `--n-predict` wall (the source of the old 20+ minute stalls).
            push(&mut args, "--reasoning", "on");
            push(&mut args, "--reasoning-budget", 4096);
            push(&mut args, "--reasoning-loop-guard", "force-close");
            push(&mut args, "--reasoning-loop-min-tokens", 96);
            push(&mut args, "--reasoning-loop-window", 512);
            push(&mut args, "--reasoning-loop-min-coverage", 48);
            push(&mut args, "--reasoning-loop-check-interval", 24);
            push(&mut args, "--reasoning-loop-interventions", 2);
            push(&mut args, "--temp", "1.0");
            push(&mut args, "--top-p", "0.95");
            push(&mut args, "--top-k", 20);
            push(&mut args, "--min-p", "0.0");
            push(&mut args, "--presence-penalty", "0.0");
            push(&mut args, "--repeat-penalty", "1.0");
        }
    }

    // Reasoning effort is a chat-template hint, tunable per profile in the TUI.
    // It selects the reasoning-token budget the template targets (Qwen3 maps
    // each level to a different token amount); it does not change VRAM/speed.
    let effort = over.effort.unwrap_or(DEFAULT_REASONING_EFFORT);
    push(&mut args, "--reasoning-effort", effort);

    if use_mtp {
        args.push("--spec-type".into());
        args.push("draft-mtp".into());
        push(&mut args, "--spec-draft-n-max", 2);
        push(&mut args, "--spec-draft-type-k", DRAFT_QUANT);
        push(&mut args, "--spec-draft-type-v", DRAFT_QUANT);
    }

    push(&mut args, "--batch-size", batch);
    push(&mut args, "--ubatch-size", ubatch);
    push(&mut args, "--threads", threads);
    args.push("--cont-batching".into());
    args.push("--metrics".into());
    args.push("--slots".into());
    args.push("--cache-prompt".into());
    args.push("--cache-idle-slots".into());

    let vr_used_gb = (weights + kv_per * ctx_used as f64 + kv_tail + draft_kv + vision_cost) / 1e9;

    // Estimated decode tok/s — a pre-run guess. BANDWIDTH-BOUND model:
    // tps ≈ effective memory bandwidth / (weights read per token).
    //   dense: reads the whole file each token.
    //   MoE:   reads shared part + only the *used* experts → much faster.
    // After the first chat run, the UI shows the MEASURED value instead (the
    // estimate is only a placeholder until real data exists).
    let bw = crate::server::gpu_bandwidth_gbps();
    let eff = match ip.backend {
        Backend::Cpu => 0.30,
        Backend::Cuda => 0.85,
        Backend::Vulkan => 0.75,
    };
    // moe_active: weight mass read per token (1.0 dense; MoE ≈ experts used).
    let moe_active = if ip.moe_active > 0.0 && ip.moe_active < 1.0 {
        // Experts dominate MoE weight mass; model the rest as a small dense core.
        (0.30 + 0.70 * ip.moe_active).max(0.20)
    } else {
        1.0
    };
    let mut tok_s = if bw > 0.0 && weights > 0.0 {
        bw * 1e9 * eff / (weights * moe_active)
    } else {
        0.0
    };
    // MTP speculative decoding verifies several drafted tokens per model step,
    // so the running rate exceeds the pure bandwidth-bound estimate. The plan
    // always enables MTP; reflect it so the estimate is a fair pre-run guess.
    if use_mtp && tok_s > 0.0 {
        tok_s *= MTP_SPEEDUP;
    }

    let summary = vec![
        ("context".into(), format!("{}", ctx_used)),
        ("slots".into(), parallel.to_string()),
        ("per_slot".into(), per_slot.to_string()),
        ("kv_quant".into(), kv_quant_label(kv_quant)),
        ("draft".into(), spec_label(use_mtp, DRAFT_QUANT)),
        ("effort".into(), effort.to_string()),
        ("tail".into(), format!("{tail_tokens} tok")),
        ("batch".into(), batch.to_string()),
        ("ubatch".into(), ubatch.to_string()),
        ("mtp".into(), if use_mtp { "on".into() } else { "off".into() }),
        ("vram".into(), format!("{:.1} GB", vr_used_gb)),
        ("toks".into(), if tok_s > 0.0 { format!("~{tok_s:.0}/s gen") } else { "?".into() }),
    ];

    LaunchPlan { args, summary, evidence, ctx: ctx_used, parallel, vr_used_gb, tok_s, kv_quant: kv_quant.to_string() }
}

/// One row of the CUDA-vs-Vulkan profile diff shown on the performance cards.
pub struct ProfileRow {
    pub label: &'static str,
    pub cuda: String,
    pub vulkan: String,
    /// Which side gains the advantage here (None = equal).
    pub winner: Option<Backend>,
}

/// Compute CUDA vs Vulkan plans for the same model/VRAM at a given preset and
/// return rows describing only what differs — told as advantages, with the real
/// header driver being the estimated tok/s.
pub fn profile_diff(ip: &Inputs, preset: Preset) -> Vec<ProfileRow> {
    let cuda_ip = Inputs { backend: Backend::Cuda, ..*ip };
    let vk_ip = Inputs { backend: Backend::Vulkan, ..*ip };
    let cuda = plan_for(preset, &cuda_ip, &Overrides::NONE);
    let vk = plan_for(preset, &vk_ip, &Overrides::NONE);

    let sum = |pl: &LaunchPlan, k: &str| -> String {
        pl.summary.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone()).unwrap_or_default()
    };

    let mut rows: Vec<ProfileRow> = Vec::new();

    // The headline: DECODE (generation) speed — memory-bandwidth-bound estimate.
    // Note: chat shows the PREFILL speed from llama-server timing, which is a
    // different phase (compute-bound, typically several × faster). Labeling both
    // "tok/s" without phase confusion is what keeps the UI honest.
    if cuda.tok_s > 0.0 && vk.tok_s > 0.0 {
        rows.push(ProfileRow {
            label: "decode (gen)",
            cuda: format!("~{:.0} tok/s", cuda.tok_s),
            vulkan: format!("~{:.0} tok/s", vk.tok_s),
            winner: Some(if cuda.tok_s >= vk.tok_s { Backend::Cuda } else { Backend::Vulkan }),
        });
    }
    if sum(&cuda, "context") != sum(&vk, "context") {
        rows.push(ProfileRow {
            label: "context",
            cuda: sum(&cuda, "context"),
            vulkan: sum(&vk, "context"),
            winner: if cuda.ctx > vk.ctx { Some(Backend::Cuda) } else { Some(Backend::Vulkan) },
        });
    }
    if sum(&cuda, "kv_quant") != sum(&vk, "kv_quant") {
        rows.push(ProfileRow {
            label: "KV cache",
            cuda: sum(&cuda, "kv_quant"),
            vulkan: sum(&vk, "kv_quant"),
            winner: None,
        });
    }
    if sum(&cuda, "per_slot") != sum(&vk, "per_slot") || cuda.parallel != vk.parallel {
        rows.push(ProfileRow {
            label: "slots",
            cuda: format!("{}× · {}", cuda.parallel, sum(&cuda, "per_slot")),
            vulkan: format!("{}× · {}", vk.parallel, sum(&vk, "per_slot")),
            winner: None,
        });
    }
    if (cuda.vr_used_gb - vk.vr_used_gb).abs() > 0.05 {
        rows.push(ProfileRow {
            label: "memory used",
            cuda: format!("{:.1} GB", cuda.vr_used_gb),
            vulkan: format!("{:.1} GB", vk.vr_used_gb),
            winner: None,
        });
    }
    rows
}

const RUNROUND: i64 = 16384;

/// Push a typed scalar arg into the arg vector.
fn push(v: &mut Vec<String>, flag: &str, value: impl std::fmt::Display) {
    v.push(flag.to_string());
    v.push(value.to_string());
}

/// Friendly label for a KV-cache quant (shows WHAT it is, since "iq4_nl" can be
/// mistaken for the model-weight quant like Q4_K_M).
pub fn kv_quant_label(q: &str) -> String {
    if let Some(n) = q.strip_prefix("kvarn") {
        return format!("KVarN {n} (KV cache)");
    }
    match q {
        "f16" | "bf16" => "f16 (KV cache)".into(),
        "q8_0" => "q8 (KV cache)".into(),
        "q5_0" | "q5_1" => "q5 (KV cache)".into(),
        "q4_0" | "q4_1" | "iq4_nl" => "q4 (KV cache)".into(),
        other => format!("{other} (KV cache)"),
    }
}

/// Label for the speculative-decoding engine the plan launches. This build uses
/// MTP (multi-token prediction) via `--spec-type draft-mtp`, so it's named
/// explicitly rather than the generic "draft"; the value shows its sub-quant.
pub fn spec_label(use_mtp: bool, q: &str) -> String {
    let qshort = match q {
        "f16" | "bf16" => "f16",
        "q8_0" => "q8",
        "q5_0" | "q5_1" => "q5",
        "q4_0" | "q4_1" => "q4",
        other => other,
    };
    if use_mtp {
        format!("MTP ({qshort} KV)")
    } else {
        "off".into()
    }
}