use crate::server::{ServerEvent, ServerProcess};
use crate::chat::Message;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::DefaultTerminal;

mod chat;
mod i18n;
mod optimizer;
mod profiles;
mod server;
mod ui;

pub enum Stage {
    Welcome,
    Language,
    Backend,
    ModelSelect,
    Optimize,
    Chat,
}

/// What the →/← arrows currently target in the Optimize screen: the profile
/// card itself, its context pool, its KV-cache quant, or the reasoning effort
/// knob. ↑/↓ move this focus, ←/→ tweak the focused target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParamFocus {
    Card,
    Ctx,
    Quant,
    Reasoning,
}

impl ParamFocus {
    pub const ALL: [ParamFocus; 4] =
        [ParamFocus::Card, ParamFocus::Ctx, ParamFocus::Quant, ParamFocus::Reasoning];

    pub fn idx(self) -> usize {
        match self {
            ParamFocus::Card => 0,
            ParamFocus::Ctx => 1,
            ParamFocus::Quant => 2,
            ParamFocus::Reasoning => 3,
        }
    }

    pub fn from_idx(i: usize) -> ParamFocus {
        ParamFocus::ALL[i.rem_euclid(ParamFocus::ALL.len())]
    }
}

pub struct App {
    pub stage: Stage,
    pub i18n: i18n::I18n,
    pub lang_index: usize,
    pub backend_index: usize,
    pub models: Vec<server::Model>,
    pub model_index: usize,
    pub status: String,
    pub messages: Vec<Message>,
    pub input: ratatui_textarea::TextArea<'static>,
    pub scrollback: i32,
    pub server_log: Vec<String>,
    pub server: Option<ServerProcess>,
    pub autopick: bool,
    pub profile: profiles::Profile,
    pub fleet_count: u32,
    // Custom model folder input (shown when adding a directory).
    pub add_folder_active: bool,
    pub add_folder_path: String,
    pub add_folder_cursor: usize,
    // ---- features ----
    pub show_help: bool,
    pub chat_history: Vec<String>,   // previous user prompts, for ↑/↓ recall
    pub history_index: Option<usize>,
    pub auto_scroll: bool,
    pub theme_inverted: bool,
    pub copies: Vec<String>,
// ---- draggable panes ----
    pub chat_ratio: f32,        // 0..1: chat vs log split
    pub log_scroll: usize,      // scroll offset into the log (lines)
    pub log_autoscroll: bool,
    pub log_tallest: usize,
    // ---- splash animation ----
    pub splash_frame: usize,
    // ---- last optimizer plans (shown in Optimize stage) ----
    pub last_plan: Option<crate::optimizer::LaunchPlan>,
    pub preset_plans: Vec<crate::optimizer::LaunchPlan>,
    pub preset_index: usize,
    pub launch_started: bool,
    // ---- profile tweak knobs (Optimize stage) ----
    /// Which dimension the ←/→ keys tweak (profile card, ctx, or KV quant).
    pub param_focus: ParamFocus,
    /// Manual context override (multiples of 16k); None = optimizer default.
    pub over_ctx: Option<i64>,
    /// Manual KV-quant override (from the backend ladder); None = auto pick.
    pub over_quant: Option<&'static str>,
    /// Manual reasoning-effort override (from `REASONING_EFFORTS`); None =
    /// auto pick (model-aware default, e.g. xhigh for Qwen3.x, else high).
    pub over_effort: Option<&'static str>,
    // ---- vision choice ----
    pub use_vision: bool,
    pub has_vision: bool, // selected model carries an mmproj
// ---- speed metrics ----
    pub last_tps: f64,          // tokens per second of last completed response
    gen_tokens: u64,            // real SSE token count of the in-flight response
    gen_started_at: Option<std::time::Instant>, // first-token time (decode only)
    msg_start: std::time::Instant,
    custom_dirs: Vec<String>,
    model_path: String,
    // Streaming assistant deltas arrive on a background thread; drained in poll.
    handle_delta_rx: Option<(usize, std::sync::mpsc::Receiver<String>)>,
    // Mouse support (buttons + hover) — capture is enabled at startup.
    pub mouse_pos: Option<(u16, u16)>,
    /// Rect + action for each clickable chat button, set by the renderer each
    /// frame so the event loop can hit-test clicks against them.
    pub chat_buttons: Vec<(ratatui::layout::Rect, ChatAction)>,
}

/// Click target rendered in the chat button bar.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChatAction {
    NewModel,
    Theme,
    Autoscroll,
    Help,
    Back,
    Quit,
}

impl App {
    pub fn new(autopick: bool, model_path: Option<String>) -> Self {
        let custom_dirs = Vec::new();
        let models = discover_with(&custom_dirs);
        let mut app = App {
            stage: if autopick { Stage::Chat } else { Stage::Welcome },
            i18n: i18n::I18n::new(i18n::Lang::En),
            lang_index: 0,
            backend_index: 0,
            models,
            model_index: 0,
            status: "choose a model".into(),
            messages: Vec::new(),
            input: ratatui_textarea::TextArea::default(),
            scrollback: 0,
            server_log: Vec::new(),
            server: None,
            autopick,
            profile: profiles::Profile::Balanced,
            fleet_count: 1,
            add_folder_active: false,
            add_folder_path: String::new(),
            add_folder_cursor: 0,
            show_help: false,
            chat_history: Vec::new(),
            history_index: None,
            auto_scroll: true,
            theme_inverted: false,
            copies: Vec::new(),
            chat_ratio: 0.62,
            log_scroll: 0,
            log_autoscroll: true,
            log_tallest: 0,
            splash_frame: 0,
            last_plan: None,
            preset_plans: Vec::new(),
            preset_index: 0,
            launch_started: false,
            param_focus: ParamFocus::Card,
            over_ctx: None,
            over_quant: None,
            over_effort: None,
            use_vision: false,
            has_vision: false,
            last_tps: 0.0,
            gen_tokens: 0,
            gen_started_at: None,
            msg_start: std::time::Instant::now(),
            custom_dirs,
            model_path: String::new(),
            handle_delta_rx: None,
            mouse_pos: None,
            chat_buttons: Vec::new(),
        };
        if let Some(p) = model_path {
            app.model_path = p;
            app.stage = Stage::Chat;
        }
        app
    }

    pub fn t(&self, key: &str) -> String {
        self.i18n.t(key)
    }

    /// Start llama-server on the given port for the currently selected model.
    pub fn start_server(&mut self, model_path: String, port: u16) -> anyhow::Result<()> {
        if self.server.is_some() {
            return Ok(());
        }
        // The all-backend runtime (CUDA+Vulkan+CPU) now ships INSIDE dist-llama-lab
        // next to llama-lab.exe, so the side-side binary is the primary source.
        // LLAMA_SERVER_EXE override wins; the standalone BeeLlama build dir is a
        // convenience fallback. Capability probing in server.rs drops flags the
        // resolved binary doesn't support (e.g. --reasoning-loop-guard).
        let bee_llama = "D:\\_SATIN_AI_2\\beellama\\build-x64-windows-cuda-vulkan-release\\bin\\llama-server.exe";
        let side = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("llama-server.exe")))
            .filter(|p| p.exists());
        let exe = std::env::var("LLAMA_SERVER_EXE")
            .ok()
            .filter(|p| std::path::Path::new(p).exists())
            .or_else(|| side.map(|p| p.to_string_lossy().into_owned()))
            .or_else(|| std::path::Path::new(bee_llama).exists().then_some(bee_llama.to_string()))
            .unwrap_or_else(|| bee_llama.to_string());
        if !std::path::Path::new(&exe).exists() {
            return Err(anyhow::anyhow!("llama-server.exe not found: {exe}"));
        }
        // If the selected model has an mmproj AND the user chose vision, pass it.
        let mmproj = if self.use_vision {
            self.models.get(self.model_index).and_then(|m| m.mmproj.clone())
        } else {
            None
        };
        let plan = self
            .preset_plans
            .get(self.preset_index)
            .cloned()
            .or_else(|| {
                // Fall back to a recomputed plan if presets were never built.
                let bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
                Some(crate::optimizer::plan_for(
                    crate::optimizer::Preset::Solo,
                    &crate::optimizer::Inputs {
                        backend: self.backend(),
                        mem_mib: server::detect_mem_mib(self.backend()),
                        model_file_bytes: bytes,
                        vision_mib: self.vision_budget_mib(),
                        moe_active: moe_active_for(&model_path),
                    },
                    &crate::optimizer::Overrides::NONE,
                ))
            })
            .expect("plan");
        let proc = server::ServerProcess::start(&model_path, port, &exe, mmproj.as_deref(), plan)?;
        self.server = Some(proc);
        self.status = self.i18n.t("status_starting");
        Ok(())
    }

    /// Draft the model_path that will be used (selected model or explicit).
    pub fn selected_model_path(&self) -> String {
        if !self.model_path.is_empty() {
            return self.model_path.clone();
        }
        self.models
            .get(self.model_index)
            .map(|m| m.path.clone())
            .unwrap_or_default()
    }

    /// Re-scan models after adding a custom folder.
    pub fn rescam_models(&mut self) {
        self.models = discover_with(&self.custom_dirs);
    }

    /// After a language is chosen, ask for the compute backend next.
    fn next_after_lang(&mut self) {
        self.stage = if self.autopick { Stage::Chat } else { Stage::Backend };
    }

    /// Current backend selected in the slides (index into available_backends()).
    pub fn backend(&self) -> crate::optimizer::Backend {
        let avail = self.available_backends();
        avail[self.backend_index.min(avail.len() - 1)]
    }

    /// Backends usable on this machine (CUDA only on NVIDIA).
    pub fn available_backends(&self) -> Vec<crate::optimizer::Backend> {
        let mut v = vec![crate::optimizer::Backend::Cpu, crate::optimizer::Backend::Vulkan];
        if server::supports_cuda() {
            v.insert(1, crate::optimizer::Backend::Cuda);
        }
        v
    }

    /// (Re)compute the three optimized preset plans for the selected model +
    /// vision choice. The currently-selected profile picks up the manual tweak
    /// overrides (ctx/quant); the other two stay at their natural tuning so they
    /// remain meaningful alternatives. Set the active plan to the selected one.
    fn recompute_plan(&mut self) {
        let path = self.selected_model_path();
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let vision = self.vision_budget_mib();
        let ip = crate::optimizer::Inputs {
            backend: self.backend(),
            mem_mib: server::detect_mem_mib(self.backend()),
            model_file_bytes: bytes,
            vision_mib: vision,
            moe_active: moe_active_for(&path),
        };
        let over = self.overrides();
        self.preset_plans = (0..3)
            .map(|i| {
                let p = crate::optimizer::Preset::ALL[i];
                let o = if i == self.preset_index { &over } else { &crate::optimizer::Overrides::NONE };
                crate::optimizer::plan_for(p, &ip, o)
            })
            .collect();
        self.preset_index = self.preset_index.min(2);
        self.last_plan = self.preset_plans.get(self.preset_index).cloned();
    }

    /// The manual tweak overrides (as the optimizer consumes them). The selected
    /// profile always carries an explicit reasoning effort so the launch and the
    /// cards agree on what is being run.
    fn overrides(&self) -> crate::optimizer::Overrides {
        crate::optimizer::Overrides {
            ctx: self.over_ctx,
            quant: self.over_quant,
            effort: Some(self.effective_effort()),
        }
    }

    /// Whether the selected model is a Qwen 3.x reasoning model (detected from
    /// its file name, e.g. "Qwen3.8-27B-Q4_K_M.gguf"). Qwen 3 templates map
    /// each `reasoning_effort` level only to a reasoning-token budget — there is
    /// no other behavioral difference — so we surface a hint and pick a high
    /// default effort for them.
    fn is_qwen_3x(&self) -> bool {
        let name = self
            .models
            .get(self.model_index)
            .map(|m| m.name.to_lowercase())
            .unwrap_or_default();
        name.contains("qwen") && (name.contains("3.8") || name.contains("3.") || name.contains("3 "))
    }

    /// Model-aware default reasoning effort: Qwen 3.x family favours `xhigh`
    /// (it only spends more on reasoning tokens at that level), everything else
    /// defaults to `high`. The user can still cycle the whole ladder.
    fn recommended_effort(&self) -> &'static str {
        if self.is_qwen_3x() {
            "xhigh"
        } else {
            "high"
        }
    }

    /// The reasoning effort currently in effect for the selected profile
    /// (manual override, else the model-aware default).
    fn effective_effort(&self) -> &'static str {
        if let Some(e) = self.over_effort {
            return e;
        }
        self.recommended_effort()
    }

    /// The currently-in-effect context pool for the selected profile
    /// (manual override, else the plan's computed value).
    fn effective_ctx(&self) -> i64 {
        if let Some(c) = self.over_ctx {
            return c;
        }
        self.preset_plans.get(self.preset_index).map(|p| p.ctx).unwrap_or(crate::optimizer::CTX_STEP)
    }

    /// The currently-in-effect KV quant for the selected profile.
    fn effective_quant(&self) -> String {
        if let Some(q) = self.over_quant {
            return q.to_string();
        }
        self.preset_plans
            .get(self.preset_index)
            .map(|p| p.kv_quant.clone())
            .unwrap_or_else(|| "q5_0".to_string())
    }

    /// Tweak the focused dimension left (−1) or right (+1). Context moves by a
    /// full 16k step; quant walks the backend ladder; the card focus cycles the
    /// profile (resetting manual tweaks so the new card shows its own tuning).
    /// Every tweak recomputes the plan so the VRAM estimate stays truthful.
    fn tweak_focus(&mut self, dir: i64) {
        match self.param_focus {
            ParamFocus::Card => {
                let n = self.preset_plans.len();
                if n == 0 {
                    return;
                }
                self.preset_index = (self.preset_index as i64 + dir).rem_euclid(n as i64) as usize;
                self.over_ctx = None;
                self.over_quant = None;
                self.over_effort = None;
                self.recompute_plan();
            }
            ParamFocus::Ctx => {
                let step = crate::optimizer::CTX_STEP;
                let base = self.effective_ctx();
                let new = ((base + dir * step) / step) * step;
                self.over_ctx = Some(new.clamp(step, crate::optimizer::MAX_CTX));
                self.recompute_plan();
            }
            ParamFocus::Quant => {
                let ladder = crate::optimizer::quant_ladder(self.backend());
                let cur = self.effective_quant();
                let idx = ladder.iter().position(|q| *q == cur.as_str()).unwrap_or(0);
                let next = ((idx as i64 + dir).rem_euclid(ladder.len() as i64)) as usize;
                self.over_quant = Some(ladder[next]);
                self.recompute_plan();
            }
            ParamFocus::Reasoning => {
                let ladder = crate::optimizer::REASONING_EFFORTS;
                let cur = self.effective_effort();
                let idx = ladder.iter().position(|e| *e == cur).unwrap_or(0);
                let next = ((idx as i64 + dir).rem_euclid(ladder.len() as i64)) as usize;
                self.over_effort = Some(ladder[next]);
                self.recompute_plan();
            }
        }
    }

    /// Move the ↑/↓ parameter focus (Card → Ctx → Quant → Card).
    fn cycle_focus(&mut self, dir: i64) {
        let i = (self.param_focus.idx() as i64 + dir).rem_euclid(ParamFocus::ALL.len() as i64);
        self.param_focus = ParamFocus::from_idx(i as usize);
    }

    /// Projector VRAM footprint of the selected model's mmproj (+10 % load
    /// overhead). Independent of `use_vision` so both alternatives can be shown.
    pub fn mmproj_budget_mib(&self) -> f64 {
        let mm = self.models.get(self.model_index).and_then(|m| m.mmproj.clone());
        match mm {
            Some(p) => {
                let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                (bytes as f64 * 1.10) / (1024.0 * 1024.0)
            }
            None => 0.0,
        }
    }

    pub fn vision_budget_mib(&self) -> f64 {
        if self.use_vision {
            self.mmproj_budget_mib()
        } else {
            0.0
        }
    }

    /// Send a user message: record history, spawn streaming against the server.
    pub fn send_message(&mut self, msg: String) {
        if msg.trim().is_empty() {
            return;
        }
        if msg.trim_start().starts_with('/') {
            self.run_command(&msg);
            return;
        }
        // History for the model = all prior messages (excludes the new user msg).
        let history = self.messages.clone();
        self.messages.push(Message { role: "user".into(), content: msg.clone() });
        self.chat_history.push(msg.clone());
        while self.chat_history.len() > 100 {
            self.chat_history.remove(0);
        }
        self.history_index = None;
        self.status = self.i18n.t("status_thinking");
        self.auto_scroll = true;
        self.msg_start = std::time::Instant::now();
        self.gen_tokens = 0;
        self.gen_started_at = None;
        let port = PORT;
        let assistant_index = self.messages.len();
        self.messages.push(Message { role: "assistant".into(), content: String::new() });
        let (delta_tx, delta_rx) = std::sync::mpsc::channel::<String>();
        let tx_delta = delta_tx.clone();
        let tx_done = delta_tx.clone();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let _ = runtime.block_on(chat::stream_chat(port, &history, &msg, move |t| {
                let _ = tx_delta.send(t.to_string());
            }));
            let _ = tx_done.send("\u{0}DONE".into());
        });
        self.handle_delta_rx = Some((assistant_index, delta_rx));
    }

    /// Handle slash commands (local UI actions; no network).
    fn run_command(&mut self, cmd: &str) {
        let c = cmd.trim();
        match c {
            "/help" | "/?" => self.show_help = !self.show_help,
            "/clear" | "/cls" => {
                self.messages.clear();
                self.server_log.clear();
            }
            "/quit" | "/exit" => std::process::exit(0),
            "/theme" => self.theme_inverted = !self.theme_inverted,
            "/autoscroll" => {
                self.auto_scroll = !self.auto_scroll;
                self.copies.push("autoscroll toggled".into());
            }
            "/copy" => {
                if let Some(last) = self.messages.last() {
                    self.copies.push(last.content.clone());
                }
            }
            _ => {
                if cmd.starts_with('/') {
                    self.status = format!("{} {}", self.i18n.t("unknown_command"), cmd);
                }
            }
        }
    }
}

fn discover_with(custom_dirs: &[String]) -> Vec<server::Model> {
    let mut models = server::discover_models();
    for d in custom_dirs {
        let extra = server::discover_dir(d);
        for m in extra {
            if !models.iter().any(|x| x.path == m.path) {
                models.push(m);
            }
        }
    }
    models
}

/// Convert a char-count index into the byte offset for String::insert/remove.
fn byte_at_char_index(input: &str, char_idx: usize) -> usize {
    input
        .char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(input.len())
}

pub const PORT: u16 = 8888;

/// Fraction of model weight mass read per generated token: 1.0 for dense
/// models; for MoE, the ratio of *used* experts (e.g. 8/128 ≈ 0.06) since only
/// those are fetched each step. Read once from GGUF metadata.
fn moe_active_for(path: &str) -> f64 {
    match server::gguf_moe_info(path) {
        Some(m) if m.expert_count > 0 => {
            let used = if m.expert_used > 0 { m.expert_used } else { 1 };
            (used as f64 / m.expert_count as f64).clamp(0.05, 1.0)
        }
        _ => 1.0,
    }
}

/// Replace a tui-textarea's content with `s` (single line). tui-textarea 0.7
/// has no set_lines/clear, so: move to start, delete forward all, insert.
fn textarea_set(ta: &mut ratatui_textarea::TextArea, s: &str) {
    ta.move_cursor(ratatui_textarea::CursorMove::Head);
    let _ = ta.delete_str(usize::MAX);
    ta.insert_str(s);
}

fn main() -> anyhow::Result<()> {
let autopick = std::env::var("LLAMA_TUI_AUTOPICK").is_ok();
    let model_env = std::env::var("LLAMA_TUI_MODEL").ok();

    let mut app = App::new(autopick, model_env);

    let mut terminal = ratatui::init();

    // Enable enhanced keyboard reporting + bracketed paste. This is the hook
    // that makes Ctrl+letter combos reliably deliver the CONTROL modifier (or a
    // proper escape sequence) from Windows Terminal / ConPTY — without it the
    // terminal may swallow or mis-report Ctrl combinations, which is exactly why
    // chat shortcuts "never worked."
    use ratatui::crossterm::event::{
        EnableMouseCapture, DisableMouseCapture, KeyboardEnhancementFlags, PushKeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    };
    use ratatui::crossterm::execute;
    // Mouse capture: when OFF, a click in Windows Terminal starts a text
    // "selection" that swallows keyboard input — that's why typing died after
    // clicking. On: mouse events go to the app (consumed, harmless) and focus
    // stays in the chat input.
    let _ = execute!(std::io::stdout(), EnableMouseCapture);
    let kb_ok = execute!(
        std::io::stdout(),
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        )
    )
    .is_ok();

    let result = run(&mut terminal, &mut app);

    let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    let _ = kb_ok;
    result
}

fn run(terminal: &mut DefaultTerminal, app: &mut App) -> anyhow::Result<()> {
    loop {
        poll_and_handle(app)?;
        terminal.draw(|f| ui::render(f, app))?;
    }
    // Drop app (kills server via ServerProcess::drop) happens on process exit.
}

fn poll_and_handle(app: &mut App) -> anyhow::Result<()> {
    // Splash / launch animations advance on a timer regardless of input.
    if matches!(app.stage, Stage::Welcome | Stage::Optimize) {
        app.splash_frame += 1;
    }
    if event::poll(std::time::Duration::from_millis(100))? {
        match event::read()? {
            Event::Key(key) => {
                if key.kind == KeyEventKind::Release {
                    return Ok(());
                }
                handle_key(app, key)?;
            }
            // Mouse: track position for hover, and on a Left click hit-test the chat
            // buttons rendered last frame (mouse capture is ON, so clicks come
            // to us instead of stealing focus).
            Event::Mouse(m) => {
                use ratatui::crossterm::event::MouseEventKind;
                use ratatui::crossterm::event::MouseButton;
                app.mouse_pos = Some((m.column, m.row));
                match m.kind {
                    MouseEventKind::Down(MouseButton::Left) | MouseEventKind::Up(MouseButton::Left) => {
                        // Fire on Up so the drag-select between down/up is safe.
                        if matches!(m.kind, MouseEventKind::Up(_)) {
                            let pt = (m.column, m.row);
                            if let Some((_, action)) = app
                                .chat_buttons
                                .iter()
                                .find(|(r, _)| pt.0 >= r.x && pt.0 < r.x + r.width && pt.1 >= r.y && pt.1 < r.y + r.height)
                            {
                                match *action {
                                    ChatAction::NewModel => { stop_server(app); app.stage = Stage::ModelSelect; }
                                    ChatAction::Theme => app.theme_inverted = !app.theme_inverted,
                                    ChatAction::Autoscroll => { app.log_autoscroll = true; app.log_scroll = 0; }
                                    ChatAction::Help => app.show_help = true,
                                    ChatAction::Back => {
                                        stop_server(app);
                                        app.stage = Stage::ModelSelect;
                                        textarea_set(&mut app.input, "");
                                        app.history_index = None;
                                    }
                                    ChatAction::Quit => std::process::exit(0),
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Focus / paste / resize: consumed and ignored. Must still be read
            // or the terminal queue backs up and raw mode wedges.
            Event::FocusGained
            | Event::FocusLost
            | Event::Paste(_)
            | Event::Resize(_, _) => {}
        }
    }
    // Drain streaming assistant deltas (from the chat background thread).
    if let Some((idx, rx)) = app.handle_delta_rx.take() {
        let mut finished = false;
while let Ok(tok) = rx.try_recv() {
            if tok == "\u{0}DONE" {
                finished = true;
            } else if let Some(m) = app.messages.get_mut(idx) {
                m.content.push_str(&tok);
                app.gen_tokens += 1;
                if app.gen_started_at.is_none() {
                    app.gen_started_at = Some(std::time::Instant::now());
                }
            }
        }
        if !finished {
            app.handle_delta_rx = Some((idx, rx));
} else {
            app.status = "ready".into();
            // Speed = REAL SSE token count / decode-only time (excludes the
            // prompt-eval/prefill phase, which used to drag tok/s to "5.9").
            if let (Some(start), true) = (app.gen_started_at, app.gen_tokens > 0) {
                let dur = start.elapsed().as_secs_f64().max(0.05);
                app.last_tps = app.gen_tokens as f64 / dur;
            }
        }
    }

    // Drain server events (logs, health).
    if let Some(srv) = app.server.as_ref() {
        while let Some(ev) = srv.try_recv() {
            match ev {
                ServerEvent::LogLine(l) => {
                    app.server_log.push(l);
                    if app.server_log.len() > 2000 {
                        app.server_log.drain(0..(app.server_log.len() - 2000));
                    }
                }
ServerEvent::Healthy(ok) => {
                    if ok {
                        app.status = "ready".into();
                        if matches!(app.stage, Stage::Optimize) {
                            app.stage = Stage::Chat;
                        }
                    }
                }
ServerEvent::Exit(code) => {
                    app.status = format!("{} {})", app.i18n.t("server_exited"), code);
                }
            }
        }
    }
    Ok(())
}

fn handle_key(app: &mut App, key: ratatui::crossterm::event::KeyEvent) -> anyhow::Result<()> {
    // Normalize Ctrl+letter across both delivery styles: Windows terminals send
    // Ctrl+T either as Char('t')+CONTROL modifier or as the raw byte 0x14 (no
    // modifier). Ctrl+<letter> is the only shortcut surface, so we derive a
    // letter here and match on it everywhere.
    let mod_ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let ctrl_key: Option<char> = match key.code {
        KeyCode::Char(c) if mod_ctrl => Some(c.to_ascii_lowercase()),
        KeyCode::Char(c) if (c as u32) >= 1 && (c as u32) <= 26 => Some((((c as u32) + 96) as u8) as char),
        _ => None,
    };

    // Global quit: Ctrl+Q (either delivery).
    if ctrl_key == Some('q') {
        std::process::exit(0);
    }
match app.stage {
        Stage::Welcome => {
            match key.code {
                KeyCode::Enter | KeyCode::Char(' ') => {
                    app.stage = if app.autopick { Stage::Chat } else { Stage::Language };
                }
                KeyCode::Esc => std::process::exit(0),
                _ => {}
            }
        }
        Stage::Language => {
            match key.code {
                KeyCode::Down | KeyCode::Char('j') => app.lang_index = (app.lang_index + 1) % 3,
                KeyCode::Up | KeyCode::Char('k') => app.lang_index = (app.lang_index + 2) % 3,
                KeyCode::Enter => {
                    app.i18n.set_lang(i18n::Lang::from_index(app.lang_index));
                    app.next_after_lang();
                }
                KeyCode::Backspace | KeyCode::Esc => app.stage = Stage::Welcome,
                _ => {}
            }
        }
        Stage::Backend => {
            let n = app.available_backends().len().max(1);
            match key.code {
                KeyCode::Right | KeyCode::Char('l') => app.backend_index = (app.backend_index + 1) % n,
                KeyCode::Left | KeyCode::Char('h') => app.backend_index = (app.backend_index + n - 1) % n,
                KeyCode::Enter => {
                    app.stage = Stage::ModelSelect;
                }
                KeyCode::Backspace | KeyCode::Esc => app.stage = Stage::Language,
                _ => {}
            }
        }
        Stage::ModelSelect => {
            if app.add_folder_active {
                // Editing the "add folder" path field.
                match key.code {
                    KeyCode::Char(c) => {
                        if !key.modifiers.contains(KeyModifiers::CONTROL) {
                            let code = c as u32;
                            if code >= 0x20 && code != 0x7f {
                                let b = byte_at_char_index(&app.add_folder_path, app.add_folder_cursor);
                                app.add_folder_path.insert(b, c);
                                app.add_folder_cursor += 1;
                            }
                        }
                    }
                    KeyCode::Backspace => {
                        if app.add_folder_cursor > 0 {
                            let b = byte_at_char_index(&app.add_folder_path, app.add_folder_cursor - 1);
                            app.add_folder_path.remove(b);
                            app.add_folder_cursor -= 1;
                        }
                    }
                    KeyCode::Left => app.add_folder_cursor = app.add_folder_cursor.saturating_sub(1),
                    KeyCode::Right => {
                        if app.add_folder_cursor < app.add_folder_path.chars().count() {
                            app.add_folder_cursor += 1;
                        }
                    }
                    KeyCode::Enter | KeyCode::Esc => {
                        let dir = app.add_folder_path.trim().to_string();
                        if !dir.is_empty() {
                            app.custom_dirs.push(dir.clone());
                            app.rescam_models();
                        }
                        app.add_folder_active = false;
                        app.add_folder_path.clear();
                        app.add_folder_cursor = 0;
                    }
                    _ => {}
                }
                return Ok(());
            }
            match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    if !app.models.is_empty() {
                        app.model_index = (app.model_index + 1) % app.models.len();
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if !app.models.is_empty() {
                        app.model_index = (app.model_index + app.models.len() - 1) % app.models.len();
                    }
                }
                KeyCode::Char('+') => {
                    app.add_folder_active = true;
                }
                KeyCode::Char('p') | KeyCode::Char('P') => {
                    app.profile = profiles::cycle(app.profile);
                }
                KeyCode::Char('f') | KeyCode::Char('F') => {
                    // cycle fleet slots 1..4
                    app.fleet_count = (app.fleet_count % 4) + 1;
                }
                KeyCode::Enter => {
                    let path = app.selected_model_path();
                    if path.is_empty() {
                        app.status = app.t("status_error");
                    } else {
                        app.status = app.t("status_starting");
                        // Don't spawn yet — Optimize lets the user pick
                        // vision/no-vision and shows the parameter plan first.
                        app.has_vision = app
                            .models
                            .get(app.model_index)
                            .map(|m| m.mmproj.is_some())
                            .unwrap_or(false);
                        app.use_vision = false; // default: max context (no vision)
                        app.recompute_plan();
                        app.launch_started = true;
                        app.stage = Stage::Optimize;
                    }
                }
                KeyCode::Backspace | KeyCode::Esc => app.stage = Stage::Backend,
                _ => {}
            }
        }
Stage::Optimize => {
            // Focus model: ↑/↓ move between tweakable dimensions (Card / Ctx /
            // Quant); ←/→ tweak the focused one (context by ±16k, quant walks
            // the ladder, card picks the profile). Space toggles vision (the
            // model must support it); Enter launches. Esc/Backspace go back.
            match key.code {
                KeyCode::Enter => {
                    if !app.server.is_some() {
                        let path = app.selected_model_path();
                        if let Err(e) = app.start_server(path, PORT) {
                            app.status = format!("{}: {e}", app.t("status_error"));
                        } else {
                            app.status = app.t("status_starting");
                        }
                    }
                }
                KeyCode::Char(' ') => {
                    if app.has_vision {
                        app.use_vision = !app.use_vision;
                        app.recompute_plan();
                    }
                }
                KeyCode::Backspace | KeyCode::Esc => {
                    app.stage = Stage::ModelSelect;
                    app.server_log.clear();
                }
                KeyCode::Up | KeyCode::Char('k') => app.cycle_focus(-1),
                KeyCode::Down | KeyCode::Char('j') => app.cycle_focus(1),
                KeyCode::Left | KeyCode::Char('h') => app.tweak_focus(-1),
                KeyCode::Right | KeyCode::Char('l') => app.tweak_focus(1),
                KeyCode::PageUp => {
                    app.log_autoscroll = false;
                    app.log_scroll = app.log_scroll.saturating_add(10);
                }
                KeyCode::PageDown => {
                    app.log_scroll = app.log_scroll.saturating_sub(10);
                    if app.log_scroll == 0 {
                        app.log_autoscroll = true;
                    }
                }
                _ => {}
            }
        }
Stage::Chat => {
            // ---- shortcuts FIRST ----
            // Ctrl+letter comes in two flavors on Windows: the proper
            // Char+CONTROL (reliable with keyboard enhancement ON) OR the raw
            // control byte with NO modifier (Console host / legacy terminals):
            //   0x0E = Ctrl+N · 0x11 = Ctrl+Q · 0x14 = Ctrl+T · 0x01 = Ctrl+A
            // We match BOTH, but ONLY for exact shortcut bytes — CR/LF/TAB/ESC
            // are handled after, so Enter can never collide again.
            if mod_ctrl {
                match key.code {
                    KeyCode::Char('t' | 'T') => { app.theme_inverted = !app.theme_inverted; return Ok(()); }
                    KeyCode::Char('a' | 'A') => { app.log_autoscroll = true; app.log_scroll = 0; return Ok(()); }
                    KeyCode::Char('n' | 'N') => { stop_server(app); app.stage = Stage::ModelSelect; return Ok(()); }
                    KeyCode::Char('q' | 'Q') => std::process::exit(0),
                    KeyCode::Char('=' | '+') => { app.chat_ratio = (app.chat_ratio + 0.05).min(0.95); return Ok(()); }
                    KeyCode::Char('-' | '_') => { app.chat_ratio = (app.chat_ratio - 0.05).max(0.20); app.log_autoscroll = false; return Ok(()); }
                    KeyCode::Char('?') => { app.show_help = true; return Ok(()); }
                    _ => {}
                }
            }
            // Raw control-byte shortcuts: 0x01 'a', 0x0E 'n', 0x11 'q', 0x14 't'.
            let ctrl_byte = match key.code {
                KeyCode::Char(c) => match c as u32 {
                    0x01 => Some('a'),
                    0x0E => Some('n'),
                    0x11 => Some('q'),
                    0x14 => Some('t'),
                    _ => None,
                },
                _ => None,
            };
            if let Some(k) = ctrl_byte {
                match k {
                    'a' => { app.log_autoscroll = true; app.log_scroll = 0; return Ok(()); }
                    'n' => { stop_server(app); app.stage = Stage::ModelSelect; return Ok(()); }
                    'q' => std::process::exit(0),
                    't' => { app.theme_inverted = !app.theme_inverted; return Ok(()); }
                    _ => {}
                }
            }
            // Escape — some terminals send byte 0x1B as Char instead of KeyCode.
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('\u{1b}')) {
                if app.show_help {
                    app.show_help = false;
                } else {
                    stop_server(app);
                    app.stage = Stage::ModelSelect;
                    textarea_set(&mut app.input, "");
                    app.history_index = None;
                }
                return Ok(());
            }
            // Enter — accept all three delivery forms (KeyCode, CR, LF). This was
            // the old bug: CR/LF bytes fell into the "unknown control" swallow.
            let is_enter = matches!(key.code, KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n'));
            // Up / Down — prompt history (checked before feeding the textarea).
            if matches!(key.code, KeyCode::Up) {
                let n = app.chat_history.len();
                if n > 0 {
                    let cur = app.history_index.unwrap_or(n);
                    let next = cur.saturating_sub(1);
                    if let Some(p) = app.chat_history.get(next) {
                        textarea_set(&mut app.input, p);
                        app.input.move_cursor(ratatui_textarea::CursorMove::End);
                        app.history_index = Some(next);
                    }
                }
                return Ok(());
            }
            if matches!(key.code, KeyCode::Down) {
                let cur = app.history_index.unwrap_or(0);
                let next = cur + 1;
                if next < app.chat_history.len() {
                    if let Some(p) = app.chat_history.get(next) {
                        textarea_set(&mut app.input, p);
                        app.input.move_cursor(ratatui_textarea::CursorMove::End);
                        app.history_index = Some(next);
                    }
                } else {
                    textarea_set(&mut app.input, "");
                    app.history_index = None;
                }
                return Ok(());
            }
            if is_enter {
                let text = app.input.lines().join("");
                if !text.trim().is_empty() {
                    textarea_set(&mut app.input, "");
                    app.history_index = None;
                    app.send_message(text);
                }
                return Ok(());
            }
            // Log scrolling (page keys are not text).
            if matches!(key.code, KeyCode::PageUp) {
                app.log_autoscroll = false;
                app.log_scroll = app.log_scroll.saturating_add(10);
                return Ok(());
            }
            if matches!(key.code, KeyCode::PageDown) {
                app.log_scroll = app.log_scroll.saturating_sub(10);
                if app.log_scroll == 0 {
                    app.log_autoscroll = true;
                }
                return Ok(());
            }
            // Everything else → the textarea component (typing, backspace,
            // left/right/home/end/delete, paste). It filters Release events and
            // control chars internally.
            app.input.input(key);
        }
    }
    Ok(())
}

/// Stop the running server (Drop kills it) and clear its log/health state so a
/// new model can be loaded. Used when navigating back from chat.
fn stop_server(app: &mut App) {
    app.server = None;
    app.server_log.clear();
    app.status = String::new();
}
