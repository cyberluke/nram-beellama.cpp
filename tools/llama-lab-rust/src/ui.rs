use crate::{App, ChatAction, Stage};
use ratatui::layout::{Constraint, HorizontalAlignment, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap};
use ratatui::Frame;

// -- modern palette ----------------------------------------------------------
const ACCENT: Color = Color::Cyan;
const ACCENT_B: Color = Color::Yellow; // secondary accent
const TEXT: Color = Color::Rgb(235, 244, 255);
const MUTED: Color = Color::Rgb(120, 140, 165);
const DIM: Color = Color::DarkGray;
const PANEL_BG: Color = Color::Rgb(20, 26, 36);
const USER_C: Color = Color::Rgb(93, 155, 255);   // user bubble accent
const BOT_C: Color = Color::Rgb(80, 200, 120);    // assistant accent

fn base() -> Style {
    Style::default()
}
fn muted() -> Style {
    Style::default().fg(MUTED)
}
fn dim() -> Style {
    Style::default().fg(DIM)
}

pub fn render(f: &mut Frame, app: &mut App) {
    // Full background wash so panes read as panels on a dark field.
    f.render_widget(
        Block::default().style(Style::default().bg(PANEL_BG)),
        f.area(),
    );

    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(f.area());

    render_header(f, header);
    match app.stage {
        Stage::Welcome => render_welcome(f, body, app),
        Stage::Language => render_language(f, body, app),
        Stage::Backend => render_backend(f, body, app),
        Stage::ModelSelect => render_model_select(f, body, app),
        Stage::Optimize => render_optimize(f, body, app),
        Stage::Chat => render_chat(f, body, app),
    }
    render_footer(f, footer, app);
    if app.show_help {
        render_help(f, app);
    }
}

fn render_header(f: &mut Frame, area: Rect) {
    let title = Line::from(vec![
        Span::styled("🦙  LLAMA — THE LAB", Style::default().fg(ACCENT_B).add_modifier(Modifier::BOLD)),
        Span::styled("  local LLM inference, tuned to your GPU", Style::default().fg(MUTED)),
    ]);
    let p = Paragraph::new(title)
        .alignment(HorizontalAlignment::Center)
        .style(Style::default().bg(PANEL_BG));
    f.render_widget(p, area);
}

/// Shared bordered-panel look with a left title and optional right status.
fn panel<'a>(
    title: &'a str,
    status: Option<(&'a str, Color)>,
    accent: Color,
    focused: bool,
) -> Block<'a> {
    let mut b = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if focused { accent } else { MUTED }))
        .style(Style::default().bg(PANEL_BG));
    if !title.is_empty() {
        b = b.title(Span::styled(format!(" {} ", title), Style::default().fg(accent).add_modifier(Modifier::BOLD)));
    }
    if let Some((s, c)) = status {
        b = b.title(Span::styled(format!(" {} ", s), Style::default().fg(c)))
            .title_alignment(HorizontalAlignment::Right);
    }
    b
}

fn render_welcome(f: &mut Frame, area: Rect, app: &App) {
    use ratatui::symbols::Marker;
    use ratatui::widgets::canvas::{Canvas, Circle, Line, Rectangle};

    // Full-screen layout: title bar / canvas / continue button.
    let div = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(5),
    ])
    .split(area);
    let (title_a, canvas_a, cont_a) = (div[0], div[1], div[2]);
    let _ = title_a;

    // ---- full-screen graphic llama (Canvas) ----
    let speed = 2.8f64; // slow, gentle walk
    let world_w = (canvas_a.width as f64).max(20.0);
    // x walks left→right with a 2-frame bob on y.
    let full = world_w + 18.0;
    let phase = (app.splash_frame as f64) * speed;
    let x = phase.rem_euclid(full) - 10.0;
    let bob = (app.splash_frame as f64 * 2.0).sin() * 0.2;
    let feet = canvas_a.y as f64 - 1.2 + bob * 0.2;
    let cy = canvas_a.y as f64 - 5.0 + bob * 0.4; // body center

    const WOOL: Color = Color::Rgb(233, 180, 140);   // llama tan
    const DARK: Color = Color::Rgb(140, 90, 55);     // ears/outline
    const FUR: Color = Color::Rgb(245, 230, 205);    // muzzle/belly
    const SADDLE: Color = Color::Rgb(200, 60, 60);   // red blanket
    const EYE: Color = Color::Rgb(25, 20, 18);
    const LEG: Color = Color::Rgb(188, 132, 92);

    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(ACCENT))
                .style(Style::default().bg(PANEL_BG)),
        )
        .marker(Marker::Braille)
        .x_bounds([0.0, world_w + 4.0])
        .y_bounds([canvas_a.y as f64 - (canvas_a.height as f64), canvas_a.y as f64 + 2.0])
        .paint(move |ctx| {
            let left = x;
            // tail
            ctx.draw(&Line { x1: left - 0.2, y1: cy - 1.0, x2: left - 2.0, y2: cy - 2.2, color: DARK });
            // body (taller rectangle) + belly
            ctx.draw(&Rectangle { x: left, y: cy - 1.4, width: 5.6, height: 3.1, color: WOOL });
            ctx.draw(&Rectangle { x: left + 0.3, y: cy + 0.6, width: 4.6, height: 1.1, color: FUR });
            // saddle blanket
            ctx.draw(&Rectangle { x: left + 0.6, y: cy - 0.6, width: 2.2, height: 1.1, color: SADDLE });
            // neck (taller) + head
            ctx.draw(&Line { x1: left + 5.2, y1: cy - 0.8, x2: left + 6.2, y2: cy - 3.4, color: WOOL });
            ctx.draw(&Line { x1: left + 5.6, y1: cy - 0.8, x2: left + 6.6, y2: cy - 3.4, color: WOOL });
            ctx.draw(&Circle { x: left + 6.8, y: cy - 3.6, radius: 1.3, color: WOOL });
            // ears
            ctx.draw(&Line { x1: left + 6.2, y1: cy - 4.2, x2: left + 6.4, y2: cy - 5.0, color: DARK });
            ctx.draw(&Line { x1: left + 6.9, y1: cy - 4.4, x2: left + 7.0, y2: cy - 5.1, color: DARK });
            // eye
            ctx.draw(&Circle { x: left + 7.3, y: cy - 3.7, radius: 0.22, color: EYE });
            // muzzle
            ctx.draw(&Circle { x: left + 6.1, y: cy - 3.2, radius: 0.5, color: FUR });
            ctx.draw(&Line { x1: left + 5.7, y1: cy - 3.2, x2: left + 6.5, y2: cy - 3.2, color: Color::Rgb(120,80,50) });
            // legs (4) with hoof tips — drawn last so they sit clearly below the body
            for (i, dx) in [0.9, 2.0, 3.2, 4.3].iter().enumerate() {
                let top_y = cy + 1.55;
                let c = if i % 2 == 0 { LEG } else { WOOL };
                ctx.draw(&Line { x1: left + dx, y1: top_y, x2: left + dx, y2: feet, color: c });
                ctx.draw(&Line { x1: left + dx + 0.10, y1: top_y, x2: left + dx + 0.10, y2: feet, color: c });
                ctx.draw(&Circle { x: left + dx, y: feet - 0.1, radius: 0.20, color: DARK });
            }
        });
    f.render_widget(canvas, canvas_a);

    // ---- Continue button ----
    let btn_w = 24u16;
    let btn_x = cont_a.x + (cont_a.width.saturating_sub(btn_w)) / 2;
    let btn_area = Rect::new(btn_x, cont_a.y, btn_w, 3);
    let blinks = (app.splash_frame / 12) % 2 == 0;
    let button = Paragraph::new(ratatui::text::Line::from(vec![
        Span::styled(format!("   ▶ {:^12}  ", app.t("continue")),
            Style::default()
                .bg(if blinks { ACCENT } else { PANEL_BG })
                .fg(if blinks { Color::Black } else { ACCENT_B })
                .add_modifier(Modifier::BOLD)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(ACCENT)));
    f.render_widget(button, btn_area);
}

fn render_optimize(f: &mut Frame, area: Rect, app: &App) {
    // Allocate the cards region, then the log region only when there is content
    // (or the server has already started). An empty log must not steal space.
    let cards_h = if app.server.is_none() && app.server_log.is_empty() {
        area.height
    } else {
        ((area.height as f32) * 0.62).round() as u16
    };
    let (plan_a, log_a) = {
        let v = Layout::vertical([
            Constraint::Length(cards_h.max(6)),
            Constraint::Min(0),
        ])
        .split(area);
        (v[0], v[1])
    };

    // The three preset plans for the current model + vision (kept in sync by
    // recompute_plan, which also applies any manual ctx/quant tweak to the
    // selected profile). Rendering from the stored plans keeps the on-screen
    // ctx/quant/VRAM identical to what will actually be launched.
    let plans = &app.preset_plans;

    // Top: vision banner (only when the model supports it).
    let y0 = area.y;
    let mut body_top = y0;
    if app.has_vision {
        let banner_a = Rect { x: plan_a.x, y: y0, width: plan_a.width, height: 2 };
        let banner = if app.use_vision {
            format!("  🖼  {}  [SPACE]", app.t("vision_on"))
        } else {
            format!("  🖼  {}  [SPACE]", app.t("vision_off"))
        };
        let bstyle = if app.use_vision {
            Style::default().fg(Color::Black).bg(ACCENT_B).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(235, 244, 255)).bg(Color::Rgb(90, 60, 40)).add_modifier(Modifier::BOLD)
        };
        f.render_widget(Paragraph::new(Line::from(Span::styled(banner, bstyle))).alignment(HorizontalAlignment::Center), banner_a);
        body_top = y0 + 2;
    }

    // Hardware header spanning the width, backend-aware.
    let hw_a = Rect { x: plan_a.x, y: body_top, width: plan_a.width, height: 1 };
    let hw_line = match app.backend() {
        crate::optimizer::Backend::Cpu => {
            let mem = crate::server::detect_mem_mib(crate::optimizer::Backend::Cpu);
            if mem > 0.0 {
                format!("  🧠  CPU · {:.1} GB RAM total", mem / 1024.0)
            } else {
                "  🧠  CPU · RAM unknown".into()
            }
        }
        crate::optimizer::Backend::Cuda | crate::optimizer::Backend::Vulkan => {
            let mem = crate::server::detect_mem_mib(app.backend());
            if mem <= 0.0 {
                "  🖥  GPU VRAM unknown".into()
            } else {
                let (gname, gused) = crate::server::gpu_info_cached();
                if gname.is_empty() {
                    format!("  🖥  {} {} {:.0} GB", app.t("gpu_total"), app.t("be_backend_of"), mem / 1024.0)
                } else {
                    format!("  🖥  {gname} · {:.0} / {:.0} GB used", gused / 1024.0, mem / 1024.0)
                }
            }
        }
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(hw_line, Style::default().fg(MUTED)))),
        hw_a,
    );

    // Cards region below the banner + GPU header (+ optional Qwen hint line).

    // Qwen 3.x models spend a lot of context on hidden reasoning tokens. A
    // one-line hint explains that the effort knob only trades reasoning-token
    // volume (the fix for the Qwen-3 runaway-reasoning issue), since these
    // models offer no other behavioral difference across effort levels.
    let qwen_hint = app.is_qwen_3x();
    let cards_top = body_top + 1 + if qwen_hint { 1 } else { 0 };
    if qwen_hint {
        let qa = Rect { x: plan_a.x, y: body_top + 1, width: plan_a.width, height: 1 };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(format!("  ⚙  {}", app.t("qwen_hint")), Style::default().fg(ACCENT)))),
            qa,
        );
    }
    let cards_a = Rect {
        x: plan_a.x,
        y: cards_top,
        width: plan_a.width,
        height: plan_a.height.saturating_sub(cards_top - area.y).max(4),
    };
    render_perf_cards(f, cards_a, app, plans);

    // CUDA ↔ Vulkan comparison for the currently selected card (real numbers).
    if matches!(app.backend(), crate::optimizer::Backend::Cuda | crate::optimizer::Backend::Vulkan)
        && app.server.is_none()
    {
        let diff_a = Rect {
            x: plan_a.x,
            y: cards_a.y + cards_a.height.saturating_sub(4),
            width: plan_a.width,
            height: 3,
        };
        render_gpu_diff(f, diff_a, app, plans, app.preset_index);
    }

    // Launch / loading hint under the cards.
    {
        let hint_y = cards_a.y + cards_a.height.saturating_sub(1);
        let hint_a = Rect { x: cards_a.x, y: hint_y, width: cards_a.width, height: 1 };
        let hint = if app.server.is_some() {
            format!("   {}{}", app.t("loading_model"), ".".repeat((app.splash_frame % 4) as usize))
        } else {
            let focus_txt = match app.param_focus {
                crate::ParamFocus::Card => "profile",
                crate::ParamFocus::Ctx => "ctx",
                crate::ParamFocus::Quant => "KV quant",
                crate::ParamFocus::Reasoning => "effort",
            };
            format!("   ▶  {} ─ ↑/↓ {}, ←/▶ {}, Space vision", app.t("choose_card"), focus_txt, focus_txt)
        };
        let hint_style = if app.server.is_some() {
            Style::default().fg(ACCENT)
        } else {
            Style::default().fg(ACCENT_B).add_modifier(Modifier::BOLD)
        };
        f.render_widget(Paragraph::new(Line::from(Span::styled(hint, hint_style))), hint_a);
    }

    // Live llama-server log pane (bottom) — only when there is content.
    if !(app.server.is_none() && app.server_log.is_empty()) {
        render_log_pane(f, log_a, app);
    }
}

/// Three vertical "racing" performance cards, one per preset. Left/Right moves
/// the highlight; each bar's height is scaled to its unified KV pool so the
/// "fastest" card reads instantly. Every card shows its own ctx + quants.
fn render_perf_cards(f: &mut Frame, area: Rect, app: &App, plans: &[crate::optimizer::LaunchPlan]) {
    use ratatui::layout::Direction;

    if plans.is_empty() {
        return;
    }
    let cols = Layout::new(Direction::Horizontal, [
        Constraint::Percentage(33),
        Constraint::Percentage(33),
        Constraint::Percentage(33),
    ])
    .split(area);

    let max_ctx = plans.iter().map(|p| p.ctx).max().unwrap_or(1).max(1) as f64;
    let colors = [ACCENT, ACCENT_B, Color::Rgb(80, 200, 120)];

    let card_h = area.height.saturating_sub(4) as usize;

    for (i, plan) in plans.iter().enumerate() {
        let card_a = cols[i];
        let selected = i == app.preset_index;
        let pres = crate::optimizer::Preset::ALL[i];
        let title = app.t(pres.title_key());
        let desc = app.t(pres.desc_key());
        let accent = colors[i];

        // Bar height: 3 cells of visual pacing at the bottom.
        let bar_h = (((plan.ctx as f64) / max_ctx) * 3.0).round().clamp(1.0, 3.0) as usize;

        let border_style = if selected {
            Style::default().fg(accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(70, 85, 105))
        };

        let mut lines = Vec::new();
        lines.push(Line::from(vec![
            Span::styled(if selected { "▸ " } else { "  " }, Style::default().fg(accent)),
            Span::styled(title.clone(), Style::default().fg(if selected { accent } else { MUTED }).add_modifier(Modifier::BOLD)),
            Span::styled(if selected && app.param_focus == crate::ParamFocus::Card { "  ◀▶" } else { "" }, Style::default().fg(accent).add_modifier(Modifier::BOLD)),
        ]));
        // Helper to pull a summary value.
        let sum = |k: &str| -> String {
            plan.summary.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone()).unwrap_or_default()
        };
        let kv_q = sum("kv_quant");
        let draft = sum("draft");
        let per = sum("per_slot");
        let vram_used = sum("vram");
        let effort = sum("effort");
        let slots = plan.parallel;
        // Which knob the ←/→ keys currently tweak (only meaningful on the card).
        let focus_ctx = selected && app.param_focus == crate::ParamFocus::Ctx;
        let focus_quant = selected && app.param_focus == crate::ParamFocus::Quant;
        let focus_effort = selected && app.param_focus == crate::ParamFocus::Reasoning;
        let focus_style = Style::default().fg(Color::Black).bg(ACCENT_B).add_modifier(Modifier::BOLD);
        // Vision badge on every card.
        let vision_text = if app.has_vision {
            if app.use_vision { "🖼 vision ON" } else { "▢ vision OFF" }
        } else {
            "no vision"
        };
        let vision_col = if app.has_vision {
            if app.use_vision {
                Style::default().fg(Color::Black).bg(ACCENT_B).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(MUTED)
            }
        } else {
            Style::default().fg(DIM)
        };
        let ctx_label = format!("{:>3}k ctx", plan.ctx / 1000);
        lines.push(Line::from(vec![
            Span::styled(ctx_label, if focus_ctx { focus_style } else { Style::default().fg(TEXT).add_modifier(Modifier::BOLD) }),
            Span::styled(if focus_ctx { " ←▶ " } else { "   " }, if focus_ctx { focus_style } else { base() }),
            Span::styled(vision_text, vision_col),
        ]));
        lines.push(Line::from(vec![
            Span::styled(format!("KV  {kv_q}"), if focus_quant { focus_style } else { Style::default().fg(MUTED) }),
            Span::styled(if focus_quant { "  ←▶" } else { "    " }, if focus_quant { focus_style } else { Style::default().fg(DIM) }),
        ]));
        lines.push(Line::from(vec![
            Span::styled(
                format!("Eff {}", if effort.is_empty() { crate::optimizer::DEFAULT_REASONING_EFFORT } else { &effort }),
                if focus_effort { focus_style } else { Style::default().fg(MUTED) },
            ),
            Span::styled(if focus_effort { "  ←▶" } else { "    " }, if focus_effort { focus_style } else { Style::default().fg(DIM) }),
        ]));
        lines.push(Line::from(Span::styled(format!("Spec {draft}"), Style::default().fg(MUTED))));
        lines.push(Line::from(Span::styled(
            format!("{slots}× slots · {per}/ea", ),
            Style::default().fg(DIM),
        )));
        lines.push(Line::from(Span::styled(
            format!("{} ~{vram_used}", if matches!(app.backend(), crate::optimizer::Backend::Cpu) { "RAM" } else { "VRAM" }),
            Style::default().fg(if selected { TEXT } else { DIM }).add_modifier(Modifier::BOLD),
        )));
        // Spacer, then the bar.
        if card_h > 11 {
            lines.push(Line::from(""));
        }
        for _ in 0..bar_h {
            let sym = "█";
            let col = if selected { accent } else { Color::Rgb(60, 80, 100) };
            lines.push(Line::from(Span::styled(
                format!("  {sym}{sym}{sym}  "),
                Style::default().fg(col),
            )));
        }
        while lines.len() < card_h.saturating_sub(3) {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(desc.clone(), Style::default().fg(if selected { TEXT } else { DIM }))));

        let p = Paragraph::new(lines)
            .alignment(HorizontalAlignment::Center)
            .block(Block::default().borders(Borders::ALL).border_style(border_style));
        f.render_widget(p, card_a);
    }
}

fn render_language(f: &mut Frame, area: Rect, app: &App) {
    let title = app.t("choose_lang");
    let block = panel(&title, None, ACCENT, true);
    let inner = block.inner(area);
    let names = ["English", "Čeština", "Slovenčina"];
    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, name) in names.iter().enumerate() {
        let selected = i == app.lang_index;
        lines.push(Line::from(if selected {
            vec![
                Span::styled("▸ ", Style::default().fg(ACCENT_B)),
                Span::styled(name.to_string(), Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
            ]
        } else {
            vec![
                Span::styled("  ", base()),
                Span::styled(name.to_string(), muted()),
            ]
        }));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(app.t("hint_lang"), dim())));
    f.render_widget(Paragraph::new(lines).block(block), inner);
}

/// Backend selection as vertical slides (CPU / CUDA / Vulkan; CUDA only on
/// NVIDIA machines). ↑/↓ moves the highlight, Enter picks it.
fn render_backend(f: &mut Frame, area: Rect, app: &App) {
    use ratatui::layout::Direction;

    let title = app.t("choose_backend");
    let block = panel(&title, None, ACCENT, true);
    let inner = block.inner(area);

    let avail = app.available_backends();
    if avail.is_empty() {
        return;
    }
    let widths = std::iter::repeat_n(Constraint::Percentage((100 / avail.len() as u16) as u16), avail.len())
        .collect::<Vec<_>>();
    let cols = Layout::new(Direction::Horizontal, widths).split(inner);

    let color_of = |be: crate::optimizer::Backend| -> Color {
        match be {
            crate::optimizer::Backend::Cpu => Color::Rgb(120, 180, 240),
            crate::optimizer::Backend::Cuda => Color::Rgb(120, 220, 140),
            crate::optimizer::Backend::Vulkan => Color::Rgb(240, 160, 110),
        }
    };
    let icon_of = |be: crate::optimizer::Backend| -> String {
        match be {
            crate::optimizer::Backend::Cpu => "🧠".into(),
            crate::optimizer::Backend::Cuda => "🖥".into(),
            crate::optimizer::Backend::Vulkan => "🎯".into(),
        }
    };

    for (i, be) in avail.iter().enumerate() {
        let selected = i == app.backend_index;
        let accent = color_of(*be);
        let card = cols[i];

        let mut lines = Vec::new();
        // Header: icon + name, highlighted when selected.
        lines.push(Line::from(vec![
            Span::styled(icon_of(*be), Style::default().fg(accent)),
            Span::styled(" ", base()),
            Span::styled(app.t(be.title_key()), Style::default().fg(if selected { accent } else { MUTED }).add_modifier(Modifier::BOLD)),
        ]));
        lines.push(Line::from(""));
        // Reason bullets — the "why this mode is good".
        for r in be.reasons() {
            let bullet = if selected { "▸ " } else { "· " };
            lines.push(Line::from(vec![
                Span::styled(bullet, Style::default().fg(if selected { accent } else { DIM })),
                Span::styled(app.t(r), Style::default().fg(if selected { TEXT } else { DIM })),
            ]));
        }
        lines.push(Line::from(""));
        // Memory hint.
        let mem = crate::server::detect_mem_mib(*be);
        if mem > 0.0 {
            lines.push(Line::from(Span::styled(
                format!("{}: {:.1} GB", app.t("mem_avail"), mem / 1024.0),
                Style::default().fg(DIM),
            )));
        }

        let border = if selected {
            Style::default().fg(accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(70, 85, 105))
        };
        let p = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).border_style(border));
        f.render_widget(p, card);
    }

    // Bottom hint line.
    let hint_y = inner.y + inner.height.saturating_sub(1);
    let hint_a = Rect { x: inner.x, y: hint_y, width: inner.width, height: 1 };
    let hint = format!("  {}", app.t("hint_backend"));
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(ACCENT_B).add_modifier(Modifier::BOLD)))),
        hint_a,
    );
}

/// Rendered CUDA↔Vulkan profile diff (called from render_backend when the
/// highlight sits on a GPU backend).
fn render_gpu_diff(
    f: &mut Frame,
    area: Rect,
    app: &App,
    plans: &[crate::optimizer::LaunchPlan],
    preset_index: usize,
) {
    if plans.is_empty() {
        return;
    }
    // Re-derive the Inputs for the CURRENT model + backend (plans are for the
    // selected backend; the diff needs both GPU backends at the same model).
    let path = app.selected_model_path();
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let mem = crate::server::detect_vram_mib_cached();
    let ip = crate::optimizer::Inputs {
        backend: app.backend(),
        mem_mib: mem,
        model_file_bytes: bytes,
        vision_mib: app.vision_budget_mib(),
        moe_active: crate::server::gguf_moe_info(&path)
            .map(|m| if m.expert_count > 0 { (m.expert_used.max(1) as f64 / m.expert_count as f64).clamp(0.05, 1.0) } else { 1.0 })
            .unwrap_or(1.0),
    };
    let preset = crate::optimizer::Preset::ALL[preset_index.min(2)];
    let rows = crate::optimizer::profile_diff(&ip, preset);
    if rows.is_empty() {
        return;
    }

    let highlight = app.backend();
    let title = match highlight {
        crate::optimizer::Backend::Cuda => app.t("gpu_diff_cuda"),
        crate::optimizer::Backend::Vulkan => app.t("gpu_diff_vulkan"),
        crate::optimizer::Backend::Cpu => return,
    };
    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(format!("  ⚖  {title}"), Style::default().fg(ACCENT_B).add_modifier(Modifier::BOLD))),
    ];
    // Measured from the last real chat run beats any estimate — show it first.
    if app.last_tps > 0.0 {
        lines.push(Line::from(vec![
            Span::styled(format!("    {:<13}", "measured"), Style::default().fg(DIM)),
            Span::styled(format!("{:.0} tok/s (last run)", app.last_tps), Style::default().fg(Color::Rgb(120, 220, 140)).add_modifier(Modifier::BOLD)),
        ]));
    }
    for r in rows {
        let hl = r.winner == Some(highlight);
        let cuda_col = if r.winner == Some(crate::optimizer::Backend::Cuda) { ACCENT_B } else { MUTED };
        let vk_col = if r.winner == Some(crate::optimizer::Backend::Vulkan) { ACCENT_B } else { MUTED };
        let hl_marker = if hl { "◀" } else { " " };
        lines.push(Line::from(vec![
            Span::styled(format!("    {:<13}", r.label), Style::default().fg(DIM)),
            Span::styled(format!("CUDA {:>12}   ", r.cuda), Style::default().fg(if r.winner == Some(crate::optimizer::Backend::Vulkan) { DIM } else { cuda_col }).add_modifier(if hl { Modifier::BOLD } else { Modifier::empty() })),
            Span::styled(format!("Vulkan {:>12}{}", r.vulkan, hl_marker), Style::default().fg(if r.winner == Some(crate::optimizer::Backend::Vulkan) { vk_col } else { DIM }).add_modifier(if hl { Modifier::BOLD } else { Modifier::empty() })),
        ]));
    }
    f.render_widget(Paragraph::new(lines).style(Style::default().bg(PANEL_BG)), area);
}

fn render_footer(f: &mut Frame, area: Rect, app: &App) {
    let hint = match app.stage {
        Stage::Chat => app.t("hint_chat"),
        Stage::ModelSelect => app.t("hint_models"),
        Stage::Welcome | Stage::Language | Stage::Backend | Stage::Optimize => String::new(),
    };
    let extra = match app.stage {
        Stage::ModelSelect => {
            let prof = match app.profile {
                crate::profiles::Profile::Small => app.t("profile_small"),
                crate::profiles::Profile::Balanced => app.t("profile_balanced"),
                crate::profiles::Profile::Big => app.t("profile_big"),
            };
            format!(" [P] {prof}  ·  [F] {}x  ·  [+] folder", app.fleet_count)
        }
        Stage::Chat => {
            if app.last_tps > 0.0 {
                format!("  {:.1} tok/s", app.last_tps)
            } else {
                String::new()
            }
        }
        Stage::Welcome | Stage::Language | Stage::Backend | Stage::Optimize => String::new(),
    };
    let line = Line::from(vec![
        Span::styled("  ", base()),
        Span::styled(hint, Style::default().fg(MUTED)),
        Span::styled(extra, Style::default().fg(ACCENT)),
    ]);
    f.render_widget(
        Paragraph::new(line).style(Style::default().bg(PANEL_BG)),
        area,
    );
}

fn render_help(f: &mut Frame, app: &App) {
    let w = (f.area().width / 3).clamp(40, 64);
    let h = 19u16.min(f.area().height);
    let x = f.area().x + (f.area().width - w) / 2;
    let y = f.area().y + (f.area().height - h) / 2;
    let area = Rect::new(x, y, w, h);
    let help = vec![
        Line::from(Span::styled(format!(" {} ", app.t("help_title")), Style::default().fg(ACCENT_B).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(Span::styled(format!("  Enter       {}", app.t("help_send")), muted())),
        Line::from(Span::styled(format!("  ↑ / ↓       {}", app.t("help_recall")), muted())),
        Line::from(Span::styled(format!("  PageUp/Dn   {}", app.t("help_scroll")), muted())),
        Line::from(Span::styled(format!("  Ctrl += / -  {}", app.t("help_resize")), muted())),
        Line::from(Span::styled(format!("  Ctrl + a    {}", app.t("help_autoscroll")), muted())),
        Line::from(Span::styled(format!("  Ctrl + t    {}", app.t("help_theme")), muted())),
        Line::from(Span::styled(format!("  Ctrl + ?    {}", app.t("help_this")), muted())),
        Line::from(Span::styled(format!("  Ctrl + Q    {}", app.t("help_quit")), muted())),
        Line::from(Span::styled(format!("  Esc         {}", app.t("help_esc")), muted())),
        Line::from(Span::styled(format!("  Ctrl + n    {}", app.t("help_new")), muted())),
        Line::from(Span::styled(format!("  F1–F5       {}", app.t("help_fn")), muted())),
        Line::from(Span::styled(format!("  Alt         {}", app.t("help_alt")), muted())),
        Line::from(""),
        Line::from(Span::styled(format!("  {}", app.t("help_commands")), Style::default().fg(ACCENT))),
    ];
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(PANEL_BG));
    f.render_widget(Paragraph::new(help).block(block), area);
}

fn render_model_select(f: &mut Frame, area: Rect, app: &App) {
    if app.add_folder_active {
        render_add_folder(f, area, app);
        return;
    }
    let title = format!(" {}  ({}) ", app.t("choose_model"), app.models.len());
    let hint = app.t("hint_pick_more");
    let block = panel(
        &title,
        Some((hint.as_str(), DIM)),
        ACCENT,
        true,
    );
    let inner = block.inner(area);

    let mut lines: Vec<Line> = Vec::new();
    if app.models.is_empty() {
        lines.push(Line::from(Span::styled(app.t("no_models"), Style::default().fg(Color::Red))));
    }
    // Use the full panel width: name fills available space, size + drive pinned right.
    let inner_w = (inner.width as usize).saturating_sub(4);
    let size_w = "999.9 GB".len();
    let drive_w = 16usize;
    let pad = 3usize; // arrow + spacing gap
    let name_w = inner_w.saturating_sub(size_w + drive_w + pad).max(12);
    for (i, m) in app.models.iter().enumerate() {
        let selected = i == app.model_index;
        let gb = format!("{:.1} GB", m.size_gb);
        let friendly = friendly_name(&m.name);
        let drv = drive_root(&m.path);
        let vis = if m.mmproj.is_some() { " 🖼" } else { "" };
        let arrow = if selected { "▸ " } else { "   " };
        // Pad name + badge together so size/drive sit at the right edge on every row.
        let name_s = friendly.chars().take(name_w).collect::<String>();
        let filled = name_s.chars().count() + vis.chars().count();
        let extra = name_w.saturating_sub(filled);
        let mut name_col = String::with_capacity(name_w);
        name_col.push_str(&name_s);
        name_col.push_str(vis);
        if extra > 0 {
            name_col.push_str(&" ".repeat(extra));
        }
        let line = Line::from(vec![
            Span::styled(arrow, Style::default().fg(if selected { ACCENT_B } else { DIM })),
            Span::styled(name_col, if selected {
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(MUTED)
            }),
            Span::styled(format!("{:>width$}  ", gb, width = size_w), Style::default().fg(DIM)),
            Span::styled(format!("{drv:12}"), Style::default().fg(Color::Rgb(90,110,130))),
        ]);
        lines.push(line);
    }
    f.render_widget(Paragraph::new(lines).block(block), inner);
    render_model_tooltips(f, area, app);
}

fn friendly_name(name: &str) -> String {
    // "lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf" ->
    // "Qwen3.8-27B Q4_K_M"
    let base = name.rsplit(['\\', '/']).next().unwrap_or(name);
    let f = base
        .trim_end_matches(".gguf")
        .replace("-Q", " Q")
        .replace("_Q", " Q");
    f
}

fn drive_root(path: &str) -> String {
    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' {
        let drive = &path[..2];
        // find the root folder name after the drive
        let rest = &path[2..];
        let first = rest.split(['\\', '/']).filter(|s| !s.is_empty()).next().unwrap_or("");
        format!("{drive}\\{first}")
    } else {
        "?".into()
    }
}

/// Tooltips for the model-select screen (F fleet / P profile).
fn render_model_tooltips(f: &mut Frame, area: Rect, app: &App) {
    let tut_lines = vec![
        Line::from(Span::styled(format!("  {}", app.t("tut_fleet")), dim())),
        Line::from(Span::styled(format!("  {}", app.t("tut_profile")), dim())),
    ];
    let tut_a = Rect::new(area.x, area.y + area.height.saturating_sub(3), area.width.saturating_sub(1), 3);
    f.render_widget(Paragraph::new(tut_lines), tut_a);
}

fn render_add_folder(f: &mut Frame, area: Rect, app: &App) {
    let title = app.t("add_folder");
    let block = panel(&title, None, ACCENT, true);
    let inner = block.inner(area);
    let save_hint = app.t("save_cancel");
    let text = Text::from(vec![
        Line::from(vec![
            Span::styled("   path > ", Style::default().fg(ACCENT_B).add_modifier(Modifier::BOLD)),
            Span::styled(app.add_folder_path.clone(), Style::default().fg(TEXT)),
        ]),
        Line::from(""),
        Line::from(Span::styled(format!("  {save_hint}"), dim())),
    ]);
    f.render_widget(Paragraph::new(text).block(block), inner);
}

fn render_chat(f: &mut Frame, area: Rect, app: &mut App) {
    let height = area.height.saturating_sub(4).max(4);
    let chat_h = (height as f32 * app.chat_ratio).round() as u16;
    let [chat_area, log_area, btn_area, input_area] = Layout::vertical([
        Constraint::Length(chat_h.max(2)),
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(3),
    ])
    .areas(area);

    render_chat_pane(f, chat_area, app);
    render_log_pane(f, log_area, app);
    render_chat_buttons(f, btn_area, app);
    render_input_box(f, input_area, app);
}

/// Clickable shortcut bar — mouse fallback for the Ctrl combos. Each button
/// records its rect into `app.chat_buttons` for hit-testing in the event loop,
/// and the hovered button (under `app.mouse_pos`) gets highlighted.
fn render_chat_buttons(f: &mut Frame, area: Rect, app: &mut App) {
    app.chat_buttons.clear();
    let labels: &[(ChatAction, &str)] = &[
        (ChatAction::NewModel, "◈ New model"),
        (ChatAction::Theme, "◐ Theme"),
        (ChatAction::Autoscroll, "⏷ Auto"),
        (ChatAction::Help, "? Help"),
        (ChatAction::Back, "← Back"),
        (ChatAction::Quit, "✕ Quit"),
    ];
    let mut x = area.x;
    let mut spans: Vec<Span> = Vec::new();
    for (action, label) in labels.iter() {
        let len = label.chars().count() as u16 + 2;
        let rect = Rect { x, y: area.y, width: len, height: 1 };
        app.chat_buttons.push((rect, *action));
        let hover = app
            .mouse_pos
            .map(|(mx, my)| mx >= rect.x && mx < rect.x + rect.width && my == rect.y)
            .unwrap_or(false);
        let style = if hover {
            Style::default().fg(Color::Black).bg(ACCENT_B).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(if *action == ChatAction::Quit { Color::Rgb(255,120,120) } else { ACCENT })
        };
        let text = if hover { format!(" {label} ") } else { format!(" [{label}]") };
        spans.push(Span::styled(text, style));
        spans.push(Span::styled(" ", Style::default().fg(DIM)));
        x += len + 1;
    }
    f.render_widget(Paragraph::new(Line::from(spans)).style(Style::default().bg(PANEL_BG)), area);
}

fn render_chat_pane(f: &mut Frame, area: Rect, app: &App) {
    let health = app.server.as_ref().map(|s| s.is_healthy()).unwrap_or(false);
    let status_color = if health { Color::Green } else { Color::Yellow };
    let accent = if app.theme_inverted { Color::Magenta } else { ACCENT };
    let title = app.t("chat");
    let block = panel(&title, Some((app.status.as_str(), status_color)), accent, true);
    let inner = block.inner(area);

    let mut lines: Vec<Line> = Vec::new();
    for m in &app.messages {
        let role_style = if m.role == "assistant" {
            Style::default().fg(BOT_C).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(USER_C).add_modifier(Modifier::BOLD)
        };
        let icon = if m.role == "assistant" { "🧪" } else { "🧑" };
        let mut first = true;
        for l in m.content.lines() {
            if first {
                lines.push(Line::from(vec![
                    Span::styled(format!("{}  ", icon), role_style),
                    Span::raw(l.to_string()),
                ]));
                first = false;
            } else {
                lines.push(Line::from(Span::raw(format!("     {}", l))));
            }
        }
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("    {}", app.t("no_messages")),
            dim(),
        )));
    }

    let total_lines = lines.len() as i32;
    let view_h = inner.height.max(1) as i32;
    let bottom = (total_lines - view_h).max(0);
    let top = if app.auto_scroll { bottom } else { (bottom - app.scrollback).max(0) };
    let p = Paragraph::new(lines).block(block.clone()).wrap(Wrap { trim: true });
    f.render_widget(p.scroll((top as u16, 0)), inner);
    draw_vert_scrollbar(f, inner, total_lines, top, app.theme_inverted);
}

fn render_log_pane(f: &mut Frame, area: Rect, app: &App) {
    let title = app.t("server_log");
    let block = panel(&title, None, ACCENT, false);
    let inner = block.inner(area);

    let mut lines: Vec<Line> = Vec::new();
    for l in app.server_log.iter() {
        lines.push(Line::from(Span::styled(l.clone(), Style::default().fg(MUTED))));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(app.t("server_not_started"), dim())));
    }

    let total = lines.len() as i32;
    let view = inner.height.max(1) as i32;
    let bottom = (total - view).max(0);
    let top = if app.log_autoscroll { bottom } else { (bottom - app.log_scroll as i32).max(0) };
    let p = Paragraph::new(lines).block(block.clone()).wrap(Wrap { trim: true });
    f.render_widget(p.scroll((top as u16, 0)), inner);
    draw_vert_scrollbar(f, inner, total, top, app.theme_inverted);
}

/// Thin right-edge scrollbar showing position `top` over `total` rows.
fn draw_vert_scrollbar(f: &mut Frame, area: Rect, total: i32, top: i32, inverted: bool) {
    if total <= area.height as i32 {
        return;
    }
    let mut state = ScrollbarState::new(total as usize).position(top as usize);
    let sb = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(Some("  "))
        .end_symbol(Some("  "))
        .thumb_symbol("█")
        .track_symbol(Some("│"))
        .style(Style::default().fg(inverted.then_some(Color::Magenta).unwrap_or(MUTED)))
        .thumb_style(Style::default().fg(ACCENT));
    f.render_stateful_widget(sb, area, &mut state);
}

fn render_input_box(f: &mut Frame, area: Rect, app: &App) {
    let placeholder = app.t("input_placeholder");
    // The input is a real tui-textarea widget: it draws its own text + cursor.
    // The border lives on OUR block (no inner block on the textarea — a second
    // block inside the pane is what caused the margin/overflow).
    let prompt_style = Style::default().fg(ACCENT_B).add_modifier(Modifier::BOLD);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" >  ", prompt_style))
        .border_style(Style::default().fg(if !app.input.is_empty() { ACCENT } else { MUTED }))
        .style(Style::default().bg(PANEL_BG));
    let inner = block.inner(area);

    let mut ta = app.input.clone();
    ta.set_placeholder_text(placeholder);
    ta.set_cursor_style(Style::default().fg(ACCENT).add_modifier(Modifier::REVERSED));
    ta.set_style(Style::default().fg(TEXT).bg(PANEL_BG));

    // Render border + textarea flush inside it (single line, no extra padding).
    f.render_widget(block, area);
    f.render_widget(&ta, Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: 1,
    });
}
