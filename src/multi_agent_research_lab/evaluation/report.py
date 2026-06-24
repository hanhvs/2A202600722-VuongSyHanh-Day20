"""Benchmark report rendering (markdown + self-contained HTML)."""

from html import escape

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown.

    TODO(student): Add richer analysis, examples, screenshots, and trace links.
    """

    lines = ["# Benchmark Report", "", "| Run | Latency (s) | Cost (USD) | Quality | Notes |", "|---|---:|---:|---:|---|"]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")
    return "\n".join(lines) + "\n"


_AGENT_COLORS = {
    "supervisor": "#6366f1",
    "researcher": "#0ea5e9",
    "analyst": "#f59e0b",
    "writer": "#10b981",
    "critic": "#ef4444",
    "baseline": "#8b5cf6",
    "done": "#64748b",
}


def _chip(label: str) -> str:
    color = _AGENT_COLORS.get(label, "#64748b")
    return f'<span class="chip" style="--c:{color}">{escape(label)}</span>'


def _flow_diagram(state: ResearchState) -> str:
    if not state.route_history:
        return '<p class="muted">(no routes)</p>'
    return '<div class="flow">' + '<span class="arrow">&rarr;</span>'.join(
        _chip(r) for r in state.route_history
    ) + "</div>"


def _timeline(state: ResearchState) -> str:
    items = ""
    for i, e in enumerate(state.trace, 1):
        color = _AGENT_COLORS.get(e["name"], "#64748b")
        payload = ", ".join(f"{k}={v}" for k, v in e["payload"].items())
        items += (
            f'<li style="--c:{color}"><span class="dot"></span>'
            f'<div><strong>{escape(e["name"])}</strong>'
            f'<div class="payload">{escape(payload)}</div></div></li>'
        )
    return f'<ol class="timeline">{items}</ol>'


# (key, label, unit, better) — better: "lower" | "higher" | "" (no winner)
_METRIC_SPECS = [
    ("latency_s", "Latency", "s", "lower"),
    ("total_tokens", "Total tokens", "", "lower"),
    ("input_tokens", "Input tokens", "", "lower"),
    ("output_tokens", "Output tokens", "", "lower"),
    ("cost_usd", "Est. cost", "$", "lower"),
    ("cost_per_1k_tokens", "Cost / 1k tokens", "$", "lower"),
    ("tokens_per_s", "Throughput", "tok/s", "higher"),
    ("agent_steps", "Agent steps", "", ""),
    ("supervisor_iterations", "Supervisor iters", "", ""),
    ("sources_used", "Sources used", "", "higher"),
    ("citation_coverage_pct", "Citation coverage", "%", "higher"),
    ("answer_words", "Answer length", "words", ""),
    ("answer_chars", "Answer length", "chars", ""),
    ("quality_score", "Quality (proxy)", "/10", "higher"),
    ("errors", "Errors", "", "lower"),
    ("failed", "Failed", "", "lower"),
]


def _fmt(key: str, val: float | None, unit: str) -> str:
    if val is None:
        return '<span class="muted">n/a</span>'
    if key == "failed":
        return "yes" if val else "no"
    if "$" in unit:
        return f"${val:.6f}".rstrip("0").rstrip(".") if val else "$0"
    if val == int(val):
        num = f"{int(val):,}"
    else:
        num = f"{val:,.2f}"
    return f"{num}&nbsp;{escape(unit)}".strip() if unit else num


def _best_index(vals: list[float | None], better: str) -> int | None:
    real = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not better or len(real) < 2 or len({v for _, v in real}) < 2:
        return None
    pick = min if better == "lower" else max
    return pick(real, key=lambda iv: iv[1])[0]


def _comparison_table(items: list[tuple[BenchmarkMetrics, ResearchState]]) -> str:
    runs = [m for m, _ in items]
    head = "<th>Metric</th>" + "".join(f"<th>{escape(m.run_name)}</th>" for m in runs)
    if len(runs) == 2:
        head += "<th>&Delta; (B&minus;A)</th>"

    body = ""
    for key, label, unit, better in _METRIC_SPECS:
        vals = [m.details.get(key) for m in runs]
        best = _best_index(vals, better)
        cells = ""
        for i, v in enumerate(vals):
            cls = ' class="win"' if best == i else ""
            cells += f"<td{cls}>{_fmt(key, v, unit)}</td>"
        delta = ""
        if len(runs) == 2 and vals[0] is not None and vals[1] is not None:
            d = vals[1] - vals[0]
            sign = "+" if d > 0 else ""
            good = better and best == 1
            bad = better and best == 0
            dcls = "up" if good else ("down" if bad else "")
            delta = f'<td class="delta {dcls}">{sign}{_fmt(key, d, unit)}</td>'
        elif len(runs) == 2:
            delta = '<td class="muted">&mdash;</td>'
        unit_lbl = f" <small>({escape(unit)})</small>" if unit and unit != "$" else ""
        body += f"<tr><th>{escape(label)}{unit_lbl}</th>{cells}{delta}</tr>"
    return f'<table class="cmp"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _metric_card(m: BenchmarkMetrics) -> str:
    cost = "n/a" if m.estimated_cost_usd is None else f"${m.estimated_cost_usd:.5f}"
    quality = "n/a" if m.quality_score is None else f"{m.quality_score:.1f}/10"
    color = _AGENT_COLORS.get("writer" if "multi" in m.run_name else "baseline", "#6366f1")
    return (
        f'<div class="card" style="--c:{color}">'
        f'<h3>{escape(m.run_name)}</h3>'
        f'<div class="stat"><span>{m.latency_seconds:.3f}s</span><label>latency</label></div>'
        f'<div class="stat"><span>{cost}</span><label>est. cost</label></div>'
        f'<div class="stat"><span>{quality}</span><label>quality</label></div>'
        f'<p class="notes">{escape(m.notes)}</p></div>'
    )


def render_html_report(items: list[tuple[BenchmarkMetrics, ResearchState]], query: str = "") -> str:
    """Render a self-contained, styled HTML report: metric cards, flow diagram, timeline."""

    cards = "".join(_metric_card(m) for m, _ in items)
    table = _comparison_table(items)

    sections = ""
    for m, state in items:
        sections += (
            f'<section class="run"><h2>{escape(m.run_name)}</h2>'
            f'<h4>Route</h4>{_flow_diagram(state)}'
            f'<h4>Trace timeline</h4>{_timeline(state)}'
            f'<details><summary>Final answer</summary>'
            f'<pre>{escape(state.final_answer or "(none)")}</pre></details></section>'
        )

    css = """
:root{--bg:#0f172a;--fg:#e2e8f0;--mut:#94a3b8;--card:#1e293b;--line:#334155}
*{box-sizing:border-box}
body{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;background:var(--bg);
color:var(--fg);line-height:1.5}
.wrap{max-width:920px;margin:0 auto;padding:2.5rem 1.25rem 4rem}
header{background:linear-gradient(120deg,#6366f1,#0ea5e9);padding:2.5rem 1.25rem;color:#fff}
header .inner{max-width:920px;margin:0 auto}
header h1{margin:0 0 .35rem;font-size:1.7rem}
header p{margin:0;opacity:.92}
h2{font-size:1.25rem;margin:2.2rem 0 .8rem;border-bottom:1px solid var(--line);padding-bottom:.4rem}
h4{margin:1.2rem 0 .5rem;color:var(--mut);font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin-top:1.2rem}
.card{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--c);
border-radius:12px;padding:1.1rem 1.2rem}
.card h3{margin:0 0 .9rem;font-size:1.05rem;color:var(--c)}
.stat{display:flex;justify-content:space-between;align-items:baseline;padding:.3rem 0;
border-bottom:1px dashed var(--line)}
.stat span{font-weight:700;font-size:1.1rem;font-variant-numeric:tabular-nums}
.stat label{color:var(--mut);font-size:.8rem}
.notes{margin:.8rem 0 0;color:var(--mut);font-size:.8rem}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem}
.chip{background:color-mix(in srgb,var(--c) 22%,transparent);color:var(--c);
border:1px solid var(--c);border-radius:999px;padding:.25rem .8rem;font-weight:600;font-size:.85rem}
.arrow{color:var(--mut)}
.timeline{list-style:none;margin:0;padding:0;position:relative}
.timeline li{display:flex;gap:.85rem;padding:.5rem 0;position:relative}
.timeline li::before{content:"";position:absolute;left:5px;top:1.4rem;bottom:-.5rem;width:2px;
background:var(--line)}
.timeline li:last-child::before{display:none}
.dot{width:12px;height:12px;border-radius:50%;background:var(--c);margin-top:.35rem;flex:none;
box-shadow:0 0 0 3px color-mix(in srgb,var(--c) 25%,transparent)}
.payload{color:var(--mut);font-size:.82rem;font-family:ui-monospace,monospace}
details{margin-top:1rem;background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:.6rem .9rem}
summary{cursor:pointer;color:var(--mut);font-size:.85rem}
pre{white-space:pre-wrap;margin:.7rem 0 0;font-size:.82rem;color:var(--fg)}
.muted{color:var(--mut)}
.badges{margin:1rem 0 0}
.badge{display:inline-block;padding:.3rem .7rem;border-radius:999px;font-size:.78rem;font-weight:700;
letter-spacing:.03em}
.badge.live{background:rgba(16,185,129,.18);color:#34d399;border:1px solid #10b981}
.badge.synth{background:rgba(245,158,11,.16);color:#fbbf24;border:1px solid #f59e0b}
table.cmp{width:100%;border-collapse:collapse;margin:.8rem 0;font-size:.88rem}
table.cmp th,table.cmp td{padding:.5rem .7rem;border-bottom:1px solid var(--line);text-align:right;
font-variant-numeric:tabular-nums}
table.cmp thead th{text-align:right;color:var(--mut);font-size:.78rem;text-transform:uppercase;
letter-spacing:.04em;border-bottom:2px solid var(--line)}
table.cmp tbody th{text-align:left;font-weight:600;color:var(--fg)}
table.cmp tbody th small{color:var(--mut);font-weight:400}
table.cmp thead th:first-child{text-align:left}
table.cmp td.win{background:rgba(16,185,129,.16);color:#34d399;font-weight:700;border-radius:4px}
table.cmp td.delta{color:var(--mut);font-size:.82rem}
table.cmp td.delta.up{color:#34d399}
table.cmp td.delta.down{color:#f87171}
table.cmp tbody tr:hover{background:rgba(255,255,255,.03)}
""".strip()

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Multi-Agent Benchmark Report</title>"
        f"<style>{css}</style></head><body>"
        "<header><div class='inner'><h1>Multi-Agent Research Benchmark</h1>"
        f"<p>{escape(query) or 'Single-agent vs Multi-agent comparison'}</p></div></header>"
        "<div class='wrap'>"
        "<h2>Single-agent vs Multi-agent</h2>"
        f"<div class='cards'>{cards}</div>"
        "<h2>Full metric comparison</h2>"
        "<p class='muted'>Green cell = better for that metric. &Delta; is multi-agent minus single-agent.</p>"
        f"{table}"
        f"{sections}"
        "</div></body></html>"
    )

