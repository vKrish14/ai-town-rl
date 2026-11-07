from __future__ import annotations

import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Random": "#b5a16a", "Rule": "#76926c", "PPO": "#6c9cac", "MAPPO": "#aa7f9f"}
METRICS = (
    ("reward", "Episode reward", "configured return"),
    ("wellbeing", "Final wellbeing", "score / 100"),
    ("mean_health", "Mean health", "score / 100"),
    ("survival_rate", "Survival", "percent"),
    ("employment_rate", "Employment", "percent"),
    ("mean_social_satisfaction", "Social satisfaction", "score / 100"),
    ("avg_money", "Average wealth", "currency units"),
    ("wealth_gini", "Wealth inequality", "Gini coefficient"),
    ("hunger_incidents", "Hunger incidents", "count"),
)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def spread(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def svg_bars(results, metric, title, unit):
    width, height, left, top, bottom = 760, 270, 58, 42, 52
    chart_h = height - top - bottom
    max_value = 100.0 if metric.endswith("_rate") or metric in ("wellbeing", "mean_health", "mean_social_satisfaction") else max(0.05, max((mean(r["episodes"], metric) or 0 for r in results), default=1) * 1.15)
    slot = (width - left - 20) / max(1, len(results))
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
              f'<text x="{left}" y="22" class="chart-title">{html.escape(title)} <tspan class="unit">· {html.escape(unit)}</tspan></text>']
    for tick in range(5):
        value = max_value * tick / 4
        y = top + chart_h * (1 - tick / 4)
        label = f"{value*100:.0f}%" if metric.endswith("_rate") else f"{value:.0f}" if max_value > 10 else f"{value:.2f}"
        pieces.append(f'<line x1="{left}" x2="{width-18}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="axis">{label}</text>')
    for index, result in enumerate(results):
        value = mean(result["episodes"], metric)
        if value is None:
            continue
        display = value * 100 if metric.endswith("_rate") else value
        bar_h = max(1.0, chart_h * min(1.0, display / max_value))
        x = left + index * slot + slot * .24
        y = top + chart_h - bar_h
        color = COLORS[result["label"]]
        pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*.52:.1f}" height="{bar_h:.1f}" rx="8" fill="{color}"/><text x="{x+slot*.26:.1f}" y="{max(top+16,y-8):.1f}" text-anchor="middle" class="value">{display:.1f}{"%" if metric.endswith("_rate") else ""}</text><text x="{x+slot*.26:.1f}" y="{height-21}" text-anchor="middle" class="axis">{result["label"]}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def svg_training(log_dir: Path, field: str, title: str):
    width, height, left, right, top, bottom = 760, 230, 56, 18, 40, 32
    data = {}
    for name in ("PPO", "MAPPO"):
        rows = []
        for line in (log_dir / f"train_{name.lower()}.jsonl").read_text(encoding="utf-8").splitlines() if (log_dir / f"train_{name.lower()}.jsonl").exists() else []:
            try:
                row = json.loads(line)
                if field in row:
                    rows.append((int(row.get("episode", len(rows) + 1)), float(row[field])))
            except (ValueError, json.JSONDecodeError):
                continue
        if rows:
            data[name] = rows
    values = [v for rows in data.values() for _, v in rows]
    if not values:
        return f'<div class="empty">No recorded {html.escape(title.lower())} traces were found.</div>'
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1.0
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}"><text x="{left}" y="22" class="chart-title">{html.escape(title)}</text>']
    for i in range(5):
        y = top + plot_h * i / 4
        val = high - (high - low) * i / 4
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="axis">{val:.1f}</text>')
    for name, rows in data.items():
        points = []
        x_max = max(1, max(ep for ep, _ in rows))
        for ep, val in rows:
            x = left + plot_w * ep / x_max
            y = top + plot_h * (high - val) / (high - low)
            points.append(f"{x:.1f},{y:.1f}")
        color = COLORS[name]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/><text x="{width-right-80}" y="{top+18+(0 if name == "PPO" else 20)}" class="legend" fill="{color}">● {name}</text>')
    parts.append(f'<text x="{width/2}" y="{height-5}" text-anchor="middle" class="axis">Training episode</text></svg>')
    return "".join(parts)


def build_dashboard(results_dir: str | Path = ROOT / "results", log_dir: str | Path = ROOT / "logs", output: str | Path | None = None) -> Path:
    results_dir, log_dir = Path(results_dir), Path(log_dir)
    results = []
    for label in ("Random", "Rule", "PPO", "MAPPO"):
        record = load_json(results_dir / f"{label.lower()}.json")
        if record and record.get("episodes"):
            results.append({"label": label, "record": record, "episodes": record["episodes"]})
    cards = []
    for result in results:
        record = result["record"]
        seeds = ", ".join(str(row.get("seed", "?")) for row in result["episodes"])
        training = record.get("checkpoint_metadata") or {}
        training_seed = training.get("seed", "—")
        training_episodes = training.get("episodes", "—")
        cards.append(
            f'<tr><th><span class="dot" style="background:{COLORS[result["label"]]}"></span>{result["label"]}</th>'
            f'<td>{html.escape(record.get("mode", result["label"]))}</td>'
            f'<td>{html.escape(str(record.get("checkpoint", "baseline")))}</td>'
            f'<td>{html.escape(str(training_seed))}</td><td>{html.escape(str(training_episodes))}</td>'
            f'<td>{html.escape(str(record.get("days", "?")))}</td><td>{len(result["episodes"])}</td>'
            f'<td>{html.escape(seeds)}</td><td>{html.escape(record.get("decision_mode", "baseline policy"))}</td></tr>'
        )
    metric_headers = "".join(f"<th>{html.escape(title)}</th>" for _, title, _ in METRICS)
    metric_rows = []
    for result in results:
        cells = []
        for key, _, _ in METRICS:
            value = mean(result["episodes"], key)
            deviation = spread(result["episodes"], key)
            if value is None:
                shown = "—"
            elif key.endswith("_rate"):
                shown = f"{value*100:.1f}% ± {deviation*100:.1f}"
            elif key == "wealth_gini":
                shown = f"{value:.3f} ± {deviation:.3f}"
            else:
                shown = f"{value:.1f} ± {deviation:.1f}"
            cells.append(f"<td>{shown}</td>")
        metric_rows.append(f'<tr><th>{result["label"]}</th>{"".join(cells)}</tr>')
    chart_sections = "".join(f'<article class="chart-card">{svg_bars(results, key, title, unit)}</article>' for key, title, unit in METRICS)
    training = f'<article class="chart-card wide">{svg_training(log_dir, "mean_reward", "Training reward by episode")}</article><article class="chart-card wide">{svg_training(log_dir, "mean_wellbeing", "Training wellbeing by episode")}</article>'
    primary = results[0]["record"] if results else {}
    config = primary.get("config", {})
    simulation = config.get("simulation", {})
    economy = config.get("economy", {})
    reward = config.get("reward", {})
    conditions = (
        f"Population {simulation.get('population', 'unknown')} · {simulation.get('ticks_per_day', 'unknown')} ticks/day · "
        f"Food price ${economy.get('food_price', 'unknown')} · Reward weights {html.escape(json.dumps(reward, sort_keys=True))}"
    )
    output_path = Path(output) if output else results_dir / "research_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Willowbrook · Research dashboard</title>
<style>
:root{{--ink:#243b31;--muted:#69796c;--paper:#f4f0df;--card:#fffdf3;--line:#ded8c2;--green:#496c55;--gold:#d4b371}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 'Segoe UI',sans-serif}}header{{background:#2d4a3d;color:#f8f3e2;padding:36px max(24px,calc((100vw - 1160px)/2));border-bottom:4px solid var(--gold)}}header p{{margin:7px 0 0;color:#d6e0c8}}main{{max-width:1160px;margin:28px auto;padding:0 22px}}h1{{margin:0;font-size:30px}}h2{{font-size:20px;margin:0 0 14px}}.section{{margin:24px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 5px 20px #354a3510}}.grid{{stroke:#e5e1d2;stroke-width:1}}.chart-title{{fill:#344a3b;font-weight:700;font-size:15px}}.unit,.axis{{fill:#798576;font-size:11px}}.value{{fill:#263d31;font-size:13px;font-weight:700}}.legend{{font-size:12px;font-weight:700}}.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}svg{{display:block;width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px 9px;border-bottom:1px solid #e8e2d0;text-align:left;vertical-align:top}}thead th{{color:#627260;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}}.note{{color:var(--muted);font-size:13px}}.wide{{grid-column:1/-1}}.empty{{padding:18px;color:var(--muted)}}@media(max-width:760px){{.charts{{grid-template-columns:1fr}}main{{margin:16px auto}}header{{padding:24px}}table{{font-size:11px}}th,td{{padding:7px 5px}}}}
</style></head><body><header><h1>Willowbrook · Research dashboard</h1><p>Recorded evaluation outcomes and training diagnostics for the AI Town multi-agent study.</p></header><main>
<section class="section card"><h2>Evaluation conditions</h2><p>{conditions}</p><p class="note">Every chart is generated from the saved evaluation episode rows in results/*.json. Table values show mean ± standard deviation across evaluation environment seeds. This does not measure variation across independently trained checkpoints. Training curves use logs/train_ppo.jsonl and logs/train_mappo.jsonl. These runs do not establish convergence or broad generalization. Reward values should be interpreted with their configured reward components.</p></section>
<section class="section card"><h2>Run provenance</h2><div style="overflow:auto"><table><thead><tr><th>Policy</th><th>Recorded label</th><th>Checkpoint</th><th>Training seed</th><th>Train episodes</th><th>Days</th><th>Episodes</th><th>Evaluation seeds</th><th>Decision mode</th></tr></thead><tbody>{''.join(cards)}</tbody></table></div></section>
<section class="section card"><h2>Evaluation metrics</h2><div style="overflow:auto"><table><thead><tr><th>Policy</th>{metric_headers}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div></section>
<section class="section"><h2>Policy outcomes</h2><div class="charts">{chart_sections}</div></section>
<section class="section"><h2>Learning diagnostics</h2><p class="note">Training reward and wellbeing are diagnostics from training episodes, shown separately from held-out policy evaluation.</p><div class="charts">{training}</div></section>
<footer class="section note">Source files remain in the project results/ and logs/ directories. Rebuild with <code>python -m experiments.research_dashboard</code>.</footer></main></body></html>'''
    output_path.write_text(document, encoding="utf-8")
    return output_path.resolve()


if __name__ == "__main__":
    print(build_dashboard())
