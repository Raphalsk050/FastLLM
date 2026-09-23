"""Render the FastLLM benchmark charts as light and dark SVGs for the README.

Every number below was measured on the home test cluster (llama.cpp b11115).
Run from anywhere:  python docs/charts/make_charts.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

THEMES = {
    "light": {"surface": "#fcfcfb", "border": "rgba(11,11,11,0.10)", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "gray": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "border": "rgba(255,255,255,0.10)", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "gray": "#5f5e59",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}

W, PAD, BAR_H, BAR_GAP, ROW_PAD = 820, 24, 16, 2, 12

# Where the weights lived during a run; the same color means the same thing in every chart.
GPU, RAM, SPILL = 0, 1, 2
PLACEMENT = ["All in GPU memory", "Partly in system RAM", "GPU overflowed (Windows spilled VRAM to RAM)"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill="", weight=400, anchor="start", tabular=False, halo=None):
    """halo: surface color painted behind the glyphs so gridlines never cut through a label."""
    style = "font-variant-numeric: tabular-nums;" if tabular else ""
    ring = (f' stroke="{halo}" stroke-width="4" stroke-linejoin="round" paint-order="stroke"' if halo else "")
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" style="{style}"{ring}>{esc(s)}</text>')


def bar(x0, x1, y, fill):
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    w = x1 - x0
    if w <= 0:
        return ""
    r = min(4.0, w, BAR_H / 2)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} '
            f'V{y + BAR_H - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + BAR_H:.1f} H{x0:.1f} Z" fill="{fill}"/>')


def chart(name, title, subtitle, legend, rows, axis_max, tick_step, fmt, footnote, label_w=300, tick_fmt=None):
    """rows: ("group", text) | ("bars", label, [(value, color_key), ...], note_or_None).
    color_key is a series index or "gray" (de-emphasis)."""
    tick_fmt = tick_fmt or (lambda v: f"{v:g}")
    for theme, t in THEMES.items():
        out = []
        y = 34
        out.append(text(PAD, y, title, 16, t["ink"], 600))
        for line in subtitle:
            y += 20
            out.append(text(PAD, y, line, 12.5, t["ink2"]))
        if legend:
            y += 28
            x = PAD
            for i, label in legend:
                out.append(f'<rect x="{x}" y="{y - 10}" width="10" height="10" rx="2" fill="{t["series"][i]}"/>')
                out.append(text(x + 16, y, label, 12, t["ink2"]))
                x += 16 + len(label) * 6.6 + 24
        plot_left, plot_right = PAD + label_w, W - PAD - 78
        plot_top = y + 22
        scale = lambda v: plot_left + (plot_right - plot_left) * min(v, axis_max) / axis_max

        body, y = [], plot_top
        for row in rows:
            if row[0] == "group":
                y += 6
                body.append(text(PAD, y + 14, row[1], 12.5, t["ink"], 600))
                y += 24
                continue
            if row[0] == "sub":  # "Single PC" / "Cluster" split inside a group
                body.append(text(PAD + 14, y + 13, row[1], 11.5, t["ink2"], 600))
                y += 20
                continue
            _, label, values, note = row
            block_h = len(values) * BAR_H + (len(values) - 1) * BAR_GAP
            body.append(text(plot_left - 12, y + block_h / 2 + 4, label, 12, t["ink"], anchor="end"))
            by = y
            for value, key in values:
                fill = t["gray"] if key == "gray" else t["series"][key]
                x1 = scale(value)
                body.append(bar(plot_left, max(x1, plot_left + 1.5), by, fill))
                body.append(text(max(x1, plot_left + 1.5) + 6, by + BAR_H / 2 + 4, fmt(value), 12, t["ink2"],
                                 tabular=True, halo=t["surface"]))
                by += BAR_H + BAR_GAP
            if note:
                body.append(text(plot_left + 6, by + BAR_H / 2 + 4, note, 11.5, t["muted"], halo=t["surface"]))
                by += BAR_H + BAR_GAP
            y = by - BAR_GAP + ROW_PAD
        plot_bottom = y - ROW_PAD / 2

        grid = []
        v = 0.0
        while v <= axis_max + 1e-9:
            gx = scale(v)
            color = t["axis"] if v == 0 else t["grid"]
            grid.append(f'<line x1="{gx:.1f}" y1="{plot_top - 6:.1f}" x2="{gx:.1f}" y2="{plot_bottom:.1f}" '
                        f'stroke="{color}" stroke-width="1"/>')
            grid.append(text(gx, plot_bottom + 18, tick_fmt(v), 11, t["muted"], anchor="middle", tabular=True))
            v += tick_step
        out += grid + body

        y = plot_bottom + 18
        for line in footnote:
            y += 20
            out.append(text(PAD, y, line, 11, t["muted"]))
        height = y + PAD
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height:.0f}" '
               f'viewBox="0 0 {W} {height:.0f}" role="img" aria-labelledby="t-{name}" font-family="{FONT}">'
               f'<title id="t-{name}">{esc(title)}</title>'
               f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1:.0f}" rx="10" fill="{t["surface"]}" '
               f'stroke="{t["border"]}"/>' + "".join(out) + "</svg>\n")
        (OUT / f"{name}-{theme}.svg").write_text(svg, encoding="utf-8")


def tok(v):
    return f"{v:.1f} tok/s" if v < 100 else f"{v:.0f} tok/s"


chart(
    "hardware-memory", "Memory per machine",
    ["GPU memory and system RAM, in GB. On the MacBook the GPU uses the same 24 GB of unified memory",
     "(up to 17.8 GB), so that pool is not additive."],
    [(0, "GPU memory"), (1, "System RAM")],
    [("sub", "Single PC (also the cluster's main host)"),
     ("bars", "Ryzen 9 5950X · RTX 3070", [(8, 0), (64, 1)], None),
     ("sub", "Cluster workers"),
     ("bars", "Ryzen 5 5600X · RTX 2060", [(6, 0), (12, 1)], None),
     ("bars", "Pentium G4560 · GTX 1050 Ti", [(4, 0), (8, 1)], None),
     ("bars", "MacBook Air · Apple M3", [(17.8, 0), (24, 1)], None)],
    72, 16, lambda v: f"{v:g} GB",
    ["Windows 11 on the Ryzen PCs, Linux on the Pentium PC, macOS on the MacBook."],
)

chart(
    "hardware-bandwidth", "GPU memory bandwidth",
    ["GB/s. Writing each token reads the model weights from this memory, so it caps generation speed."],
    [],
    [("bars", "RTX 3070 · 8 GB", [(448, 0)], None),
     ("bars", "RTX 2060 · 6 GB", [(336, 0)], None),
     ("bars", "GTX 1050 Ti · 4 GB", [(112, 0)], None),
     ("bars", "Apple M3 · 24 GB unified", [(100, 0)], None)],
    500, 100, lambda v: f"{v:g} GB/s",
    ["Manufacturer figures. In a layer split, the slowest GPU that holds a big share of the model sets the pace."],
)

chart(
    "network-latency", "Round trip from each worker to the main PC",
    ["Median ping in milliseconds. The two worker PCs are wired (1 Gbps Ethernet); the MacBook is on Wi-Fi."],
    [],
    [("bars", "Ryzen 5 5600X · RTX 2060 · Ethernet", [(0.32, "gray")], None),
     ("bars", "Pentium G4560 · GTX 1050 Ti · Ethernet", [(0.29, "gray")], None),
     ("bars", "MacBook Air · Apple M3 · Wi-Fi", [(3.98, 0)], None)],
    4.5, 1, lambda v: f"{v:.2f} ms",
    ["Wi-Fi costs about 13x the latency of a cable. Copying files to the RTX 2060 PC over SSH ran at 90 MB/s."],
)

chart(
    "generation-speed", "Generation speed",
    ["Tokens per second while writing the answer. Color shows where the model weights lived."],
    [(GPU, PLACEMENT[GPU]), (RAM, PLACEMENT[RAM]), (SPILL, PLACEMENT[SPILL])],
    [("group", "Qwen3-14B Q4_K_M · 9.0 GB"),
     ("sub", "Single PC"),
     ("bars", "RTX 3070 + 64 GB RAM", [(7.9, RAM)], None),
     ("sub", "Cluster"),
     ("bars", "RTX 3070 + RTX 2060", [(26.7, GPU)], None),
     ("bars", "RTX 3070 + RTX 2060 · 8k context †", [(20.5, GPU)], None),
     ("bars", "RTX 3070 + RTX 2060 · 8k context, 1 GB margin †", [(0.5, SPILL)], None),
     ("bars", "RTX 3070 + GTX 1050 Ti", [(10.5, RAM)], None),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti + M3", [(7.1, GPU)], None),
     ("group", "Qwen3-32B Q4_K_M · 19.8 GB"),
     ("sub", "Single PC"),
     ("bars", "RTX 3070 + 64 GB RAM", [(1.9, RAM)], None),
     ("sub", "Cluster"),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti †", [(2.2, RAM)], None),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti + M3", [(4.7, GPU)], None),
     ("group", "Qwen3.8-Flash-Next UD-Q2_K_XL · 78.9 GB"),
     ("sub", "Cluster"),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti + M3 †", [(3.5, RAM)], None)],
    30, 5, tok,
    ["llama.cpp b11115; llama-bench tg128/tg32, † measured on the running server (16k context unless the label says 8k).",
     "The RTX 3070 PC (Ryzen 9 5950X, 64 GB) is the main host. Workers: RTX 2060 (Ryzen 5 5600X, 12 GB), GTX 1050 Ti",
     "(Pentium G4560, 8 GB) and MacBook Air M3 (24 GB, on Wi-Fi). Flash-Next is a 125B MoE that uses 6B parameters",
     "per token; about 42 GB of it sat in the main PC's RAM."],
    label_w=330, tick_fmt=lambda v: f"{v:g}",
)

chart(
    "prompt-speed", "Prompt processing speed",
    ["Tokens per second while reading the prompt: llama-bench pp512 unless the label says otherwise."],
    [(GPU, PLACEMENT[GPU]), (RAM, PLACEMENT[RAM])],
    [("group", "Qwen3-14B Q4_K_M · 9.0 GB"),
     ("sub", "Single PC"),
     ("bars", "RTX 3070 + 64 GB RAM", [(639.5, RAM)], None),
     ("sub", "Cluster"),
     ("bars", "RTX 3070 + RTX 2060", [(672.1, GPU)], None),
     ("bars", "RTX 3070 + GTX 1050 Ti", [(219.4, RAM)], None),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti + M3", [(100.1, GPU)], None),
     ("group", "Qwen3-32B Q4_K_M · 19.8 GB"),
     ("sub", "Single PC"),
     ("bars", "RTX 3070 + 64 GB RAM", [(210.3, RAM)], None),
     ("sub", "Cluster"),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti †", [(112, RAM)], None),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti + M3", [(53.2, GPU)], None),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti + M3 †", [(48, GPU)], None),
     ("group", "Qwen3.8-Flash-Next UD-Q2_K_XL · 78.9 GB"),
     ("sub", "Cluster"),
     ("bars", "RTX 3070 + RTX 2060 + GTX 1050 Ti + M3 ‡", [(10.9, RAM)], None)],
    700, 100, tok,
    ["† server, 2,500-token prompt. ‡ server, short prompt after warm-up; the first, cold prompt ran at 0.5 tok/s.",
     "Reading a prompt is compute-bound, so the slower GPUs and the M3's Wi-Fi hop hurt it more than generation.",
     "The server caches the prompt prefix: a repeated system prompt came back in 0.6-0.9 s."],
    label_w=330, tick_fmt=lambda v: f"{v:g}",
)

chart(
    "load-time", "Time until the server is ready",
    ["Seconds. The first load sends the weights over the network; later loads reuse each worker's disk cache."],
    [(0, "First load"), (1, "With the worker cache")],
    [("bars", "Qwen3-14B · RTX 3070 + RTX 2060", [(80, 0), (14, 1)], None),
     ("bars", "Qwen3-32B · all four machines", [(380, 0), (50, 1)], None),
     ("bars", "Qwen3.8-Flash-Next · all four machines", [(440, 0)], None)],
    500, 100, lambda v: f"{v:g} s",
    ["All four machines: RTX 3070, RTX 2060, GTX 1050 Ti and MacBook Air M3 (Wi-Fi). First loads of 14B and 32B are",
     "llama-bench runs minus the benchmark itself. The 32B cached reload sent only 0.3 GB over the network.",
     "Flash-Next's cached reload was not measured."],
)

print("charts written to", OUT)
