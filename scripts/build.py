#!/usr/bin/env python3
"""Draw the figures on the profile README and refresh its upstream-PR table.

Standard library only. The banner and the project cards are static; the
contribution figure and the upstream table need a GITHUB_TOKEN (or GH_TOKEN)
and are skipped without one.

    python3 scripts/build.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import urllib.request
from datetime import date
from html import escape
from pathlib import Path

USER = "ColinHouse"
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
README = ROOT / "README.md"

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,'Noto Sans',sans-serif"
SERIF = "Georgia,'Times New Roman',Times,serif"
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"
KANA = f"'Hiragino Sans','Yu Gothic','Noto Sans CJK JP','Noto Sans JP',{SANS}"

# Blue and cream are sampled from the avatar; the rest is derived from them.
# Each figure is one SVG that carries both palettes and switches on
# prefers-color-scheme, which browsers resolve against the page's colour scheme.
PALETTE = {
    #          light      dark
    "surface": ("#fbf9f5", "#131a2c"),
    "panel": ("#f1ede4", "#19223b"),
    "border": ("#e2dbce", "#283355"),
    "ink": ("#1f2a44", "#f0eadf"),
    "text": ("#414a63", "#c3cadb"),
    "muted": ("#7d8497", "#8790a7"),
    "blue": ("#3b58a8", "#8fa8f0"),
    "sand": ("#a8753f", "#dbb488"),
    "tile": ("#ffffff", "#222d4d"),
    "sky0": ("#4463b8", "#34509c"),
    "sky1": ("#324d98", "#243a78"),
}
CREAM = "#f3ece5"
VARS = (
    ":root{" + ";".join(f"--{k}:{v[0]}" for k, v in PALETTE.items()) + "}"
    "@media (prefers-color-scheme:dark){:root{" + ";".join(f"--{k}:{v[1]}" for k, v in PALETTE.items()) + "}}"
)


def num(value: float) -> str:
    """Format a coordinate with at most two decimals and no trailing zeros."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def svg(width: int, height: int, title: str, css: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">\n'
        f"<title>{escape(title)}</title>\n<style>\n{VARS}\n{css.strip()}\n</style>\n{body.strip()}\n</svg>\n"
    )


# --------------------------------------------------------------------------
# Fig. 1: the banner, a stylized mel spectrogram of the word "syrinx"
# --------------------------------------------------------------------------

DURATION = 0.88
# A rough hand segmentation of /ˈsɪrɪŋks/, in seconds.
PHONES = [
    ("s", 0.04, 0.19),
    ("ɪ", 0.19, 0.31),
    ("r", 0.31, 0.39),
    ("ɪ", 0.39, 0.49),
    ("ŋ", 0.49, 0.59),
    ("k", 0.59, 0.67),
    ("s", 0.67, 0.84),
]
# time, F1, F2, F3 (Hz), amplitude, weight of the upper formants
VOICED = [
    (0.180, 380, 1850, 2550, 0.00, 1.00),
    (0.205, 400, 1950, 2600, 0.95, 1.00),
    (0.250, 420, 2050, 2650, 1.00, 1.00),
    (0.350, 340, 1150, 1650, 0.80, 0.90),
    (0.440, 400, 1950, 2550, 0.90, 1.00),
    (0.500, 320, 2200, 2420, 0.62, 0.70),
    (0.545, 270, 2100, 2500, 0.42, 0.32),
    (0.580, 260, 2050, 2450, 0.30, 0.25),
    (0.600, 250, 2000, 2400, 0.00, 0.20),
]
MEL_TOP = 8000.0


def mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def voiced_frame(t: float):
    """Interpolate the formant track at time t, or None outside voicing."""
    if t <= VOICED[0][0] or t >= VOICED[-1][0]:
        return None
    for a, b in zip(VOICED, VOICED[1:]):
        if a[0] <= t <= b[0]:
            k = smooth((t - a[0]) / (b[0] - a[0]))
            return [a[i] + (b[i] - a[i]) * k for i in range(1, 6)]
    return None


def plateau(t: float, start: float, end: float, ramp: float = 0.03) -> float:
    return smooth((t - start) / ramp) * smooth((end - t) / ramp)


def energy(t: float, hz: float, rng: random.Random) -> float:
    """Energy in [0, 1] at time t (s) and frequency hz."""
    m = mel(hz)
    e = 0.0
    frame = voiced_frame(t)
    if frame:
        f1, f2, f3, amp, upper = frame
        peaks = (
            (f1, 95.0, 1.0),
            (f2, 105.0, 0.86 * upper),
            (f3, 110.0, 0.62 * upper),
        )
        formants = max(w * math.exp(-0.5 * ((m - mel(f)) / s) ** 2) for f, s, w in peaks)
        voice_bar = 0.6 * math.exp(-0.5 * (m / 150.0) ** 2)
        e = max(e, amp * max(formants, voice_bar))
    # /s/: frication above about 3.5 kHz
    sibilant = smooth((hz - 3000.0) / 1800.0) * (0.62 + 0.38 * rng.random())
    e = max(e, 0.84 * plateau(t, 0.045, 0.195) * sibilant)
    e = max(e, 0.72 * plateau(t, 0.675, 0.835) * sibilant)
    # /k/: silence, then a short burst centred in the mid frequencies
    burst = math.exp(-(((t - 0.652) / 0.011) ** 2))
    shape = 0.3 + 0.7 * math.exp(-0.5 * ((m - mel(2600.0)) / 330.0) ** 2)
    e = max(e, 0.8 * burst * shape * (0.7 + 0.3 * rng.random()))
    return min(1.0, e)


def banner() -> str:
    width, height = 846, 270
    pitch = 9.0
    cols, rows = 49, 20
    x0, y0 = 366.0, 24.0  # top-left of the spectrogram
    spec_w, spec_h = cols * pitch, rows * pitch
    rng = random.Random(7)
    sweep = 0.062  # seconds the playhead spends on one column

    columns = []
    for c in range(cols):
        t = (c + 0.5) / cols * DURATION
        dots = []
        for r in range(rows):
            hz = 700.0 * (10 ** ((rows - r - 0.5) / rows * mel(MEL_TOP) / 2595.0) - 1.0)
            e = energy(t, hz, rng)
            if e < 0.07:
                continue
            radius = 0.75 + 3.3 * e**0.8
            dots.append(
                f'<circle cx="{num(x0 + (c + 0.5) * pitch)}" '
                f'cy="{num(y0 + (r + 0.5) * pitch)}" r="{num(radius)}"/>'
            )
        if dots:
            columns.append(
                f'<g class="col" style="animation-delay:{num(c * sweep)}s">{"".join(dots)}</g>'
            )

    tier_y = y0 + spec_h + 9
    ticks, labels = [], []
    for i, (label, start, end) in enumerate(PHONES):
        xs = x0 + start / DURATION * spec_w
        xe = x0 + end / DURATION * spec_w
        ticks.append(f"M{num(xs)} {num(tier_y)}v7")
        if i == len(PHONES) - 1:
            ticks.append(f"M{num(xe)} {num(tier_y)}v7")
        delay = (start + end) / 2 / DURATION * cols * sweep
        labels.append(
            f'<text class="ph" x="{num((xs + xe) / 2)}" y="{num(tier_y + 22)}" '
            f'style="animation-delay:{num(delay)}s">{label}</text>'
        )

    axis = []
    for hz, name in ((8000, "8k"), (4000, "4k"), (2000, "2k"), (1000, "1k")):
        y = y0 + spec_h * (1 - mel(hz) / mel(MEL_TOP))
        axis.append(f'<text x="{num(x0 + spec_w + 8)}" y="{num(y + 6)}">{name}</text>')

    css = f"""
text{{fill:{CREAM}}}
.col{{fill:{CREAM};opacity:.8;animation:glow 7s linear infinite}}
.ph{{font:500 14px {SANS};opacity:.72;text-anchor:middle;animation:glow 7s linear infinite}}
.axis{{font:9px {MONO};opacity:.5}}
.fig{{font:10px {MONO};letter-spacing:.14em;opacity:.6}}
.word{{font:700 84px {SERIF};letter-spacing:-.01em}}
.ipa{{font:19px {SANS};opacity:.9}}
.gloss{{font:italic 17px {SERIF};opacity:.8}}
.head{{fill:{CREAM};animation:sweep 7s linear infinite}}
@keyframes glow{{0%,9%,100%{{opacity:.8}}3%{{opacity:1}}}}
@keyframes sweep{{0%{{transform:translateX(0);opacity:0}}2%{{opacity:.55}}41%{{opacity:.55}}43.4%{{transform:translateX({num(spec_w)}px);opacity:0}}100%{{transform:translateX({num(spec_w)}px);opacity:0}}}}
@media (prefers-reduced-motion:reduce){{.col,.ph,.head{{animation:none}}.head{{opacity:0}}}}
"""
    body = f"""
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="1" y2="1"><stop offset="0" style="stop-color:var(--sky0)"/><stop offset="1" style="stop-color:var(--sky1)"/></linearGradient>
<pattern id="beads" x="{num(x0 % pitch)}" y="{num(y0 % pitch)}" width="{num(pitch)}" height="{num(pitch)}" patternUnits="userSpaceOnUse"><circle cx="{num(pitch / 2)}" cy="{num(pitch / 2)}" r="0.8" fill="{CREAM}"/></pattern>
<clipPath id="frame"><rect width="{width}" height="{height}" rx="14"/></clipPath>
</defs>
<g clip-path="url(#frame)">
<rect width="{width}" height="{height}" fill="url(#sky)"/>
<rect width="{width}" height="{height}" fill="url(#beads)" opacity=".13"/>
{"".join(columns)}
<rect class="head" x="{num(x0)}" y="{num(y0 - 4)}" width="1.5" height="{num(spec_h + 8)}"/>
<path d="M{num(x0)} {num(tier_y)}h{num(spec_w)}{"".join(ticks)}" fill="none" stroke="{CREAM}" stroke-opacity=".45"/>
{"".join(labels)}
<g class="axis">{"".join(axis)}</g>
<text class="fig" x="40" y="50">FIG. 1 — MEL SPECTROGRAM, STYLIZED</text>
<text class="word" x="38" y="143">Syrinx.</text>
<text class="ipa" x="41" y="182">/ˈsɪrɪŋks/<tspan class="gloss" dx="10">noun</tspan></text>
<text class="gloss" x="41" y="210">the vocal organ of birds</text>
</g>
"""
    return svg(width, height, "Syrinx. /ˈsɪrɪŋks/, with a stylized mel spectrogram of the word", css, body)


# --------------------------------------------------------------------------
# Project cards
# --------------------------------------------------------------------------

LANG_COLORS = {"Java": "#b07219", "Python": "#3572a5"}

SPRIG_CODE = [
    [("k", "variant"), ("", " "), ("t", "Shape"), ("p", ":")],
    [("", "    "), ("t", "Circle"), ("p", "("), ("", "radius"), ("p", ": "), ("t", "Float"), ("p", ")")],
    [
        ("", "    "), ("t", "Rect"), ("p", "("), ("", "width"), ("p", ": "), ("t", "Float"),
        ("p", ", "), ("", "height"), ("p", ": "), ("t", "Float"), ("p", ")"),
    ],
    [],
    [
        ("k", "func"), ("", " area"), ("p", "("), ("", "shape"), ("p", ": "), ("t", "Shape"),
        ("p", ") -> "), ("t", "Float"), ("p", ":"),
    ],
    [("", "    "), ("k", "match"), ("", " shape"), ("p", ":")],
    [
        ("", "        "), ("k", "case"), ("", " "), ("t", "Shape.Circle"), ("", " "),
        ("k", "as"), ("", " circle"), ("p", ":"),
    ],
]


def art_code() -> tuple[str, str]:
    lines = []
    for i, tokens in enumerate(SPRIG_CODE):
        if not tokens:
            continue
        spans = "".join(
            f'<tspan class="{kind}">{escape(text)}</tspan>' if kind else escape(text)
            for kind, text in tokens
        )
        lines.append(f'<text x="20" y="{num(31 + i * 14.2)}" xml:space="preserve">{spans}</text>')
    css = f"""
.code{{font:10px {MONO};fill:var(--text);white-space:pre}}
.k{{fill:var(--blue);font-weight:700}}.t{{fill:var(--ink)}}.p{{fill:var(--muted)}}
.fade0{{stop-color:var(--panel);stop-opacity:0}}.fade1{{stop-color:var(--panel)}}
"""
    body = f"""
<linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" class="fade0"/><stop offset="1" class="fade1"/></linearGradient>
<g class="code">{"".join(lines)}</g>
<rect y="88" width="272" height="40" fill="url(#fade)"/>
"""
    return css, body


def art_tiles() -> tuple[str, str]:
    tiles = []
    for i, (kana, turn) in enumerate(zip("ことばこ", (-4, 3, -3, 4))):
        cx, cy = 58 + i * 52, 52
        hot = " hot" if i == 2 else ""
        tiles.append(
            f'<g class="tile{hot}" transform="rotate({turn} {cx} {cy})">'
            f'<rect class="shade" x="{cx - 21}" y="{cy - 23}" width="42" height="50" rx="8"/>'
            f'<rect class="face" x="{cx - 21}" y="{cy - 26}" width="42" height="50" rx="8"/>'
            f'<text x="{cx}" y="{cy + 7}">{kana}</text></g>'
        )
    # review dates drift apart, the way a spaced-repetition schedule does
    dots = "".join(
        f'<circle class="{"done" if i < 3 else "due"}" cx="{x}" cy="103" r="3.2"/>'
        for i, x in enumerate((37, 47, 63, 89, 131, 199))
    )
    css = f"""
.tile text{{font:600 24px {KANA};text-anchor:middle;fill:var(--ink)}}
.shade{{fill:var(--border)}}.face{{fill:var(--tile);stroke:var(--border)}}
.hot .face{{fill:var(--blue);stroke:var(--blue)}}.hot text{{fill:var(--surface)}}
.line{{stroke:var(--border);stroke-width:1.4}}
.done,.due{{stroke:var(--blue);stroke-width:1.4}}.done{{fill:var(--blue)}}.due{{fill:var(--panel);stroke-opacity:.55}}
"""
    body = f'<g lang="ja">{"".join(tiles)}</g><path class="line" d="M37 103H235"/>{dots}'
    return css, body


def art_ledger() -> tuple[str, str]:
    rows = [("給料", "+280,000", "in"), ("食費", "−3,200", "out"), ("交通費", "−480", "out")]
    out = []
    for i, (label, amount, kind) in enumerate(rows):
        y = 37 + i * 22
        out.append(
            f'<circle class="{kind}" cx="28" cy="{y - 4}" r="2.6"/>'
            f'<text class="label" x="38" y="{y}">{label}</text>'
            f'<text class="amount {kind}" x="246" y="{y}">{amount}</text>'
            f'<path class="rule" d="M24 {y + 8}H248"/>'
        )
    css = f"""
.label{{font:11.5px {KANA};fill:var(--text)}}
.amount{{font:11px {MONO};text-anchor:end;fill:var(--text)}}
circle.in,.amount.in{{fill:var(--blue)}}circle.out{{fill:var(--sand)}}
.rule{{stroke:var(--border)}}
.total{{font-weight:700;fill:var(--ink)}}
"""
    body = (
        f'<g lang="ja">{"".join(out)}'
        '<text class="label total" x="24" y="112">残高</text>'
        '<text class="amount total" x="246" y="112">¥276,320</text></g>'
    )
    return css, body


CARDS = [
    {
        "file": "sprig",
        "art": art_code,
        "kicker": "LANGUAGE · JVM",
        "name": "Sprig",
        "en": ["A statically typed JVM language", "for people and coding agents:", "less to guess, easier to review."],
        "zh": ["面向人和编码 Agent 的 JVM 静态类型", "语言：少一点猜测，方便审查。"],
        "lang": "Java",
        "tags": "ANTLR4 · Fabric",
    },
    {
        "file": "kotobako",
        "art": art_tiles,
        "kicker": "APP · LOCAL-FIRST",
        "name": "kotobako",
        "en": ["A context-keeping Japanese", "reading companion: screenshot", "+ audio capture, FSRS review."],
        "zh": ["会记住语境的日语伴读工具：", "截图 + 原声收藏台词，FSRS 复习。"],
        "lang": "Python",
        "tags": "FastAPI · Vue 3 · PWA",
    },
    {
        "file": "ledger",
        "art": art_ledger,
        "kicker": "API · SPRING BOOT",
        "name": "Personal Ledger API",
        "en": ["A household-ledger REST API:", "accounts, categories, records", "and monthly budgets."],
        "zh": ["Spring Boot + MyBatis-Plus", "实现的家計簿 API。"],
        "lang": "Java",
        "tags": "MySQL · Docker",
    },
]


def card(spec: dict) -> str:
    width, height = 272, 338
    art_css, art = spec["art"]()
    en = "".join(f'<text class="en" x="20" y="{212 + i * 18}">{escape(s)}</text>' for i, s in enumerate(spec["en"]))
    zh = "".join(f'<text class="zh" x="20" y="{273 + i * 17}">{escape(s)}</text>' for i, s in enumerate(spec["zh"]))
    css = f"""
.card{{fill:var(--surface);stroke:var(--border)}}
.art{{fill:var(--panel)}}
.rule{{stroke:var(--border)}}
.kicker{{font:9.5px {MONO};letter-spacing:.12em;fill:var(--sand)}}
.name{{font:700 21px {SERIF};fill:var(--ink)}}
.go{{fill:none;stroke:var(--muted);stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round}}
.en{{font:13px {SANS};fill:var(--text)}}
.zh{{font:12px {SANS};fill:var(--muted)}}
.meta{{font:11.5px {SANS};fill:var(--text)}}.meta tspan{{fill:var(--muted)}}
{art_css.strip()}
"""
    body = f"""
<defs><clipPath id="top"><path d="M0 12a12 12 0 0 1 12-12h248a12 12 0 0 1 12 12v116H0z"/></clipPath></defs>
<rect class="card" x=".5" y=".5" width="{width - 1}" height="{height - 1}" rx="12"/>
<g clip-path="url(#top)"><rect class="art" x="1" y="1" width="{width - 2}" height="127"/>
{art.strip()}</g>
<path class="rule" d="M1 128H271"/>
<text class="kicker" x="20" y="157">{escape(spec["kicker"])}</text>
<text class="name" x="20" y="186">{escape(spec["name"])}</text>
<path class="go" d="M244 158l8-8m-6.5 0h6.5v6.5"/>
{en}
<g lang="zh-Hans">{zh}</g>
<circle cx="24.5" cy="313" r="4.5" fill="{LANG_COLORS[spec["lang"]]}"/>
<text class="meta" x="35" y="317">{spec["lang"]}<tspan dx="10">{escape(spec["tags"])}</tspan></text>
"""
    return svg(width, height, f'{spec["name"]}: {" ".join(spec["en"])}', css, body)


# --------------------------------------------------------------------------
# Fig. 2: contributions drawn as a waveform
# --------------------------------------------------------------------------

DAYS = 84
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def activity(data: dict) -> str:
    width, height = 846, 196
    days = data["days"][-DAYS:]
    peak = max((count for _, count in days), default=0) or 1
    x0, x1, mid, reach = 272.0, 818.0, 96.0, 48.0
    pitch = (x1 - x0) / len(days)
    bar = 3.4

    marks, months = [], []
    best = max(range(len(days)), key=lambda i: days[i][1])
    for i, (day, count) in enumerate(days):
        cx = x0 + (i + 0.5) * pitch
        if day.endswith("-01"):
            months.append(
                f'<path d="M{num(cx)} {num(mid + reach + 8)}v5"/>'
                f'<text x="{num(cx + 5)}" y="{num(mid + reach + 14)}">{MONTHS[int(day[5:7]) - 1]}</text>'
            )
        if count == 0:
            marks.append(f'<circle class="rest" cx="{num(cx)}" cy="{num(mid)}" r="1.1"/>')
            continue
        half = 2.5 + (reach - 2.5) * math.log1p(count) / math.log1p(peak)
        marks.append(
            f'<rect class="bar" x="{num(cx - bar / 2)}" y="{num(mid - half)}" width="{num(bar)}" '
            f'height="{num(half * 2)}" rx="{num(bar / 2)}" style="animation-delay:{num(i * 0.03)}s"/>'
        )
    peak_x = x0 + (best + 0.5) * pitch
    peak_day = date.fromisoformat(days[best][0])
    peak_note = f"{peak} on {MONTHS[peak_day.month - 1].title()} {peak_day.day}"
    anchor = "end" if peak_x > x1 - 60 else "middle"

    css = f"""
.card{{fill:var(--surface);stroke:var(--border)}}
.kicker{{font:10px {MONO};letter-spacing:.14em;fill:var(--muted)}}
.big{{font:700 54px {SERIF};fill:var(--ink)}}
.cap{{font:13px {SANS};fill:var(--text)}}
.zh{{font:12px {SANS};fill:var(--muted)}}
.stat{{font:12px {SANS};fill:var(--muted)}}.stat tspan{{font-weight:600;fill:var(--ink)}}
.axis{{stroke:var(--border);stroke-dasharray:1 5}}
.month text,.note{{font:9px {MONO};letter-spacing:.08em;fill:var(--muted)}}
.month path{{stroke:var(--border)}}
.note{{fill:var(--sand)}}
.rest{{fill:var(--muted);opacity:.55}}
.bar{{fill:var(--blue);animation:play 9s linear infinite}}
@keyframes play{{0%,5%,100%{{fill:var(--blue)}}1.6%{{fill:var(--sand)}}}}
@media (prefers-reduced-motion:reduce){{.bar{{animation:none}}}}
"""
    body = f"""
<rect class="card" x=".5" y=".5" width="{width - 1}" height="{height - 1}" rx="12"/>
<text class="kicker" x="28" y="38">FIG. 2 — CONTRIBUTIONS, LAST 12 WEEKS</text>
<text class="big" x="26" y="104">{data["total"]:,}</text>
<text class="cap" x="28" y="128">contributions in the past year</text>
<text class="zh" x="28" y="146" lang="zh-Hans">过去一年的贡献，右侧为近 12 周的每日波形</text>
<text class="stat" x="28" y="174"><tspan>{data["prs"]:,}</tspan> pull requests<tspan dx="12">{data["issues"]:,}</tspan> issues<tspan dx="12">{data["upstream"]:,}</tspan> merged upstream</text>
<path class="axis" d="M{num(x0)} {num(mid)}H{num(x1)}"/>
{"".join(marks)}
<text class="note" x="{num(peak_x)}" y="{num(mid - reach - 7)}" text-anchor="{anchor}">{peak_note}</text>
<g class="month">{"".join(months)}</g>
"""
    return svg(width, height, f'{data["total"]} contributions in the past year, drawn as a waveform', css, body)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

QUERY = """
query($login: String!, $prs: String!, $issues: String!, $merged: String!) {
  user(login: $login) {
    contributionsCollection {
      contributionCalendar {
        totalContributions
        weeks { contributionDays { date contributionCount } }
      }
    }
  }
  prs: search(query: $prs, type: ISSUE) { issueCount }
  issues: search(query: $issues, type: ISSUE) { issueCount }
  merged: search(query: $merged, type: ISSUE, first: 100) {
    nodes {
      ... on PullRequest {
        number title url mergedAt
        repository { nameWithOwner url isPrivate stargazerCount }
      }
    }
  }
}
"""


def fetch(token: str) -> dict:
    variables = {
        "login": USER,
        "prs": f"author:{USER} is:pr is:public",
        "issues": f"author:{USER} is:issue is:public",
        "merged": f"author:{USER} is:pr is:merged is:public -user:{USER} sort:updated-desc",
    }
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": QUERY, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {token}", "User-Agent": f"{USER}-profile"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise SystemExit(f"GitHub API error: {payload['errors']}")
    result = payload["data"]
    calendar = result["user"]["contributionsCollection"]["contributionCalendar"]
    merged = [
        pr for pr in result["merged"]["nodes"]
        if pr and not pr["repository"]["isPrivate"]
    ]
    return {
        "total": calendar["totalContributions"],
        "days": [
            (day["date"], day["contributionCount"])
            for week in calendar["weeks"]
            for day in week["contributionDays"]
        ],
        "prs": result["prs"]["issueCount"],
        "issues": result["issues"]["issueCount"],
        "upstream": len(merged),
        "merged": merged,
    }


def stars(count: int) -> str:
    return f"{count / 1000:.0f}k" if count >= 10000 else f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def upstream_table(merged: list) -> str:
    repos: dict = {}
    for pr in merged:
        repos.setdefault(pr["repository"]["nameWithOwner"], []).append(pr)
    ordered = sorted(repos.values(), key=lambda prs: -prs[0]["repository"]["stargazerCount"])
    lines = ["| Project | Merged | Latest pull request |", "|:--|:-:|:--|"]
    for prs in ordered:
        repo = prs[0]["repository"]
        latest = max(prs, key=lambda pr: pr["mergedAt"])
        query = f"is%3Apr+is%3Amerged+author%3A{USER}"
        title = latest["title"].replace("|", "\\|")
        lines.append(
            f'| [**{repo["nameWithOwner"]}**]({repo["url"]})&nbsp;<sub>★&nbsp;{stars(repo["stargazerCount"])}</sub> '
            f'| [{len(prs)}]({repo["url"]}/pulls?q={query}) '
            f'| [{title}]({latest["url"]}) |'
        )
    return "\n".join(lines)


def patch_readme(table: str) -> None:
    text = README.read_text(encoding="utf-8")
    block = f"<!-- upstream:start -->\n{table}\n<!-- upstream:end -->"
    patched, count = re.subn(
        r"<!-- upstream:start -->.*?<!-- upstream:end -->", lambda _: block, text, flags=re.S
    )
    if count and patched != text:
        README.write_text(patched, encoding="utf-8")
        print("updated the upstream table in README.md")


def write(name: str, content: str) -> None:
    path = ASSETS / name
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    write("banner.svg", banner())
    for spec in CARDS:
        write(f'{spec["file"]}.svg', card(spec))

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("no GITHUB_TOKEN: skipped the contribution figure and the upstream table", file=sys.stderr)
        return
    data = fetch(token)
    write("activity.svg", activity(data))
    patch_readme(upstream_table(data["merged"]))


if __name__ == "__main__":
    main()
