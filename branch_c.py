# =============================================
# C ブランチ: Character（テキスト判定）
# =============================================
"""文字種バランスを判定し、キャラクター性・文体のスコアを返す。"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("snc.c")

LABELS = [
    ("漢字多め", "漢文調・硬い専門文に寄りやすい", 0.7),
    ("ひらがな多め", "柔らかい口語・読みやすさ寄り", 0.8),
    ("カタカナ多め", "外来語・テクニカル寄り", 0.6),
    ("英数多め", "コード・固有名詞・英語混じり", 0.5),
    ("バランス型", "文字種が比較的均等", 1.0),
]


def validate_text(text: Optional[str]) -> Tuple[bool, str]:
    if text is None:
        return False, "テキストが指定されていません"
    s = text.strip()
    if not s:
        return False, "テキストが空です"
    if "\ufffd" in s:
        return False, "文字化けの可能性があるテキストです"
    if len(s) > 50_000:
        return False, "テキストが長すぎます"
    return True, ""


def char_type_ratios(text: str) -> Dict[str, float]:
    """各文字種の割合 (0-1)。空白除外。"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"kanji": 0, "hiragana": 0, "katakana": 0, "alnum": 0, "other": 0}
    n = len(chars)

    def ratio(pred):
        return sum(1 for c in chars if pred(c)) / n

    return {
        "kanji": ratio(lambda c: "\u4e00" <= c <= "\u9fff"),
        "hiragana": ratio(lambda c: "\u3040" <= c <= "\u309f"),
        "katakana": ratio(lambda c: "\u30a0" <= c <= "\u30ff"),
        "alnum": ratio(lambda c: bool(re.match(r"[A-Za-z0-9]", c))),
        "other": ratio(
            lambda c: not (
                ("\u4e00" <= c <= "\u9fff")
                or ("\u3040" <= c <= "\u309f")
                or ("\u30a0" <= c <= "\u30ff")
                or re.match(r"[A-Za-z0-9]", c)
            )
        ),
    }


def classify_character(text: str) -> Tuple[np.ndarray, Dict[str, float]]:
    rates = char_type_ratios(text)
    # 各ラベルの親和度
    kanji, hira, kata, alnum = (
        rates["kanji"],
        rates["hiragana"],
        rates["katakana"],
        rates["alnum"],
    )
    # エントロピー的なバランス: 上位種の偏りが小さいほどバランス
    vec = np.array([kanji, hira, kata, alnum], dtype=np.float64)
    if vec.sum() > 0:
        p = vec / vec.sum()
        ent = -np.sum(p * np.log(p + 1e-12))
        balance = float(ent / np.log(4))  # 0-1
    else:
        balance = 0.0

    raw = np.array(
        [
            kanji,
            hira,
            kata,
            alnum,
            balance,
        ],
        dtype=np.float64,
    )
    raw = np.clip(raw, 1e-6, None)
    probs = raw / raw.sum()
    return probs, rates


def representative_score(probs: np.ndarray, rates: Dict[str, float]) -> float:
    """合算用: バランス型と読みやすさ（ひらがな）をやや重視。"""
    # LABELS index: 0 kanji, 1 hira, 2 kata, 3 alnum, 4 balance
    return float(
        min(
            1.0,
            probs[4] * 0.5 + probs[1] * 0.3 + (1.0 - abs(rates["kanji"] - 0.3)) * 0.2,
        )
    )


def make_bar_chart(probs: np.ndarray) -> "Image.Image":
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [x[0] for x in LABELS]
    colors = ["#5d4037", "#00838f", "#6a1b9a", "#37474f", "#2e7d32"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(labels))
    ax.barh(y, probs, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("C: Character Type")
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


def run(text: Optional[str]) -> Dict[str, Any]:
    ok, err = validate_text(text)
    if not ok:
        logger.warning("C NG: %s", err)
        return {
            "ok": False,
            "error": err,
            "score": 0.0,
            "probs": None,
            "status": f"❌ C NG: {err}",
            "detail_md": f"**エラー**  \n{err}",
            "chart_img": None,
            "rates": {},
        }

    probs, rates = classify_character(text)
    score = representative_score(probs, rates)
    top = LABELS[int(np.argmax(probs))][0]
    status = f"C score: {score:.1%}  |  Top: {top}"

    lines = [
        f"**{status}**",
        "",
        "### 文字種割合",
        "| Type | Ratio |",
        "|------|-------|",
        f"| 漢字 | {rates['kanji']:.1%} |",
        f"| ひらがな | {rates['hiragana']:.1%} |",
        f"| カタカナ | {rates['katakana']:.1%} |",
        f"| 英数 | {rates['alnum']:.1%} |",
        f"| その他 | {rates['other']:.1%} |",
        "",
        "### ラベル分布",
        "| Label | Score |",
        "|-------|-------|",
    ]
    for (name, _, _), p in zip(LABELS, probs):
        lines.append(f"| {name} | {p:.1%} |")
    detail_md = "\n".join(lines)

    return {
        "ok": True,
        "error": "",
        "score": score,
        "probs": probs,
        "status": status,
        "detail_md": detail_md,
        "chart_img": make_bar_chart(probs),
        "rates": rates,
    }
