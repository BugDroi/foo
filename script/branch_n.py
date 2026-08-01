# =============================================
# N ブランチ: Necessary（テキスト判定）
# =============================================
"""必要情報テキストを PR 例と照合し、類似度スコアを返す。"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("snc.n")

# 分類ラベル（表示名, 説明, 重み）
LABELS = [
    ("必須要素あり", "目的・条件・制約が明確", 1.0),
    ("背景・文脈あり", "前提や状況の説明がある", 0.8),
    ("具体例あり", "数値・固有名・手順など具体性", 0.9),
    ("曖昧・不足", "情報が薄く判断が難しい", 0.3),
    ("無関係", "課題と関係が薄い", 0.1),
]

# デフォルトの PR（評価基準）例 — UI から差し替え可能
DEFAULT_PR_EXAMPLES = [
    "目的と対象読者を明確にし、必要な前提知識を先に述べる。",
    "制約条件（時間・予算・技術）を箇条書きで示す。",
    "具体的な数値目標と成功基準を含める。",
    "手順は再現可能な粒度で、入力と出力を定義する。",
]


def validate_text(text: Optional[str]) -> Tuple[bool, str]:
    if text is None:
        return False, "テキストが指定されていません"
    s = text.strip()
    if not s:
        return False, "テキストが空です"
    # 文字化けっぽい連続置換文字
    if "\ufffd" in s or s.count("?") > max(10, len(s) // 5):
        return False, "文字化けの可能性があるテキストです"
    if len(s) > 50_000:
        return False, "テキストが長すぎます"
    return True, ""


def _tokenize(text: str) -> set:
    # 簡易トークン（日本語は文字 bigram + 英数字単語）
    text = text.lower()
    words = set(re.findall(r"[a-z0-9_]+", text))
    chars = re.sub(r"\s+", "", text)
    bigrams = {chars[i : i + 2] for i in range(max(0, len(chars) - 1))}
    return words | bigrams


def similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def score_against_pr(text: str, pr_examples: List[str]) -> Dict[str, float]:
    """各 PR 例との類似度と、総合スコア。"""
    if not pr_examples:
        pr_examples = DEFAULT_PR_EXAMPLES
    per = {f"PR{i+1}": similarity(text, ex) for i, ex in enumerate(pr_examples)}
    vals = list(per.values())
    overall = float(np.mean(vals)) if vals else 0.0
    # ヒューリスティック加点: 長さ・箇条書き・数字
    bonus = 0.0
    if len(text.strip()) >= 80:
        bonus += 0.05
    if re.search(r"[-*・]|^\s*\d+[\.\)]", text, re.M):
        bonus += 0.05
    if re.search(r"\d", text):
        bonus += 0.05
    overall = min(1.0, overall + bonus)
    per["overall"] = overall
    return per


def classify_necessary(text: str, pr_examples: List[str]) -> np.ndarray:
    """LABELS に対応する疑似分布を作る。"""
    sim = score_against_pr(text, pr_examples)
    overall = sim["overall"]
    # overall が高い → 必須・背景・具体例、低い → 曖昧・無関係
    raw = np.array(
        [
            overall * 1.0,  # 必須
            overall * 0.7 + 0.1,  # 背景
            overall * 0.8,  # 具体例
            max(0.0, 0.6 - overall),  # 曖昧
            max(0.0, 0.4 - overall * 0.8),  # 無関係
        ],
        dtype=np.float64,
    )
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum()


def make_bar_chart(probs: np.ndarray) -> "Image.Image":
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [x[0] for x in LABELS]
    colors = ["#2e7d32", "#1565c0", "#6a1b9a", "#ef6c00", "#c62828"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(labels))
    ax.barh(y, probs, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("N: Necessary Score")
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


def run(
    text: Optional[str],
    pr_examples_text: Optional[str] = None,
) -> Dict[str, Any]:
    ok, err = validate_text(text)
    if not ok:
        logger.warning("N NG: %s", err)
        return {
            "ok": False,
            "error": err,
            "score": 0.0,
            "probs": None,
            "status": f"❌ N NG: {err}",
            "detail_md": f"**エラー**  \n{err}",
            "chart_img": None,
            "pr_scores": {},
        }

    if pr_examples_text and pr_examples_text.strip():
        pr_examples = [ln.strip() for ln in pr_examples_text.splitlines() if ln.strip()]
    else:
        pr_examples = list(DEFAULT_PR_EXAMPLES)

    pr_scores = score_against_pr(text, pr_examples)
    probs = classify_necessary(text, pr_examples)
    overall = float(pr_scores["overall"])
    top = LABELS[int(np.argmax(probs))][0]
    status = f"N score: {overall:.1%}  |  Top: {top}"

    lines = [
        f"**{status}**",
        "",
        "### PR 例との類似度",
        "| PR | Similarity |",
        "|----|------------|",
    ]
    for k, v in pr_scores.items():
        if k == "overall":
            continue
        lines.append(f"| {k} | {v:.1%} |")
    lines += ["", "### ラベル分布", "| Label | Score |", "|-------|-------|"]
    for (name, _, _), p in zip(LABELS, probs):
        lines.append(f"| {name} | {p:.1%} |")
    detail_md = "\n".join(lines)

    return {
        "ok": True,
        "error": "",
        "score": overall,
        "probs": probs,
        "status": status,
        "detail_md": detail_md,
        "chart_img": make_bar_chart(probs),
        "pr_scores": pr_scores,
    }
