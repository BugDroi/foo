# =============================================
# S・N・C 合算と改善レポート
# =============================================
"""3ブランチのスコアを重み付き合算し、グラフと改善案を生成する。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# デフォルト重み（フロー上の「分類ラベルの重みを反映」）
DEFAULT_WEIGHTS = {"S": 0.40, "N": 0.35, "C": 0.25}


def merge_scores(
    s: Dict[str, Any],
    n: Dict[str, Any],
    c: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    # NG ブランチは重みを 0 にして正規化
    parts = {"S": s, "N": n, "C": c}
    active = {k: w[k] for k in parts if parts[k].get("ok")}
    if not active:
        return {
            "ok": False,
            "total": 0.0,
            "scores": {"S": 0.0, "N": 0.0, "C": 0.0},
            "weights_used": w,
            "status": "❌ 全ブランチが NG のため合算できません",
        }
    z = sum(active.values())
    norm = {k: (active[k] / z) for k in active}
    scores = {
        "S": float(s["score"]) if s.get("ok") else 0.0,
        "N": float(n["score"]) if n.get("ok") else 0.0,
        "C": float(c["score"]) if c.get("ok") else 0.0,
    }
    total = sum(scores[k] * norm.get(k, 0.0) for k in scores)
    return {
        "ok": True,
        "total": float(total),
        "scores": scores,
        "weights_used": norm,
        "status": f"合算スコア: {total:.1%}  (S={scores['S']:.1%}, N={scores['N']:.1%}, C={scores['C']:.1%})",
    }


def make_merge_chart(scores: Dict[str, float], total: float) -> "Image.Image":
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["S (Sexual)", "N (Necessary)", "C (Character)", "Total"]
    vals = [scores.get("S", 0), scores.get("N", 0), scores.get("C", 0), total]
    colors = ["#616161", "#2e7d32", "#5e35b1", "#f9a825"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("S · N · C Merge")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


def build_improvement_report(
    s: Dict[str, Any],
    n: Dict[str, Any],
    c: Dict[str, Any],
    merged: Dict[str, Any],
) -> str:
    lines: List[str] = ["## 総合改善レポート", ""]

    # 異常値
    lines.append("### 1. スコア異常値の検知・警告")
    warnings = []
    if merged.get("ok") and merged["total"] < 0.2:
        warnings.append("- 合算スコアが極端に低いです。入力品質かラベル定義を見直してください。")
    if s.get("ok") and s["score"] > 0.7:
        warnings.append("- **S**: センシティブ寄りスコアが高いです。画像の露出・表現を確認してください。")
    if n.get("ok") and n["score"] < 0.25:
        warnings.append("- **N**: PR 例との類似度が低いです。必要情報の不足が疑われます。")
    if c.get("ok") and c.get("rates"):
        r = c["rates"]
        if r.get("alnum", 0) > 0.5:
            warnings.append("- **C**: 英数字比率が高いです。日本語本文としては偏りがあります。")
        if r.get("kanji", 0) > 0.55:
            warnings.append("- **C**: 漢字比率が高いです。読みやすさの観点からひらがなを増やす余地があります。")
    if not warnings:
        warnings.append("- 特に大きな異常は検出されませんでした。")
    lines.extend(warnings)
    lines.append("")

    # ブランチ別課題
    lines.append("### 2. ブランチ別の課題")
    for name, res in (("S", s), ("N", n), ("C", c)):
        if not res.get("ok"):
            lines.append(f"- **{name}**: NG — {res.get('error', '不明')}")
        else:
            lines.append(f"- **{name}**: OK — score={res['score']:.1%} / {res.get('status', '')}")
    lines.append("")

    # ラベル・重み
    lines.append("### 3. ラベル・重み見直し提案")
    wu = merged.get("weights_used", DEFAULT_WEIGHTS)
    lines.append(f"- 今回使用した正規化重み: S={wu.get('S', 0):.2f}, N={wu.get('N', 0):.2f}, C={wu.get('C', 0):.2f}")
    if s.get("ok") and s["score"] > 0.6 and n.get("ok") and n["score"] < 0.3:
        lines.append("- S が高く N が低い場合、画像リスクに対して説明文が追いついていない可能性があります。")
    lines.append("- 用途に応じて `DEFAULT_WEIGHTS`（snc_merge.py）を調整してください。")
    lines.append("")

    # データ品質
    lines.append("### 4. データ品質フィードバック")
    lines.append("- **S**: 境界事例（水着・軽度露出）の画像を追加すると閾値調整が安定します。")
    lines.append("- **N**: PR 例をプロジェクトの実文に差し替えると類似度が実用的になります。")
    lines.append("- **C**: ジャンル別の理想的な文字種比率を目標値として持つと評価が明確になります。")
    lines.append("")

    lines.append("### 5. 合算結果")
    lines.append(merged.get("status", ""))
    return "\n".join(lines)


# ---------- Gradio 向けアダプタ（UI は出力タプルだけ受け取る） ----------
def _missing_branch(name: str) -> dict:
    msg = f"{name} 未配置（{name.lower()}.py がありません）"
    return {
        "ok": False,
        "error": msg,
        "score": 0.0,
        "status": f"❌ {msg}",
        "detail_md": f"**エラー**  \n{msg}",
        "chart_img": None,
        "heat_img": None,
        "attn_score": 0.0,
        "probs": None,
        "rates": {},
        "pr_scores": {},
    }


def gradio_s(image, heatmap_target, alpha, gamma):
    import branch_s

    r = branch_s.run(image, heatmap_target=heatmap_target, alpha=alpha, gamma=gamma)
    return r["chart_img"], r["heat_img"], r["status"], r["detail_md"]


def gradio_n(text, pr_examples):
    try:
        import branch_n
    except ImportError:
        r = _missing_branch("branch_n")
        return r["chart_img"], r["status"], r["detail_md"]
    r = branch_n.run(text, pr_examples_text=pr_examples)
    return r["chart_img"], r["status"], r["detail_md"]


def gradio_c(text):
    try:
        import branch_c
    except ImportError:
        r = _missing_branch("branch_c")
        return r["chart_img"], r["status"], r["detail_md"]
    r = branch_c.run(text)
    return r["chart_img"], r["status"], r["detail_md"]


def gradio_all(
    image,
    heatmap_target,
    alpha,
    gamma,
    text_n,
    pr_examples,
    text_c,
    w_s,
    w_n,
    w_c,
):
    import branch_s

    s = branch_s.run(image, heatmap_target=heatmap_target, alpha=alpha, gamma=gamma)

    try:
        import branch_n
        n = branch_n.run(text_n, pr_examples_text=pr_examples)
    except ImportError:
        n = _missing_branch("branch_n")

    try:
        import branch_c
        c = branch_c.run(text_c)
    except ImportError:
        c = _missing_branch("branch_c")

    weights = {"S": float(w_s), "N": float(w_n), "C": float(w_c)}
    merged = merge_scores(s, n, c, weights=weights)
    chart = make_merge_chart(merged["scores"], merged["total"]) if merged.get("ok") else None
    report = build_improvement_report(s, n, c, merged)
    summary = (
        f"S: {s.get('status', '')}\n"
        f"N: {n.get('status', '')}\n"
        f"C: {c.get('status', '')}\n"
        f"---\n"
        f"{merged.get('status', '')}"
    )
    return (
        s.get("chart_img"),
        s.get("heat_img"),
        n.get("chart_img"),
        c.get("chart_img"),
        chart,
        summary,
        report,
    )
