# =============================================
# S ブランチ: Sexual（画像判定）
# 元 app.py / main.py の CLIP 分類 + Attention ヒートマップ
# =============================================
"""
アニメ画像の性的・露出表現を CLIP (ViT-B/32) でスコア化する。

フロー (SNC の S):
  画像判定 → 画像読み込み・検証 → 分類ラベル定義 → スコア計算
  NG 時はエラーを返し、合算側で重み 0 扱いになる。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("snc.s")

# ---------- 定数（元 app.py と同じ） ----------
MODEL_NAME = "ViT-B/32"
HEATMAP_ALPHA = 0.4
HEATMAP_GAMMA = 0.5

# 分類ラベル定義（英語プロンプト, 表示名, 色）
LABEL_MAP = [
    # 日常的な私服・露出の少ない服装
    ("an anime girl fully clothed in casual wear",
     "Safe: Casual", "blue"),
    # 制服や仕事着などの一般的な服装
    ("an anime girl in school uniform or work clothes",
     "Safe: Uniform", "blue"),
    # スポーツウェアや運動着
    ("an anime girl in sports or athletic wear",
     "Safe: Sports", "blue"),
    # 水着やビキニなど肌の露出が比較的多い服装
    ("an anime girl in a swimsuit or bikini",
     "Warning: Swimsuit", "orange"),
    # 短いスカートなど露出が目立つ表現
    ("an anime girl in a short skirt with visible underwear",
     "Warning: Upskirt", "orange"),
    # 胸部が強調された描写
    ("an anime girl with large breasts heavily emphasized",
     "Sensitive: Breast", "red"),
    # 肌の露出が多い服装やデザイン
    ("an anime girl in a revealing outfit showing much skin",
     "Sensitive: Exposure", "red"),
    # 誘導的・挑発的と解釈されうるポーズ
    ("an anime girl in a seductive or suggestive pose",
     "Sensitive: Pose", "red"),
    # 特定の感情表現や表情の強調
    ("an anime girl with a sexually suggestive facial expression",
     "Sensitive: Expression", "red"),
    # 明示的な成人向け表現や裸体表現
    ("an explicit or nude anime illustration",
     "Danger: Explicit", "darkred"),
    # 下着のみが主要な服装として描かれている
    ("an anime girl in lingerie or underwear only",
     "Danger: Lingerie", "darkred"),
]

# ヒートマップ用の注目ターゲット
HEATMAP_TARGETS = {
    "露出・服装": "a character in a suggestive or revealing outfit",
    "ポーズ": "a character in a seductive or provocative pose",
}

# 閾値設定（一箇所で管理）
THRESHOLDS = {
    "safe": (0, 3, 0.6),       # インデックス0-2の合計 > 0.6 → SAFE
    "warning": (3, 4, 0.3),    # インデックス3の値   > 0.3 → WARNING
    "sensitive": (4, 7, 0.3),  # インデックス4-6の合計 > 0.3 → SENSITIVE
}

english_labels = [m[0] for m in LABEL_MAP]
short_labels = [m[1] for m in LABEL_MAP]
bar_colors = [m[2] for m in LABEL_MAP]

# ---------- モデル（遅延ロード・1回だけ） ----------
_model = None
_preprocess = None
_device = None
_cls_text_features = None
_heatmap_text_features = None


def _ensure_model() -> None:
    """CLIP とテキスト特徴量を初回だけ読み込む。"""
    global _model, _preprocess, _device, _cls_text_features, _heatmap_text_features
    if _model is not None:
        return

    import torch
    import clip

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("S branch: loading CLIP (%s) on %s", MODEL_NAME, _device)

    _model, _preprocess = clip.load(MODEL_NAME, device=_device)
    _model.eval()  # 推論モード（重みの勾配は不要）

    with torch.no_grad():
        text_tokens = clip.tokenize(english_labels).to(_device)
        feat = _model.encode_text(text_tokens)
        _cls_text_features = feat / feat.norm(dim=-1, keepdim=True)

        _heatmap_text_features = {}
        for label, target_text in HEATMAP_TARGETS.items():
            tokens = clip.tokenize([target_text]).to(_device)
            f = _model.encode_text(tokens)
            _heatmap_text_features[label] = f / f.norm(dim=-1, keepdim=True)

    logger.info("S branch: model and text features ready")


# ---------- 検証 ----------
def validate_image(image: Optional[Image.Image]) -> Tuple[bool, str]:
    """
    形式・サイズ検証。
    OK → (True, "") / NG → (False, 理由)
    """
    if image is None:
        return False, "画像が指定されていません"
    try:
        img = image.convert("RGB")
    except Exception as e:
        return False, f"画像形式を読み取れません: {e}"
    w, h = img.size
    if w < 32 or h < 32:
        return False, f"画像が小さすぎます ({w}x{h})"
    if w * h > 50_000_000:
        return False, f"画像が大きすぎます ({w}x{h})"
    return True, ""


# ---------- コア（元 app.py の分析関数） ----------
def classify_image(img_input) -> np.ndarray:
    """画像の分類スコアを返す（推論のみ）。"""
    import torch

    _ensure_model()
    with torch.no_grad():
        image_features = _model.encode_image(img_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # コサイン類似度を softmax でスコア化
        logits = (image_features @ _cls_text_features.T) * _model.logit_scale.exp()
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    return probs


def compute_gradients(img_input, target_feat) -> Tuple[np.ndarray, float]:
    """
    入力画像テンソルに対する勾配を計算する。
    モデルの重みは変化させず、img_input への勾配のみ取得する。
    """
    _ensure_model()
    img_input = img_input.detach().requires_grad_(True)

    image_features = _model.encode_image(img_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    score = (image_features @ target_feat.T).squeeze()
    score.backward()

    # RGB チャンネルを平均で集約し、小さなチャネルノイズの影響を減らす
    grads = img_input.grad.detach().cpu()[0].abs().mean(dim=0).numpy()
    return grads, float(score.item())


def normalize_gradients(grads: np.ndarray, gamma: float = HEATMAP_GAMMA) -> np.ndarray:
    """勾配をガンマ補正付きで 0-1 正規化する。"""
    eps = 1e-8
    g_min, g_max = grads.min(), grads.max()
    if g_max == g_min:
        return np.zeros_like(grads, dtype=np.float32)
    normalized = (grads - g_min) / (g_max - g_min + eps)
    return np.power(normalized, gamma)


def blend_heatmap(
    img_rgb: Image.Image,
    grads_norm: np.ndarray,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """正規化済み勾配から元画像にヒートマップを合成する。"""
    import cv2

    # 斑点抑制のため平滑化
    grads_blur = cv2.GaussianBlur(np.float32(grads_norm), (13, 13), 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * grads_blur), cv2.COLORMAP_JET)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    blended = cv2.addWeighted(img_cv, 1 - alpha, heatmap_resized, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def generate_review(probs: np.ndarray) -> str:
    """スコアからステータス文を生成する。"""
    safe_score = probs[0:3].sum()       # Casual + Uniform + Sports
    warning_score = probs[3]            # Swimsuit
    sensitive_score = probs[4:7].sum()  # Breast + Exposure + Pose

    if sensitive_score > THRESHOLDS["sensitive"][2]:
        status = "⚠️ Status: SENSITIVE"
    elif warning_score > THRESHOLDS["warning"][2]:
        status = "🟡 Status: WARNING"
    elif safe_score > THRESHOLDS["safe"][2]:
        status = "✅ Status: SAFE"
    else:
        status = "🔵 Status: BORDERLINE"

    top_idx = int(np.argmax(probs))
    return f"{status}  |  Top: {short_labels[top_idx]} ({probs[top_idx]:.1%})"


def make_bar_chart(
    probs: np.ndarray,
    title: str = "S: Classification Score",
) -> Image.Image:
    """横棒グラフを PIL Image で返す。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    y_pos = np.arange(len(short_labels))
    ax.barh(y_pos, probs, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(title)
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


# ---------- エントリポイント ----------
def run(
    image: Optional[Image.Image],
    heatmap_target: str = "露出・服装",
    alpha: float = HEATMAP_ALPHA,
    gamma: float = HEATMAP_GAMMA,
) -> Dict[str, Any]:
    """
    S ブランチを1回実行する。

    Returns
    -------
    dict
        ok, error, score, probs, status, detail_md,
        chart_img, heat_img, attn_score
    """
    ok, err = validate_image(image)
    if not ok:
        logger.warning("S NG: %s", err)
        return {
            "ok": False,
            "error": err,
            "score": 0.0,
            "probs": None,
            "status": f"❌ S NG: {err}",
            "detail_md": f"**エラー**  \n{err}",
            "chart_img": None,
            "heat_img": None,
            "attn_score": 0.0,
        }

    _ensure_model()

    img_rgb = image.convert("RGB")
    img_input = _preprocess(img_rgb).unsqueeze(0).to(_device)

    # 1. 分類スコア
    probs = classify_image(img_input)
    review = generate_review(probs)

    # 合算用代表スコア: Sensitive 以降（Breast〜Lingerie）の合計
    risk = float(min(1.0, probs[4:].sum()))

    lines = [f"**{review}**", "", "| Label | Score |", "|-------|-------|"]
    for label, p in zip(short_labels, probs):
        lines.append(f"| {label} | {p:.1%} |")
    detail_md = "\n".join(lines)

    # 2. ヒートマップ
    target_feat = _heatmap_text_features.get(
        heatmap_target,
        list(_heatmap_text_features.values())[0],
    )
    grads, attn_score = compute_gradients(img_input, target_feat)
    grads_norm = normalize_gradients(grads, gamma=gamma)
    heat_img = Image.fromarray(blend_heatmap(img_rgb, grads_norm, alpha=alpha))

    # 3. 棒グラフ
    chart_img = make_bar_chart(probs)

    return {
        "ok": True,
        "error": "",
        "score": risk,
        "probs": probs,
        "status": f"{review}  ·  Attention ({heatmap_target}): {attn_score:.3f}",
        "detail_md": detail_md,
        "chart_img": chart_img,
        "heat_img": heat_img,
        "attn_score": attn_score,
    }
