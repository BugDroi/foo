# =============================================
# Buzzwole - Anime Image Content Classifier WebUI
# CLIP-based classification + attention heatmap
# =============================================

import logging
import warnings
import os
import random
from typing import Optional, Tuple, List

import torch
import clip
import cv2
import numpy as np
from PIL import Image
import gradio as gr

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("buzzwole")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ---------- constants ----------
MODEL_NAME = "ViT-B/32"
HEATMAP_ALPHA = 0.4
HEATMAP_GAMMA = 0.5

LABEL_MAP = [
    ("an anime girl fully clothed in casual wear", "Safe: Casual", "blue"),
    ("an anime girl in school uniform or work clothes", "Safe: Uniform", "blue"),
    ("an anime girl in sports or athletic wear", "Safe: Sports", "blue"),
    ("an anime girl in a swimsuit or bikini", "Warning: Swimsuit", "orange"),
    ("an anime girl in a short skirt with visible underwear", "Warning: Upskirt", "orange"),
    ("an anime girl with large breasts heavily emphasized", "Sensitive: Breast", "red"),
    ("an anime girl in a revealing outfit showing much skin", "Sensitive: Exposure", "red"),
    ("an anime girl in a seductive or suggestive pose", "Sensitive: Pose", "red"),
    ("an anime girl with a sexually suggestive facial expression", "Sensitive: Expression", "red"),
    ("an explicit or nude anime illustration", "Danger: Explicit", "darkred"),
    ("an anime girl in lingerie or underwear only", "Danger: Lingerie", "darkred"),
]

HEATMAP_TARGETS = {
    "露出・服装": "a character in a suggestive or revealing outfit",
    "ポーズ": "a character in a seductive or provocative pose",
}

THRESHOLDS = {
    "safe": (0, 3, 0.6),
    "warning": (3, 4, 0.3),
    "sensitive": (4, 7, 0.3),
}

english_labels = [m[0] for m in LABEL_MAP]
short_labels = [m[1] for m in LABEL_MAP]
bar_colors = [m[2] for m in LABEL_MAP]

# ---------- model load (once) ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

model, preprocess = clip.load(MODEL_NAME, device=device)
model.eval()

with torch.no_grad():
    text_tokens = clip.tokenize(english_labels).to(device)
    cls_text_features = model.encode_text(text_tokens)
    cls_text_features = cls_text_features / cls_text_features.norm(dim=-1, keepdim=True)

    heatmap_text_features = {}
    for label, target_text in HEATMAP_TARGETS.items():
        tokens = clip.tokenize([target_text]).to(device)
        feat = model.encode_text(tokens)
        heatmap_text_features[label] = feat / feat.norm(dim=-1, keepdim=True)

logger.info("Model and text features ready")


# ---------- core functions ----------
def classify_image(img_input: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        image_features = model.encode_image(img_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ cls_text_features.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    return probs


def compute_gradients(
    img_input: torch.Tensor, target_feat: torch.Tensor
) -> Tuple[np.ndarray, float]:
    img_input = img_input.detach().requires_grad_(True)
    image_features = model.encode_image(img_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    score = (image_features @ target_feat.T).squeeze()
    score.backward()
    grads = img_input.grad.detach().cpu()[0].abs().mean(dim=0).numpy()
    return grads, float(score.item())


def normalize_gradients(grads: np.ndarray, gamma: float = HEATMAP_GAMMA) -> np.ndarray:
    eps = 1e-8
    g_min, g_max = grads.min(), grads.max()
    if g_max == g_min:
        return np.zeros_like(grads, dtype=np.float32)
    normalized = (grads - g_min) / (g_max - g_min + eps)
    return np.power(normalized, gamma)


def blend_heatmap(
    img_rgb: Image.Image, grads_norm: np.ndarray, alpha: float = HEATMAP_ALPHA
) -> np.ndarray:
    grads_blur = cv2.GaussianBlur(np.float32(grads_norm), (13, 13), 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * grads_blur), cv2.COLORMAP_JET)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    blended = cv2.addWeighted(img_cv, 1 - alpha, heatmap_resized, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def generate_review(probs: np.ndarray) -> str:
    safe_score = probs[0:3].sum()
    warning_score = probs[3]
    sensitive_score = probs[4:7].sum()

    if sensitive_score > THRESHOLDS["sensitive"][2]:
        status = "Status: SENSITIVE"
    elif warning_score > THRESHOLDS["warning"][2]:
        status = "Status: WARNING"
    elif safe_score > THRESHOLDS["safe"][2]:
        status = "Status: SAFE"
    else:
        status = "Status: BORDERLINE"

    top_idx = int(np.argmax(probs))
    return f"{status}  |  Top: {short_labels[top_idx]} ({probs[top_idx]:.1%})"


def make_bar_chart(probs: np.ndarray) -> Image.Image:
    """Horizontal bar chart of classification scores as PIL Image."""
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
    ax.set_title("Classification Score")
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


# ---------- Gradio analysis pipeline ----------
def analyze(
    image: Optional[Image.Image],
    heatmap_target: str,
    alpha: float,
    gamma: float,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str, str]:
    if image is None:
        return None, None, "画像をアップロードしてください。", ""

    img_rgb = image.convert("RGB")
    img_input = preprocess(img_rgb).unsqueeze(0).to(device)

    # 1. Classification
    probs = classify_image(img_input)
    review_text = generate_review(probs)

    # Detailed score table
    lines = [f"**{review_text}**", "", "| Label | Score |", "|-------|-------|"]
    for label, p in zip(short_labels, probs):
        lines.append(f"| {label} | {p:.1%} |")
    detail_md = "\n".join(lines)

    # 2. Heatmap
    target_feat = heatmap_text_features.get(
        heatmap_target, list(heatmap_text_features.values())[0]
    )
    grads, attn_score = compute_gradients(img_input, target_feat)
    grads_norm = normalize_gradients(grads, gamma=gamma)
    heat_arr = blend_heatmap(img_rgb, grads_norm, alpha=alpha)
    heat_img = Image.fromarray(heat_arr)

    # 3. Bar chart
    chart_img = make_bar_chart(probs)

    status_line = f"{review_text}  ·  Attention score ({heatmap_target}): {attn_score:.3f}"
    return chart_img, heat_img, status_line, detail_md


# ---------- Gradio UI ----------
CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; }
.status-box textarea { font-size: 1.05em !important; font-weight: 600; }
"""

with gr.Blocks(
    title="Buzzwole - Anime Content Classifier",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
        # Blaceaphon — Character Image Content Classifier
        CLIP (ViT-B/32) によるアニメ画像の安全度分類と、注目領域のヒートマップ可視化によるフィードバックを提供します。
        画像をアップロードして「分析する」を押してください。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="入力画像",
                sources=["upload", "clipboard"],
                height=360,
            )
            heatmap_target = gr.Dropdown(
                choices=list(HEATMAP_TARGETS.keys()),
                value=list(HEATMAP_TARGETS.keys())[0],
                label="ヒートマップ対象",
            )
            with gr.Accordion("ヒートマップ調整", open=False):
                alpha_slider = gr.Slider(
                    0.1, 0.8, value=HEATMAP_ALPHA, step=0.05, label="重ね合わせ強度 (alpha)"
                )
                gamma_slider = gr.Slider(
                    0.2, 1.5, value=HEATMAP_GAMMA, step=0.05, label="ガンマ補正 (gamma)"
                )
            analyze_btn = gr.Button("分析する", variant="primary", size="lg")

        with gr.Column(scale=1):
            status_out = gr.Textbox(label="ステータス", interactive=False, elem_classes=["status-box"])
            chart_out = gr.Image(type="pil", label="分類スコア", height=320)
            heat_out = gr.Image(type="pil", label="Attention ヒートマップ", height=320)

    detail_out = gr.Markdown(label="詳細スコア")

    analyze_btn.click(
        fn=analyze,
        inputs=[input_image, heatmap_target, alpha_slider, gamma_slider],
        outputs=[chart_out, heat_out, status_out, detail_out],
    )
    # Also run on image change if desired (optional; keep explicit button for control)
    # input_image.change(...)

    gr.Markdown(
        """
        ---
        **ラベル凡例**  
        Safe (Casual / Uniform / Sports)　·　Warning (Swimsuit / Upskirt)  
        Sensitive (Breast / Exposure / Pose / Expression)　·　Danger (Explicit / Lingerie)

        **閾値**  
        SAFE: Safe系合計 > 60%　·　WARNING: Swimsuit > 30%　·　SENSITIVE: Sensitive系合計 > 30%
        """
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
