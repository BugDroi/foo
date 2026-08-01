# =============================================
# SNC WebUI — Gradio WEB UI for SNC (Sexual / Necessary / Character) scanner
# 処理本体は branch_s / branch_n / branch_c / snc_merge
# =============================================

import sys
from pathlib import Path

# gradio-ui.py から見た script フォルダの絶対パスを取得して追加
SCRIPT_DIR = Path(__file__).resolve().parent / "script"
sys.path.append(str(SCRIPT_DIR))

# script 内のモジュールをインポート
import branch_c
import branch_n
import branch_s
import snc_merge

import gradio as gr

CUSTOM_CSS = """
.gradio-container { max-width: 1120px !important; }
.status-box textarea { font-size: 1.05em !important; font-weight: 600; }
"""

with gr.Blocks(
    title="SNC — Character / NSFW Scanner",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
        # SNC Scanner
        **S** Sexual（画像） · **N** Necessary（テキスト） · **C** Character（テキスト）  
        各ブランチは独立実行でき、合算タブでスコア統合と改善レポートを生成します。
        """
    )

    with gr.Tabs():
        # ---- S ----
        with gr.Tab("S — Sexual（画像）"):
            with gr.Row():
                with gr.Column():
                    s_image = gr.Image(
                        type="pil",
                        label="入力画像",
                        sources=["upload", "clipboard"],
                        height=320,
                    )
                    s_target = gr.Dropdown(
                        choices=list(branch_s.HEATMAP_TARGETS.keys()),
                        value=list(branch_s.HEATMAP_TARGETS.keys())[0],
                        label="ヒートマップ対象",
                    )
                    with gr.Accordion("ヒートマップ調整", open=False):
                        s_alpha = gr.Slider(0.1, 0.8, value=0.4, step=0.05, label="alpha")
                        s_gamma = gr.Slider(0.2, 1.5, value=0.5, step=0.05, label="gamma")
                    s_btn = gr.Button("S を実行", variant="primary")
                with gr.Column():
                    s_status = gr.Textbox(label="ステータス", elem_classes=["status-box"])
                    s_chart = gr.Image(type="pil", label="分類スコア")
                    s_heat = gr.Image(type="pil", label="Attention ヒートマップ")
            s_detail = gr.Markdown()
            s_btn.click(
                fn=snc_merge.gradio_s,
                inputs=[s_image, s_target, s_alpha, s_gamma],
                outputs=[s_chart, s_heat, s_status, s_detail],
            )

        # ---- N ----
        with gr.Tab("N — Necessary（テキスト）"):
            with gr.Row():
                with gr.Column():
                    n_text = gr.Textbox(
                        lines=10,
                        label="評価するテキスト",
                        placeholder="必要情報・説明文を入力",
                    )
                    n_pr = gr.Textbox(
                        lines=6,
                        label="PR 例（1行1例・空ならデフォルト）",
                        placeholder="\n".join(branch_n.DEFAULT_PR_EXAMPLES),
                    )
                    n_btn = gr.Button("N を実行", variant="primary")
                with gr.Column():
                    n_status = gr.Textbox(label="ステータス", elem_classes=["status-box"])
                    n_chart = gr.Image(type="pil", label="Necessary スコア")
            n_detail = gr.Markdown()
            n_btn.click(
                fn=snc_merge.gradio_n,
                inputs=[n_text, n_pr],
                outputs=[n_chart, n_status, n_detail],
            )

        # ---- C ----
        with gr.Tab("C — Character（テキスト）"):
            with gr.Row():
                with gr.Column():
                    c_text = gr.Textbox(
                        lines=12,
                        label="評価するテキスト",
                        placeholder="文字種バランスを見たい本文",
                    )
                    c_btn = gr.Button("C を実行", variant="primary")
                with gr.Column():
                    c_status = gr.Textbox(label="ステータス", elem_classes=["status-box"])
                    c_chart = gr.Image(type="pil", label="Character スコア")
            c_detail = gr.Markdown()
            c_btn.click(
                fn=snc_merge.gradio_c,
                inputs=[c_text],
                outputs=[c_chart, c_status, c_detail],
            )

        # ---- Merge ----
        with gr.Tab("合算 · 改善レポート"):
            gr.Markdown("S / N / C をまとめて実行し、重み付き合算と改善案を出します。")
            with gr.Row():
                with gr.Column():
                    a_image = gr.Image(type="pil", label="S: 画像", height=240)
                    a_target = gr.Dropdown(
                        choices=list(branch_s.HEATMAP_TARGETS.keys()),
                        value=list(branch_s.HEATMAP_TARGETS.keys())[0],
                        label="ヒートマップ対象",
                    )
                    with gr.Accordion("ヒートマップ調整", open=False):
                        a_alpha = gr.Slider(0.1, 0.8, value=0.4, step=0.05, label="alpha")
                        a_gamma = gr.Slider(0.2, 1.5, value=0.5, step=0.05, label="gamma")
                with gr.Column():
                    a_text_n = gr.Textbox(lines=6, label="N: テキスト")
                    a_pr = gr.Textbox(lines=4, label="N: PR 例（任意）")
                    a_text_c = gr.Textbox(
                        lines=6,
                        label="C: テキスト（N とは別に入力）",
                    )
            with gr.Accordion("合算重み", open=False):
                with gr.Row():
                    w_s = gr.Slider(0, 1, value=0.40, step=0.05, label="重み S")
                    w_n = gr.Slider(0, 1, value=0.35, step=0.05, label="重み N")
                    w_c = gr.Slider(0, 1, value=0.25, step=0.05, label="重み C")
            a_btn = gr.Button("S・N・C を一括実行", variant="primary")
            a_summary = gr.Textbox(label="サマリー", lines=5, elem_classes=["status-box"])
            with gr.Row():
                a_s_chart = gr.Image(type="pil", label="S スコア")
                a_s_heat = gr.Image(type="pil", label="S ヒートマップ")
            with gr.Row():
                a_n_chart = gr.Image(type="pil", label="N スコア")
                a_c_chart = gr.Image(type="pil", label="C スコア")
                a_merge_chart = gr.Image(type="pil", label="合算グラフ")
            a_report = gr.Markdown(label="改善レポート")
            a_btn.click(
                fn=snc_merge.gradio_all,
                inputs=[
                    a_image,
                    a_target,
                    a_alpha,
                    a_gamma,
                    a_text_n,
                    a_pr,
                    a_text_c,
                    w_s,
                    w_n,
                    w_c,
                ],
                outputs=[
                    a_s_chart,
                    a_s_heat,
                    a_n_chart,
                    a_c_chart,
                    a_merge_chart,
                    a_summary,
                    a_report,
                ],
            )

    gr.Markdown(
        """
        ---
        **凡例**  S: 画像の性的・露出表現（CLIP）　·　N: 必要情報と PR 例の類似度　·　C: 文字種バランス  
        NG のブランチは合算時に重み 0 として正規化されます。
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
