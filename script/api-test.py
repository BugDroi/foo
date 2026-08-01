import httpx
from openai import OpenAI


def llm_process(text: str, timeout: float = 90.0) -> str:
    """ローカルLLMにリクエストを送信する基本関数"""
    try:
        client = OpenAI(
            base_url="http://127.0.0.1:8080/v1",
            api_key="local",
            timeout=timeout,
        )
        r = client.chat.completions.create(
            model="local",
            messages=[{"role": "user", "content": text}],
            temperature=0.3,
            max_tokens=512,  # 要約結果が途中で切れないよう少し大きめに設定
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"（LLM 利用不可: {e}）"


def summarize_text(content: str, max_length: int = 100) -> str:
    """入力されたテキストを指定された長さ程度に要約する関数"""
    prompt = f"""以下のテキストを読んだ上で、重要なポイントを抑えて{max_length}字程度で要約してください。
余計な解説は含めず、要約文のみを出力してください。

---
【対象テキスト】
{content}
"""
    return llm_process(prompt)


# --- 実行例 ---
if __name__ == "__main__":
    sample_text = """
    人工知能（AI）技術の急速な発展に伴い、大規模言語モデル（LLM）の利活用が多様な分野で進んでいます。
    以前はクラウド上のAPIを利用するのが主流でしたが、近年ではオープンソースの軽量モデルが登場し、
    個人用PCやオンプレミス環境のローカルサーバー上で直接モデルを動かす手法が注目を集めています。
    ローカル環境でLLMを運用することで、データの外部送信を防ぎプライバシーを保護できるほか、
    通信コストの削減やオフラインでの動作といった多くのメリットが得られます。
    """

    print("=== 元のテキスト ===")
    print(sample_text.strip())

    print("\n=== 要約結果 (約100字指定) ===")
    summary = summarize_text(sample_text, max_length=100)
    print(summary)