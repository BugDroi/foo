import platform
import subprocess
import time
from pathlib import Path
from openai import OpenAI

# api-test.py のあるディレクトリを基準にする
ROOT_DIR = Path(__file__).resolve().parent

# llm フォルダの場所を特定（同階層、またはひとつ上の親階層を探す）
if (ROOT_DIR / "llm").exists():
    LLM_DIR = ROOT_DIR / "llm"
elif (ROOT_DIR.parent / "llm").exists():
    LLM_DIR = ROOT_DIR.parent / "llm"
else:
    LLM_DIR = ROOT_DIR

BASE_URL = "http://127.0.0.1:8080/v1"


def start_llm_server() -> subprocess.Popen:
    """OSに応じて run.bat または run.sh をバックグラウンドで起動する"""
    is_windows = platform.system() == "Windows"

    if is_windows:
        run_script = LLM_DIR / "run.bat"
        cmd = [str(run_script)]
    else:
        run_script = LLM_DIR / "run.sh"
        cmd = ["bash", str(run_script)]

    if not run_script.exists():
        raise FileNotFoundError(f"起動スクリプトが見つかりません: {run_script}")

    print(f"🚀 LLM サーバーを起動しています... ({run_script})")

    # stdout/stderr を DEVNULL に指定してバックグラウンド実行
    process = subprocess.Popen(
        cmd,
        cwd=str(LLM_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process


def wait_for_server(client: OpenAI, max_retries: int = 30) -> bool:
    """サーバーの準備が整うまでヘルスメッセージを送って待機する"""
    print("⏳ サーバーの応答を待機中...")
    for _ in range(max_retries):
        try:
            client.models.list()
            print("✅ サーバーが正常に起動しました！\n")
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    # 1. LLM サーバーの起動
    try:
        server_process = start_llm_server()
    except Exception as e:
        print(f"❌ サーバー起動エラー: {e}")
        return

    client = OpenAI(base_url=BASE_URL, api_key="sk-no-key-required")

    try:
        # 2. 起動完了を待機
        if not wait_for_server(client):
            print("❌ サーバーの起動タイムアウト（起動に失敗した可能性があります）")
            return

        print("==========================================")
        print("💬 Chat Start! (『exit』または『quit』で終了)")
        print("==========================================")

        # 会話履歴を保持するリスト
        messages = [
            {
                "role": "system",
                "content": "あなたは親切で優秀なAIアシスタントです。日本語で短く分かりやすく答えてください。",
            }
        ]

        # 3. 対話ループ
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n対話を終了します。")
                break

            # 終了判定
            if user_input.lower() in ["exit", "quit", "終了"]:
                print("対話を終了します。")
                break

            if not user_input:
                continue

            # ユーザーの発言を履歴に追加
            messages.append({"role": "user", "content": user_input})

            print("AI: ", end="", flush=True)

            try:
                # LLMへリクエスト送信 (過去の履歴も一緒に送る)
                stream = client.chat.completions.create(
                    model="sarashina2.2-3b-instruct",
                    messages=messages,
                    temperature=0.7,
                    stream=True,
                )

                assistant_response = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        print(content, end="", flush=True)
                        assistant_response += content

                print()  # 改行

                # AIの返答を履歴に追加
                messages.append(
                    {"role": "assistant", "content": assistant_response}
                )

            except Exception as e:
                print(f"\n[エラーが発生しました]: {e}")

    finally:
        # 4. 終了時にサーバープロセスを停止
        print("\n🧹 LLM サーバーを終了しています...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("👋 終了しました。")


if __name__ == "__main__":
    main()