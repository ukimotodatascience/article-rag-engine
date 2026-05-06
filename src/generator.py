import requests
import json
import config

class LocalLLMGenerator:
    def __init__(self, model_name: str = config.OLLAMA_MODEL_NAME):
        """
        Ollama API を使用して回答を生成するクラス。
        """
        self.model_name = model_name
        self.base_url = config.OLLAMA_BASE_URL
        self.last_done_reason = None
        
        # Ollama の接続確認 (初期化時)
        self.is_connected()

    def is_connected(self) -> bool:
        """Ollama サーバーに接続可能か確認します"""
        try:
            # 接続確認用のタイムアウトは短めに設定 (5秒)
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
            return False
        except:
            return False

    def get_available_models(self) -> list:
        """Ollama で利用可能なモデル一覧を取得します"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if not models:
                    return [self.model_name]
                return [m["name"] for m in models]
            return [self.model_name]
        except:
            # 取得失敗時はデフォルトモデルのみを返す
            return [self.model_name]

    def _prepare_messages(self, query: str, retrieved_points: list, chat_history: list = None) -> list:
        """検索結果とチャット履歴からプロンプト用メッセージを作成します"""
        context_texts = []
        for i, point in enumerate(retrieved_points):
            payload = point.payload
            title = payload.get('title', '無題')
            source = payload.get('source', '不明')
            text = payload.get('text', '')
            context_texts.append(f"[資料 {i+1}] タイトル: {title} (ファイル: {source})\n{text}")
            
        context_str = "\n\n".join(context_texts)
        
        system_prompt = (
            "あなたは優秀なアシスタントです。提供された参考資料を元に、ユーザーの質問に正確に答えてください。\n"
            "参考資料に記載されていないことは「資料には記載がありません」と答え、推測で回答しないでください。"
        )
        
        # メッセージリストの初期化
        messages = [{"role": "system", "content": system_prompt}]
        
        # チャット履歴がある場合は追加（直近10件程度に制限することも検討可能）
        if chat_history:
            for msg in chat_history:
                # ロールが 'user' または 'assistant' であることを確認
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 最新の質問とコンテキストを追加
        user_prompt = f"【参考資料】\n{context_str}\n\n【質問】\n{query}"
        messages.append({"role": "user", "content": user_prompt})

        return messages

    def generate(self, query: str, retrieved_points: list, chat_history: list = None, model_name: str = None, max_new_tokens: int = None) -> str:
        """
        標準出力にストリーミングしながら最終的な回答を返します。
        """
        target_model = model_name or self.model_name
        messages = self._prepare_messages(query, retrieved_points, chat_history)
        full_response = ""
        self.last_done_reason = None
        
        print(f"\n[{target_model}] 回答を生成中...\n")
        print("-" * 50)
        
        try:
            payload = {
                "model": target_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": max_new_tokens or config.LLM_MAX_NEW_TOKENS
                }
            }
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]
                        print(token, end="", flush=True)
                        full_response += token
                    if chunk.get("done"):
                        self.last_done_reason = chunk.get("done_reason")
                        break
            
            print("\n" + "-" * 50)
            return full_response
            
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            return f"エラーが発生しました: {e}"

    def generate_stream(self, query: str, retrieved_points: list, chat_history: list = None, model_name: str = None, max_new_tokens: int = None):
        """
        Streamlit 用にトークンを逐次 yield します。
        """
        target_model = model_name or self.model_name
        messages = self._prepare_messages(query, retrieved_points, chat_history)
        self.last_done_reason = None
        
        try:
            payload = {
                "model": target_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": max_new_tokens or config.LLM_MAX_NEW_TOKENS
                }
            }
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=30)
            
            if response.status_code != 200:
                error_msg = f"Ollamaエラー ({response.status_code}): {response.text}"
                yield error_msg
                return

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "error" in chunk:
                        yield f"Ollamaエラー: {chunk['error']}"
                        return
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                    if chunk.get("done"):
                        self.last_done_reason = chunk.get("done_reason")
                        break
                        
        except requests.exceptions.Timeout:
            yield "エラー: Ollama サーバーへの接続がタイムアウトしました。サーバーが起動しているか、ポート番号が正しいか確認してください。"
        except Exception as e:
            yield f"予期せぬエラーが発生しました: {e}"
