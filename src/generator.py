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
        
        # Ollama の接続確認
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            if response.status_code == 200:
                print(f"✅ Ollama サーバーに接続しました ({self.base_url})")
            else:
                print(f"⚠️ Ollama サーバーからの応答が異常です: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Ollama サーバー ({self.base_url}) との通信で待機中、または接続できません: {e}")

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

    def _prepare_messages(self, query: str, retrieved_points: list) -> list:
        """検索結果からプロンプト用メッセージを作成します"""
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
        
        user_prompt = f"【参考資料】\n{context_str}\n\n【質問】\n{query}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def generate(self, query: str, retrieved_points: list) -> str:
        """
        標準出力にストリーミングしながら最終的な回答を返します。
        """
        messages = self._prepare_messages(query, retrieved_points)
        full_response = ""
        
        print(f"\n[{self.model_name}] 回答を生成中...\n")
        print("-" * 50)
        
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_NEW_TOKENS
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
                        break
            
            print("\n" + "-" * 50)
            return full_response
            
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            return f"エラーが発生しました: {e}"

    def generate_stream(self, query: str, retrieved_points: list):
        """
        Streamlit 用にトークンを逐次 yield します。
        """
        messages = self._prepare_messages(query, retrieved_points)
        
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_NEW_TOKENS
                }
            }
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                    if chunk.get("done"):
                        break
                        
        except Exception as e:
            yield f"エラーが発生しました: {e}"
