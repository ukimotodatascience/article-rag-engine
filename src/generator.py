import os
import torch
from pathlib import Path

# モデルの保存先をプロジェクトルート直下の models フォルダに設定
# transformersをインポートする前に環境変数を設定する必要があります
project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / "models"
os.makedirs(models_dir, exist_ok=True)
os.environ["HF_HOME"] = str(models_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import config

class LocalLLMGenerator:
    def __init__(self, model_name: str = config.LLM_MODEL_NAME):
        """
        Hugging Face transformers を使用してローカルで回答を生成するクラス
        指定したモデルはプロジェクト直下の models フォルダにダウンロードされます。
        """
        self.model_name = model_name
        
        print(f"\n[{self.model_name}] モデルを初期化しています (初回はダウンロードに数GBの通信と時間がかかります)...\n")
        
        # デバイスの自動判別 (GPUがあれば使用、なければCPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # low_cpu_mem_usage=True を指定するとロード時のピークメモリを抑えられます
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        
        # ストリーマーの設定（回答を逐次表示するため）
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
    def generate(self, query: str, retrieved_points: list) -> str:
        """
        検索結果のコンテキストを元に、LLMに回答を生成させます。
        ストリーミング形式で標準出力に表示しながら、最終的な文字列を返します。
        """
        # 1. 検索結果からコンテキストテキストを組み立てる
        context_texts = []
        for i, point in enumerate(retrieved_points):
            payload = point.payload
            title = payload.get('title', '無題')
            source = payload.get('source', '不明')
            text = payload.get('text', '')
            context_texts.append(f"[資料 {i+1}] タイトル: {title} (ファイル: {source})\n{text}")
            
        context_str = "\n\n".join(context_texts)
        
        # 2. プロンプトの作成
        system_prompt = (
            "あなたは優秀なアシスタントです。提供された参考資料を元に、ユーザーの質問に正確に答えてください。\n"
            "参考資料に記載されていないことは「資料には記載がありません」と答え、推測で回答しないでください。"
        )
        
        user_prompt = f"【参考資料】\n{context_str}\n\n【質問】\n{query}"

        # ChatMLフォーマット等、モデルが期待するプロンプト形式に整形
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # tokenizer.apply_chat_template を使用してフォーマット
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        print(f"\n[{self.model_name}] 回答を生成中...\n")
        print("-" * 50)
        
        # 3. 推論の実行
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                top_p=0.9,
                repetition_penalty=1.1,
                streamer=self.streamer,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # デコード (streamerがすでに出力しているが、最終文字列も取得して返す)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print("\n" + "-" * 50)
            return full_response
            
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            return ""
