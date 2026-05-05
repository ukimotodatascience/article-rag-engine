import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import config

class E5Embedder:
    """
    intfloat/multilingual-e5-base を用いてエンベディング（ベクトル化）を行うクラス。
    
    注意: e5モデルでは、検索クエリには 'query: ' を、
    インデックスする文書（パッセージ）には 'passage: ' をテキストの先頭に付与する必要があります。
    """
    def __init__(self, model_name: str = config.DENSE_EMBEDDING_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """トークンレベルの出力を平均化して文全体のエンベディングを取得します"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, texts: Union[str, List[str]], is_query: bool = False) -> List[List[float]]:
        """
        テキストのエンベディングを生成します。
        
        Args:
            texts: エンベディングするテキスト、またはテキストのリスト。
            is_query: 検索クエリの場合はTrue、インデックスする文書の場合はFalse。
        
        Returns:
            エンベディングされたベクトルのリスト（各ベクトルは768次元）。
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # E5モデルの要件に従い、プレフィックスを付与
        prefix = "query: " if is_query else "passage: "
        processed_texts = [f"{prefix}{text}" for text in texts]

        # トークナイズ
        batch_dict = self.tokenizer(
            processed_texts, 
            max_length=config.EMBEDDING_MAX_LENGTH, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )

        # モデルの推論
        outputs = self.model(**batch_dict)
        
        # プーリング
        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # L2正規化 (検索時には正規化済みのベクトルの内積をとることでコサイン類似度が計算できる)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.tolist()

if __name__ == "__main__":
    # 簡単な動作確認用のコード
    embedder = E5Embedder()
    
    # 登録する文書データ (is_query=False)
    documents = [
        "Pythonはシンプルで読みやすいプログラミング言語です。",
        "RAG（Retrieval-Augmented Generation）は、大規模言語モデルに外部の知識ベースを組み込む技術です。",
        "今日はいい天気ですね。"
    ]
    print("文書をエンベディングしています...")
    doc_embeddings = embedder.encode(documents, is_query=False)
    print(f"文書エンベディングの次元数: {len(doc_embeddings[0])}次元\n")
    
    # 検索クエリ (is_query=True)
    query = "RAGとは何ですか？"
    print(f"検索クエリ: '{query}' をエンベディングしています...")
    query_embedding = embedder.encode(query, is_query=True)
    
    # コサイン類似度による検索のシミュレーション
    import torch
    query_tensor = torch.tensor(query_embedding)
    doc_tensor = torch.tensor(doc_embeddings)
    
    # 正規化済みのため、内積を取るだけでコサイン類似度となる
    scores = (query_tensor @ doc_tensor.T) * 100
    
    print("\n--- 検索結果（類似度スコア） ---")
    for i, score in enumerate(scores[0]):
        print(f"スコア: {score:.2f} | 文書: {documents[i]}")

class BM25SparseEmbedder:
    """
    BM25を用いた疎ベクトル（Sparse Vector）を生成するクラス。
    日本語テキストをJanomeで分かち書きし、FastEmbedのBM25モデルを使用してベクトル化します。
    """
    def __init__(self):
        try:
            from janome.tokenizer import Tokenizer
            from fastembed import SparseTextEmbedding
        except ImportError:
            raise ImportError(
                "BM25SparseEmbedder を使用するには janome と fastembed パッケージが必要です。\n"
                "環境で `pip install fastembed janome` を実行してください。"
            )
        self.tokenizer = Tokenizer()
        self.model = SparseTextEmbedding("Qdrant/bm25")

    def _tokenize(self, text: str) -> str:
        """テキストを形態素解析してスペース区切りの文字列に変換します"""
        # 不要な改行などを除去して分かち書き
        clean_text = text.replace("\n", " ").replace("　", " ")
        tokens = self.tokenizer.tokenize(clean_text, wakati=True)
        return " ".join(tokens)

    def encode(self, texts: Union[str, List[str]]) -> List[dict]:
        """
        テキストの疎ベクトルを生成します。
        
        Args:
            texts: エンベディングするテキスト、またはテキストのリスト。
        
        Returns:
            疎ベクトルのリスト（各要素は {'indices': [int], 'values': [float]} の辞書）
        """
        if isinstance(texts, str):
            texts = [texts]
            
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # fastembedのモデルで疎ベクトルを生成
        embeddings = list(self.model.embed(tokenized_texts))
        
        result = []
        for emb in embeddings:
            result.append({
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist()
            })
            
        return result
