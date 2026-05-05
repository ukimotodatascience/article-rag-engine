import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embedding import E5Embedder, BM25SparseEmbedder
import config

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = config.COLLECTION_NAME

class HybridRetriever:
    def __init__(self):
        if not QDRANT_URL or not QDRANT_API_KEY:
            print("エラー: QDRANT_URL または QDRANT_API_KEY が設定されていません。")
            exit(1)
            
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        self.dense_embedder = E5Embedder()
        self.sparse_embedder = BM25SparseEmbedder()
        
    def search(self, query_text: str, limit: int = config.RETRIEVER_LIMIT):
        """
        ハイブリッド検索 (Dense + Sparse) と RRF による結合を実行する
        """
        print(f"\n検索クエリ: '{query_text}'")
        
        # 1. 密ベクトル生成 (is_query=True)
        dense_vector = self.dense_embedder.encode(query_text, is_query=True)[0]
        
        # 2. 疎ベクトル生成 (BM25)
        sparse_vector_dict = self.sparse_embedder.encode(query_text)[0]
        sparse_vector = models.SparseVector(
            indices=sparse_vector_dict["indices"],
            values=sparse_vector_dict["values"]
        )
        
        # 3. Qdrant Query APIを用いたPrefetch + RRF Fusion
        # - コサイン類似度(密ベクトル)の上位30件を取得
        # - BM25(疎ベクトル)の上位20件を取得
        # - Qdrant内部でRRFを計算し、最終的に上位limit件(5件)を返す
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="", # デフォルトの密ベクトル
                    limit=30
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="text-sparse", # BM25の疎ベクトル
                    limit=20
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        
        return results.points

def main():
    retriever = HybridRetriever()
    
    # 動作確認用テストクエリ
    test_queries = [
        "Notionからデータを取得する方法について",
    ]
    
    for query in test_queries:
        results = retriever.search(query, limit=config.RETRIEVER_LIMIT)
        print(f"\n=== 検索結果 (上位 {len(results)} 件) ===")
        for i, point in enumerate(results):
            payload = point.payload
            title = payload.get('title', '無題')
            source = payload.get('source', '不明')
            text_snippet = payload.get('text', '').replace('\n', ' ')[:100]
            print(f"[{i+1}] スコア: {point.score:.4f} | 記事: {title} ({source})")
            print(f"    {text_snippet}...")

if __name__ == "__main__":
    main()
