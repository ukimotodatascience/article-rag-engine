import os
import uuid
import json
import hashlib
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SparseVectorParams, SparseIndexParams, SparseVector, Modifier
from chunking import load_markdown_files, process_documents
from embedding import E5Embedder, BM25SparseEmbedder
import config

# .envファイルから環境変数を読み込む（Infisical実行時は環境変数が直接渡されます）
load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = config.COLLECTION_NAME

def get_file_hash(content: str) -> str:
    """コンテンツのMD5ハッシュを計算する"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def load_state(state_file: str) -> dict:
    """ローカルの状態ファイル（ハッシュ履歴）を読み込む"""
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state_file: str, state: dict):
    """ローカルの状態ファイルを保存する"""
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)

def get_qdrant_client() -> QdrantClient:
    """Qdrant Cloudのクライアントを初期化する"""
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("エラー: QDRANT_URL または QDRANT_API_KEY が設定されていません。")
        print("Infisical または .env ファイルで設定してください。")
        exit(1)
        
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60.0, # タイムアウトを長めに設定（デフォルトは短いため、大容量データのアップロードでエラーになりやすい）
    )

def setup_collection(client: QdrantClient, collection_name: str, vector_size: int = 768):
    """コレクションが存在しない場合は作成する"""
    collections = client.get_collections().collections
    exists = any(col.name == collection_name for col in collections)
    
    if not exists:
        print(f"コレクション '{collection_name}' を作成します... (次元数: {vector_size}, Sparse: text-sparse)")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    ),
                    modifier=Modifier.IDF
                )
            }
        )
    else:
        print(f"コレクション '{collection_name}' は既に存在します。")

def main():
    data_dir = "data"
    state_file = "index_state.json"
    
    # 実行ディレクトリ調整
    if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", data_dir)):
        data_dir = os.path.join("..", data_dir)
        state_file = os.path.join("..", "index_state.json")
        
    print("マークダウンファイルを読み込んでいます...")
    docs = load_markdown_files(data_dir)
    
    state = load_state(state_file)
    new_state = {}
    is_first_run = len(state) == 0
    
    docs_to_process = []
    sources_to_delete = []
    
    # 現在のファイルのハッシュを計算・比較
    for doc in docs:
        # パス依存をなくすため、ファイル名のみをキーにする
        source_key = os.path.basename(doc["source"])
        content_hash = get_file_hash(doc["content"])
        new_state[source_key] = content_hash
        
        # Qdrant上で一貫して識別できるように source もファイル名に上書き
        doc["source"] = source_key
        
        if source_key not in state or state[source_key] != content_hash:
            docs_to_process.append(doc)
            # もし更新（既に存在してハッシュが変わった）なら、古いデータをQdrantから削除対象にする
            if source_key in state:
                sources_to_delete.append(source_key)
        else:
            print(f"⏭️ スキップ (変更なし): {source_key}")
            
    # 削除されたファイルを特定 (stateにはあるが、現在のファイルにはない)
    for old_source_key in state:
        if old_source_key not in new_state:
            print(f"🗑️ 削除対象 (ファイルなし): {old_source_key}")
            sources_to_delete.append(old_source_key)
    
    # 初期化
    client = get_qdrant_client()
    
    # 初回実行時：既存のコレクションが存在する場合は、ゴミデータ排除のために一旦削除して再作成
    if is_first_run:
        print("\n⚠️ 初回の差分更新実行のため、既存のコレクションを再作成してデータをクリーンアップします。")
        collections = client.get_collections().collections
        if any(col.name == COLLECTION_NAME for col in collections):
            client.delete_collection(collection_name=COLLECTION_NAME)
        setup_collection(client, COLLECTION_NAME, vector_size=config.DENSE_VECTOR_SIZE)
    else:
        setup_collection(client, COLLECTION_NAME, vector_size=config.DENSE_VECTOR_SIZE)
        
        # 古いチャンクの削除処理
        for src_to_delete in sources_to_delete:
            print(f"🧹 古いデータをQdrantから削除中: {src_to_delete}")
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=src_to_delete))]
                )
            )

    if not docs_to_process and not sources_to_delete:
        print("\n✅ すべてのドキュメントは最新です。更新は必要ありません。")
        return
        
    if not docs_to_process:
        print("\n✅ 追加・更新されるドキュメントはありません。不要なデータの削除のみ完了しました。")
        save_state(state_file, new_state)
        return

    # 1. チャンクデータの準備
    print(f"\n{len(docs_to_process)} 件のドキュメントをチャンキングします...")
    chunks = process_documents(docs_to_process, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    print(f"合計 {len(chunks)} 個のチャンクをベクトル化します。")
    
    embedder = E5Embedder() # model: intfloat/multilingual-e5-base, vector_size: 768
    sparse_embedder = BM25SparseEmbedder() # BM25 (Janome + FastEmbed)
    
    # 3. ベクトル化とQdrantへのアップロード
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    print(f"\nQdrant Cloud へのインデクシングを開始します (全 {total_batches} バッチ)")
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # テキストリストを抽出してベクトル化 (is_query=False -> passage: としてエンコード)
        texts = [chunk["text"] for chunk in batch_chunks]
        embeddings = embedder.encode(texts, is_query=False)
        sparse_embeddings = sparse_embedder.encode(texts)
        
        # Qdrantへ保存するためのPointStructを作成
        points = []
        for j, chunk in enumerate(batch_chunks):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "": embeddings[j],
                        "text-sparse": SparseVector(
                            indices=sparse_embeddings[j]["indices"],
                            values=sparse_embeddings[j]["values"]
                        )
                    },
                    payload={
                        "text": chunk["text"],
                        "source": chunk["metadata"]["source"], # ファイル名のみに上書きされた値
                        "title": chunk["metadata"]["title"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    }
                )
            )
            
        # アップロード
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"バッチ {i//batch_size + 1}/{total_batches} 完了 ({(i + len(batch_chunks))} / {len(chunks)} 件)")

    # 成功したら状態を保存
    save_state(state_file, new_state)
    print("\n✅ すべてのチャンクのインデクシングと状態の保存が完了しました！")

if __name__ == "__main__":
    main()
