import os
import glob
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def load_markdown_files(data_dir: str) -> list[dict]:
    """dataディレクトリ内のマークダウンファイルを読み込む"""
    files = glob.glob(os.path.join(data_dir, "*.md"))
    documents = []
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # ファイル名からタイトルを推測（拡張子を除去）
            filename = os.path.basename(file_path)
            title = os.path.splitext(filename)[0]
            
            # URLを抽出 (<!-- URL: ... -->)
            url_match = re.search(r"<!-- URL:\s*(.*?)\s*-->", content)
            url = url_match.group(1) if url_match else ""

            documents.append({
                "title": title,
                "content": content,
                "source": file_path,
                "url": url
            })
    return documents

def get_text_splitter(chunk_size: int = 400, chunk_overlap: int = 50):
    """
    RecursiveCharacterTextSplitterのインスタンスを生成して返す
    E5モデルのトークン上限（512）を考慮し、デフォルトのchunk_sizeを400（文字）程度に設定。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ]
    )

def process_documents(documents: list[dict], chunk_size: int = 400, chunk_overlap: int = 50) -> list[dict]:
    """ドキュメントのリストをチャンクに分割する"""
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    all_chunks = []
    
    for doc in documents:
        # ドキュメントのテキストを分割
        texts = splitter.split_text(doc["content"])
        
        # チャンクごとにメタデータを付与して保存
        for i, text in enumerate(texts):
            all_chunks.append({
                "chunk_id": f"{doc['title']}_chunk_{i}",
                "text": text,
                "metadata": {
                    "source": doc["source"],
                    "title": doc["title"],
                    "url": doc.get("url", ""),
                    "chunk_index": i
                }
            })
            
    return all_chunks

if __name__ == "__main__":
    # 動作確認用スクリプト
    data_dir = "data"
    
    # スクリプトの実行ディレクトリからの相対パスを調整
    # src/ フォルダ内で実行された場合への対策
    if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", data_dir)):
        data_dir = os.path.join("..", data_dir)

    if not os.path.exists(data_dir):
        print(f"エラー: {data_dir} フォルダが見つかりません。")
        exit(1)
        
    print("ドキュメントを読み込み中...")
    docs = load_markdown_files(data_dir)
    print(f"{len(docs)}件のドキュメントを読み込みました。")
    
    if not docs:
        print("処理するドキュメントがありません。")
        exit(0)
        
    print(f"\nチャンキングを実行中... (chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP})")
    chunks = process_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    print(f"合計 {len(chunks)} 個のチャンクが生成されました。\n")
    
    # 最初の数個のチャンクを表示して確認
    print("--- チャンクのサンプル（先頭2件） ---")
    for i in range(min(2, len(chunks))):
        chunk = chunks[i]
        print(f"[{chunk['chunk_id']}] (長さ: {len(chunk['text'])}文字)")
        print(f"{chunk['text']}\n")
        print("-" * 40)
