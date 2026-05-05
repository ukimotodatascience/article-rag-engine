# InsightArc | AI Article RAG Engine 🧬

InsightArcは、Notionに蓄積された記事やドキュメントを知識ベースとして活用し、ローカルLLM（Ollama）を用いて対話形式で情報を引き出すことができるRAG（Retrieval-Augmented Generation）システムです。

## 🌟 主な特徴

- **Notion 連携**: Notionデータベースから記事を自動取得し、Markdown形式で処理します。
- **ハイブリッド検索**: `multilingual-e5-base` モデルを使用した高精度な意味検索（Dense Retrieval）を搭載。
- **ローカルLLM統合**: Ollamaを活用し、プライバシーを保ちながら高速な回答生成を実現。ストリーミング出力にも対応。
- **プレミアム UI**: Streamlitを採用した、モダンで直感的なグラスモーフィズムデザイン。
- **高度なシークレット管理**: Infisicalによるセキュアな環境変数管理。

## 🏗️ システムアーキテクチャ

1.  **Fetch**: `notion_fetch_database.py` がNotion API経由でドキュメントを取得。
2.  **Process**: `chunking.py` がテキストを適切なサイズに分割。
3.  **Embed**: `embedding.py` が `multilingual-e5-base` を用いてベクトル化。
4.  **Index**: `indexer.py` が Qdrant ベクトルデータベースへデータを格納。
5.  **Retrieve**: `retriever.py` がユーザーの質問に関連する情報をQdrantから検索。
6.  **Generate**: `generator.py` が Ollama を通じて回答を生成。
7.  **Serve**: `app.py` (Streamlit) がユーザーインターフェースを提供。

## 🚀 セットアップ

### 1. 前提条件

- Python 3.10以上
- [Ollama](https://ollama.ai/) (Llama3, Gemmaなどのモデルがインストールされていること)
- [Qdrant](https://qdrant.tech/) (Cloudまたはローカルインスタンス)
- Notion API Key と Database ID
- [Infisical CLI](https://infisical.com/) (推奨) または `.env` ファイル

### 2. インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd article-rag-engine

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 3. 環境設定

`.env.example` を参考に `.env` ファイルを作成するか、Infisicalに以下の変数を設定してください。

```env
NOTION_TOKEN=your_notion_token
NOTION_DATABASE_ID=your_database_id
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3
```

## 📖 使い方

### インデクシング (データの同期)

Notionの最新データを取得し、ベクトルデータベースを更新します。

```bash
infisical run -- python src/indexer.py
```

### アプリケーションの起動

#### Web UI (推奨)
Streamlit UIを起動してチャットを開始します。

```bash
infisical run -- streamlit run src/app.py
```

#### CLI モード
コマンドラインから直接対話することも可能です。

```bash
infisical run -- python src/main.py
```

## ⚙️ 設定

`src/config.py` を編集することで、以下の設定を調整できます。

- `CHUNK_SIZE`: テキスト分割の文字数
- `RETRIEVER_LIMIT`: LLMに渡す参考資料の件数
- `LLM_TEMPERATURE`: 回答の創造性調整
- `DENSE_EMBEDDING_MODEL`: 使用するエンベディングモデル

## 🛠️ 技術スタック

- **Frontend**: Streamlit
- **Vector DB**: Qdrant
- **LLM Engine**: Ollama
- **Embedding**: Sentence-Transformers (multilingual-e5-base)
- **Secret Management**: Infisical
- **Language**: Python

---
Developed by fuben | InsightArc v1.2.0
