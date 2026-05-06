# InsightArc | AI Article RAG Engine 🧬

InsightArcは、Notionに蓄積された記事やドキュメントを知識ベースとして活用し、ローカルLLM（Ollama）を用いて対話形式で情報を引き出すことができるRAG（Retrieval-Augmented Generation）システムです。

## 🌟 主な特徴

-   **Notion 連携**: Notionデータベースから記事を自動取得し、増分更新（Incremental Update）をサポート。
-   **ハイブリッド検索**: `multilingual-e5-base` モデルを使用した高精度な意味検索（Dense Retrieval）を搭載。
-   **ローカルLLM統合**: Ollamaを活用し、プライバシーを確保しながら高速な回答生成を実現。ストリーミング出力に対応。
-   **プレミアム UI**: Streamlitを採用したモダンなデザイン。サイドバーからのモデル切り替えやパラメータ調整が可能。
-   **セッション管理**: チャット履歴の保存と管理。過去の対話を簡単に呼び出せます。
-   **自動同期 (Auto-Sync)**: GitHub Actionsにより、毎日自動的にNotionの最新データを取得・インデックス化。
-   **高度なシークレット管理**: Infisicalによるセキュアな環境変数管理。

## 🏗️ システムアーキテクチャ

1.  **Sync & Index**: `update.py` が `notion_fetch_database.py` と `indexer.py` を順次実行し、データを同期。
2.  **Process**: `chunking.py` がテキストを最適なサイズに分割。
3.  **Embed**: `embedding.py` が `multilingual-e5-base` を用いてベクトル化。
4.  **Retrieve**: `retriever.py` がユーザーの質問に関連する情報を Qdrant から検索。
5.  **Generate**: `generator.py` が Ollama を通じてソースを引用しつつ回答を生成。
6.  **Serve**: `app.py` (Streamlit) が洗練されたユーザーインターフェースを提供。

## 🚀 セットアップ

### 1. 前提条件

-   Python 3.10以上
-   [Ollama](https://ollama.ai/) (Llama3, Gemma などのモデルがインストールされていること)
-   [Qdrant](https://qdrant.tech/) (Cloud またはローカルインスタンス)
-   Notion API Key と Database ID
-   [Infisical CLI](https://infisical.com/) (推奨) または `.env` ファイル

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

### データの同期 (Indexing)

Notionの最新データを取得し、ベクトルデータベースを更新します。

```bash
# 手動での同期実行
infisical run -- python src/update.py
```

※ 実行後、最後に更新された時刻が `last_update.txt` に記録されます。

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

## 🤖 自動更新 (GitHub Actions)

本リポジトリには毎日自動でデータを同期するワークフローが組み込まれています。

-   **スケジュール**: 毎日午前 2 時 (JST)
-   **仕組み**: Notion API から更新分を取得し、Qdrant インデックスを更新。成功時に `last_update.txt` を自動コミットします。
-   **設定**: GitHub リポジトリの `Settings > Secrets and variables > Actions` に上記の環境変数を登録してください。

## ⚙️ 詳細設定

### UI サイドバー
アプリケーション起動後、左側のサイドバーから以下の設定をリアルタイムで変更できます。

-   **Model Selection**: 使用する Ollama モデルの切り替え。
-   **Max Tokens**: 生成される回答の最大トークン数。
-   **Context Chunks**: 検索時に参照するドキュメントの数。

### ソースコード設定
`src/config.py` を編集することで、システムの挙動をより細かく調整できます。

-   `CHUNK_SIZE`: テキスト分割の文字数。
-   `RETRIEVER_LIMIT`: デフォルトの検索件数。
-   `DENSE_EMBEDDING_MODEL`: 使用するエンベディングモデル。

---
Developed by fuben | InsightArc v1.4.0
