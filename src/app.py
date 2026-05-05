import streamlit as st
import time
import os
from dotenv import load_dotenv
from retriever import HybridRetriever
from generator import LocalLLMGenerator
import config

# ページ設定
st.set_page_config(
    page_title="Article RAG Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSSでデザインを調整
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stChatInput {
        border-radius: 20px;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .source-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-title {
        font-weight: bold;
        color: #2980b9;
        margin-bottom: 5px;
    }
    .source-snippet {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_components():
    """RAGコンポーネントの初期化とキャッシュ"""
    with st.spinner("🚀 システムを起動中... (モデルのロードに時間がかかる場合があります)"):
        retriever = HybridRetriever()
        generator = LocalLLMGenerator()
    return retriever, generator

def main():
    st.title("📚 Article RAG Engine")
    st.markdown("Notionに保存された技術記事から回答を生成するAIアシスタントです。")

    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ 設定")
        st.info("このシステムはローカルLLM (Qwen2.5) を使用して回答を生成します。")
        
        st.divider()
        st.subheader("検索パラメータ")
        limit = st.slider("取得チャンク数", 1, 10, config.RETRIEVER_LIMIT)
        
        st.divider()
        if st.button("チャット履歴をクリア"):
            st.session_state.messages = []
            st.rerun()

    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []

    # コンポーネントの取得
    try:
        retriever, generator = get_components()
    except Exception as e:
        st.error(f"初期化中にエラーが発生しました: {e}")
        return

    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力
    if prompt := st.chat_input("質問を入力してください..."):
        # ユーザーメッセージの表示と保存
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AIの回答生成
        with st.chat_message("assistant"):
            # 1. 検索フェーズ
            with st.status("🔍 関連記事を検索中...", expanded=False) as status:
                results = retriever.search(prompt, limit=limit)
                st.session_state.last_retrieved = results
                if results:
                    status.update(label=f"✅ {len(results)}件の資料が見つかりました", state="complete", expanded=False)
                else:
                    status.update(label="❌ 関連資料が見つかりませんでした", state="error", expanded=True)
            
            # 2. 生成フェーズ
            response_placeholder = st.empty()
            full_response = ""
            
            if results:
                # ストリーミング生成
                for token in generator.generate_stream(prompt, results):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            else:
                full_response = "申し訳ありませんが、関連する資料が見つからなかったため回答できません。"
                response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 参考資料の表示（メイン画面の下部またはサイドバー）
    if st.session_state.last_retrieved:
        with st.expander("📖 今回の回答で使用した参考資料"):
            for i, point in enumerate(st.session_state.last_retrieved):
                payload = point.payload
                title = payload.get('title', '無題')
                source = payload.get('source', '不明')
                text = payload.get('text', '')
                
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">[{i+1}] {title}</div>
                    <div class="source-snippet">ソース: {source}</div>
                    <hr style="margin: 10px 0;">
                    <div>{text}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
