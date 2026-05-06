import streamlit as st
import time
import os
import base64
import html
from pathlib import Path
from dotenv import load_dotenv
from retriever import HybridRetriever
from generator import LocalLLMGenerator
from session_manager import SessionManager
import config

# ページ設定
st.set_page_config(
    page_title="InsightArc | AI Article RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@300;500;700&display=swap');

    :root {{
        --primary: #6366f1;
        --secondary: #a855f7;
        --background: #0f172a;
        --surface: rgba(30, 41, 59, 0.7);
        --text: #f8fafc;
        --text-muted: #94a3b8;
    }}

    .stApp {{
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}

    /* グラスモーフィズム */
    .glass-card {{
        background: var(--surface);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }}

    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .stChatMessage {{
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: #1e293b;
    }}

    .stButton > button {{
        border-radius: 8px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        font-weight: 600;
    }}

    .source-card {{
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 12px;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_components():
    """RAGコンポーネントの初期化"""
    with st.spinner("✨ 知識エンジンを起動中..."):
        retriever = HybridRetriever()
        generator = LocalLLMGenerator()
        session_manager = SessionManager()
    return retriever, generator, session_manager

def main():
    # コンポーネントの取得
    try:
        retriever, generator, session_manager = get_components()
    except Exception as e:
        st.error(f"初期化中にエラーが発生しました: {e}")
        return

    # セッション状態の初期化
    if "current_session_id" not in st.session_state:
        sessions = session_manager.list_sessions()
        if sessions:
            st.session_state.current_session_id = sessions[0]["id"]
        else:
            st.session_state.current_session_id = session_manager.create_session()
    
    # 現在のセッションデータの読み込み
    session_data = session_manager.load_session(st.session_state.current_session_id)
    if not session_data:
        # 万が一読み込めない場合は新規作成
        st.session_state.current_session_id = session_manager.create_session()
        session_data = session_manager.load_session(st.session_state.current_session_id)

    st.session_state.messages = session_data.get("messages", [])
    st.session_state.last_retrieved = session_data.get("last_retrieved", [])

    # サイドバー
    with st.sidebar:
        st.title("🧬 InsightArc")
        
        # 接続ステータス
        is_ollama_online = generator.is_connected()
        status_color = "#4ade80" if is_ollama_online else "#f87171"
        status_text = "OLLAMA ONLINE" if is_ollama_online else "OLLAMA OFFLINE"
        st.markdown(f'<span style="color: {status_color}; font-weight: bold; font-size: 0.8rem;">● {status_text}</span>', unsafe_allow_html=True)
        
        st.divider()
        
        with st.expander("⚙️ Settings"):
            available_models = generator.get_available_models()
            default_index = available_models.index(config.OLLAMA_MODEL_NAME) if config.OLLAMA_MODEL_NAME in available_models else 0
            selected_model = st.selectbox("Active Model", available_models, index=default_index)
            limit = st.slider("Context Chunks", 1, 10, config.RETRIEVER_LIMIT)

        st.divider()

        # セッション管理
        if st.button("＋ New Chat", use_container_width=True):
            st.session_state.current_session_id = session_manager.create_session()
            st.rerun()
            
        st.markdown("### 💬 Recent Chats")
        sessions = session_manager.list_sessions()
        for s in sessions:
            col1, col2 = st.columns([0.8, 0.2])
            is_active = s["id"] == st.session_state.current_session_id
            
            with col1:
                if st.button(f"{s['name'][:20]}...", key=f"sel_{s['id']}", use_container_width=True, help=s['name']):
                    st.session_state.current_session_id = s["id"]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{s['id']}", help="Delete session"):
                    session_manager.delete_session(s["id"])
                    if st.session_state.current_session_id == s["id"]:
                        st.session_state.pop("current_session_id")
                    st.rerun()

    # メインエリア
    st.title(session_data.get("name", "New Chat"))
    
    # チャット履歴表示
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # ユーザー入力
    if prompt := st.chat_input("Ask anything from your library..."):
        # ユーザーメッセージ追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 回答生成
        with chat_container:
            with st.chat_message("assistant"):
                with st.status("🔍 Searching articles...", expanded=False) as status:
                    # 検索
                    results = retriever.search(prompt, limit=limit)
                    
                    # JSONシリアライズ可能な形式に変換して保存
                    st.session_state.last_retrieved = [
                        {
                            "payload": point.payload,
                            "score": float(point.score)
                        }
                        for point in results
                    ]
                    status.update(label=f"🎯 Found {len(results)} relevant chunks", state="complete")
                
                response_placeholder = st.empty()
                full_response = ""
                
                # 生成時にチャット履歴を渡す
                history = st.session_state.messages[:-1]
                
                # results (ScoredPoint) をそのまま generator に渡す
                for token in generator.generate_stream(prompt, results, chat_history=history, model_name=selected_model):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # セッション保存
                session_manager.save_session(
                    st.session_state.current_session_id, 
                    st.session_state.messages,
                    last_retrieved=st.session_state.last_retrieved
                )
                st.rerun()

    # 参考資料の表示
    if st.session_state.last_retrieved:
        st.divider()
        st.subheader("📚 Supporting References")
        
        # 重複を排除（同じ記事の複数チャンクがヒットした場合）
        unique_refs = []
        seen_titles = set()
        for point in st.session_state.last_retrieved:
            payload = point.get("payload", {})
            title = payload.get('title', 'Untitled')
            url = payload.get('url', '')
            if title not in seen_titles:
                unique_refs.append({"title": title, "url": url})
                seen_titles.add(title)

        for i, ref in enumerate(unique_refs):
            safe_title = html.escape(ref['title'])
            if ref['url']:
                st.markdown(f"**{i+1}. {safe_title}** — [{ref['url']}]({ref['url']})")
            else:
                st.markdown(f"**{i+1}. {safe_title}**")

if __name__ == "__main__":
    main()
