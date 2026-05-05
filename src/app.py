import streamlit as st
import time
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from retriever import HybridRetriever
from generator import LocalLLMGenerator
import config

# ページ設定
st.set_page_config(
    page_title="InsightArc | AI Article RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# カスタムCSSでプレミアムなデザインを実現
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

    /* タイポグラフィ */
    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* チャットメッセージの調整 */
    .stChatMessage {{
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        transition: transform 0.2s ease;
    }}

    .stChatMessage:hover {{
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.05) !important;
    }}

    /* サイドバー */
    [data-testid="stSidebar"] {{
        background-color: #1e293b;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* ボタン・入力欄 */
    .stButton > button {{
        border-radius: 8px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        opacity: 0.9;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }}

    .stChatInputContainer {{
        background-color: var(--surface) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}

    /* 引用ソースのスタイリング */
    .source-container {{
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}

    .source-card {{
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 12px;
        font-size: 0.9rem;
    }}

    .source-header {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }}

    .source-title {{
        font-weight: 600;
        color: #e2e8f0;
    }}

    .source-tag {{
        font-size: 0.7rem;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }}

    /* アニメーション */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .fade-in {{
        animation: fadeIn 0.5s ease forwards;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_components():
    """RAGコンポーネントの初期化とキャッシュ"""
    with st.spinner("✨ 知識エンジンを起動中..."):
        retriever = HybridRetriever()
        generator = LocalLLMGenerator()
    return retriever, generator

def main():
    # ヘッダーセクション（画像を削除しコンパクトに）
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown("### 🧬")
    with col2:
        st.title("InsightArc")
    st.markdown("*Transforming Notion knowledge into intelligence.*")
    st.markdown('</div>', unsafe_allow_html=True)

    # コンポーネントの取得
    try:
        retriever, generator = get_components()
    except Exception as e:
        st.error(f"初期化中にエラーが発生しました: {e}")
        return

    # サイドバー設定
    with st.sidebar:
        st.markdown("### 🛠️ System Control")
        
        # 接続ステータスをバッジ風に表示
        is_ollama_online = generator.is_connected()
        if is_ollama_online:
            st.markdown('<span style="color: #4ade80; font-weight: bold;">● OLLAMA ONLINE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #f87171; font-weight: bold;">○ OLLAMA OFFLINE</span>', unsafe_allow_html=True)
            st.info(f"URL: {config.OLLAMA_BASE_URL}")
        
        st.divider()
        
        with st.expander("🧠 Model Configuration", expanded=True):
            available_models = generator.get_available_models()
            default_index = available_models.index(config.OLLAMA_MODEL_NAME) if config.OLLAMA_MODEL_NAME in available_models else 0
            
            selected_model = st.selectbox(
                "Active Model",
                available_models,
                index=default_index,
                help="Ollamaにインストールされているモデルを選択します"
            )
            
            if st.button("Refresh Models", use_container_width=True):
                st.rerun()
        
        with st.expander("🔍 Search Parameters"):
            limit = st.slider("Context Chunks", 1, 10, config.RETRIEVER_LIMIT)
        
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_retrieved = []
            st.rerun()
        
        st.markdown("---")
        st.caption("v1.2.0 | Powered by E5 & Ollama")

    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []

    # メインチャットエリア
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # ユーザー入力
    if prompt := st.chat_input("Ask anything from your library..."):
        # ユーザーメッセージ
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 回答生成
        with chat_container:
            with st.chat_message("assistant"):
                # 検索フェーズ
                with st.status("🔍 Digging through articles...", expanded=False) as status:
                    results = retriever.search(prompt, limit=limit)
                    st.session_state.last_retrieved = results
                    if results:
                        status.update(label=f"🎯 Found {len(results)} relevant articles", state="complete")
                    else:
                        status.update(label="❓ No specific context found", state="error")
                
                # 生成フェーズ
                response_placeholder = st.empty()
                full_response = ""
                
                if results:
                    for token in generator.generate_stream(prompt, results, model_name=selected_model):
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                else:
                    full_response = "I couldn't find any specific information in your articles to answer this accurately."
                    response_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 参考資料の表示（フローティング的なカード形式）
    if st.session_state.last_retrieved:
        st.divider()
        st.subheader("📚 Supporting References")
        
        # カラムで並べる
        cols = st.columns(min(3, len(st.session_state.last_retrieved)))
        for i, point in enumerate(st.session_state.last_retrieved):
            with cols[i % len(cols)]:
                payload = point.payload
                title = payload.get('title', 'Untitled')
                source = payload.get('source', 'Unknown')
                text = payload.get('text', '')[:200] + "..." # 抜粋
                
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-title">#{i+1} {title[:20]}</span>
                        <span class="source-tag">Source</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 8px;">{source[:30]}</div>
                    <div style="color: #cbd5e1; font-style: italic;">"{text}"</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
