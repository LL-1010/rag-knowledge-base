import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi

# ===== 页面配置 =====
st.set_page_config(
    page_title="企业知识库问答",
    page_icon="📚",
    layout="wide"
)

st.title("📚 企业知识库智能问答系统")
st.caption("基于通义千问 + Chroma 构建，支持溯源引用、智能拒答和多轮对话")

# ===== 加载环境变量 =====
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DISTANCE_THRESHOLD = 7500.0

if not DASHSCOPE_API_KEY:
    st.error("❌ 请在 .env 文件中设置 DASHSCOPE_API_KEY")
    st.stop()


# ===== 加载向量库和模型（缓存） =====
@st.cache_resource
def load_vector_db():
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


@st.cache_resource
def load_llm():
    return Tongyi(
        model="qwen-plus",
        dashscope_api_key=DASHSCOPE_API_KEY
    )


try:
    vector_db = load_vector_db()
    llm = load_llm()
except Exception as e:
    st.error(f"❌ 加载失败: {e}")
    st.stop()

# ===== 侧边栏：系统状态与对话控制 =====
with st.sidebar:
    st.header("📊 系统状态")
    try:
        all_data = vector_db.get()
        st.metric("文档片段数", len(all_data['ids']))
    except:
        st.metric("文档片段数", "未知")
    st.metric("拒答阈值", DISTANCE_THRESHOLD)
    st.divider()

    # 多轮对话控制
    st.subheader("💬 多轮对话")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.caption(f"当前{len(st.session_state.get('messages', []))}条消息")

    st.divider()
    st.caption("💡 输入问题，系统会从知识库中检索并回答")
    st.caption("🔒 支持上下文追问和多轮对话")

# ===== 初始化对话历史 =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== 显示历史消息 =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===== 输入框 =====
if question := st.chat_input("请输入您的问题..."):
    # 将用户问题添加到历史
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # ===== 1. 检索 =====
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索..."):
            docs_with_scores = vector_db.similarity_search_with_score(question, k=3)
            docs = [doc for doc, _ in docs_with_scores]
            distances = [score for _, score in docs_with_scores]
            best_distance = distances[0] if distances else 9999.0

        # ===== 2. 思维链 =====
        with st.expander("🧠 思维链（内部思考过程）", expanded=False):
            st.write(f"**步骤1: 检索**")
            st.write(f"找到 {len(docs)} 个相关片段")
            st.write(f"最佳匹配距离: `{best_distance:.2f}`")
            st.write(f"**步骤2: 判断**")
            st.write(f"阈值: `{DISTANCE_THRESHOLD}`")
            if best_distance <= DISTANCE_THRESHOLD:
                st.success(f"✅ 最佳距离 {best_distance:.2f} <= 阈值 {DISTANCE_THRESHOLD}，放行")
            else:
                st.error(f"❌ 最佳距离 {best_distance:.2f} > 阈值 {DISTANCE_THRESHOLD}，拒答")

        # ===== 3. 判断是否拒答 =====
        if best_distance > DISTANCE_THRESHOLD:
            response = "抱歉，知识库中没有找到与您问题相关的信息。"
            st.warning(response)
        else:
            # ===== 4. 构建上下文（含历史对话） =====
            context = "\n\n".join([doc.page_content for doc in docs])

            # 构建历史对话上下文（最近3轮）
            history_text = ""
            if len(st.session_state.messages) > 1:
                recent = st.session_state.messages[-5:-1]  # 最近几条历史
                if recent:
                    history_text = "【对话历史】\n" + "\n".join([
                        f"用户: {m['content']}" if m['role'] == 'user' else f"助手: {m['content']}"
                        for m in recent
                    ]) + "\n\n"

            # ===== 5. 构造Prompt =====
            prompt = f"""你是一个企业知识库助手。请基于以下【参考文档】和【对话历史】回答问题。

{history_text}
【参考文档】
{context}

【问题】
{question}

【要求】
1. 如果参考文档中没有相关信息，请直接回答"抱歉，知识库中没有找到相关信息"。
2. 不要编造任何文档中没有的内容。
3. 回答要简洁、准确。
4. 如果用户的问题是基于之前对话的追问，请结合上下文回答。

【回答】
"""

            with st.spinner("🤖 正在生成回答..."):
                response = llm.invoke(prompt)

            st.success("✅ 回答：")
            st.write(response)

            # ===== 6. 引用来源 =====
            st.divider()
            st.caption("📚 参考来源：")
            for i, (doc, dist) in enumerate(zip(docs, distances)):
                with st.expander(f"来源 [{i + 1}] 第{doc.metadata.get('page', '?')}页 (距离: {dist:.2f})"):
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

        # 将助手回答添加到历史
        st.session_state.messages.append({"role": "assistant", "content": response})

# ===== 底部 =====
st.divider()
st.caption("🔒 本系统基于通义千问 + Chroma 构建 | 支持多轮对话 | 仅供学习参考")