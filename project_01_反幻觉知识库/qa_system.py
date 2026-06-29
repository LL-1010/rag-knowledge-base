import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from dotenv import load_dotenv

# 加载API Key
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 请先在 .env 文件中设置 DASHSCOPE_API_KEY")
DISTANCE_THRESHOLD = 7500.0
# ==================== 加载向量库 ====================
print("🧠 正在加载向量数据库...")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY
)
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ==================== 初始化大模型 ====================
llm = Tongyi(
    model="qwen-plus",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# ==================== 问答循环（带拒答机制）====================
print("\n🤖 知识库问答系统已启动！输入问题开始查询，输入 exit 退出\n")

while True:
    question = input("\n💬 请输入问题: ")
    if question.lower() in ["exit", "quit", "退出"]:
        print("👋 再见！")
        break

    # ========== 1. 检索相关文档 ==========
    print(f"\n🧠 [1. 检索] 正在从数据库中查找与问题相关的片段...")
    docs_with_scores = vector_db.similarity_search_with_score(question, k=3)
    docs = [doc for doc, _ in docs_with_scores]
    distances = [score for _, score in docs_with_scores]
    best_distance = distances[0] if distances else 9999.0
    print(f"   ✅ 找到 {len(docs)} 个相关片段")
    print(f"   📊 最佳匹配距离: {best_distance:.2f}")

    # ========== 2. 相关性判断 ==========
    print(f"\n🔍 [2. 判断] 阈值: {DISTANCE_THRESHOLD}")
    if best_distance > DISTANCE_THRESHOLD:
        print(f"   ❌ 最佳距离 {best_distance:.2f} > 阈值 {DISTANCE_THRESHOLD}，拒答")
        print("\n🤖 回答: 抱歉，知识库中没有找到与您问题相关的信息。")
        continue
    else:
        print(f"   ✅ 最佳距离 {best_distance:.2f} <= 阈值 {DISTANCE_THRESHOLD}，放行")

    # ========== 3. 拼接上下文 ==========
    print(f"\n📝 [3. 生成] 基于 {len(docs)} 个片段生成回答...")
    context = "\n\n".join([doc.page_content for doc in docs])

    # ========== 4. 构造Prompt ==========
    prompt = f"""你是一个企业知识库助手。请基于以下【参考文档】回答问题。

【参考文档】
{context}

【问题】
{question}

【要求】
1. 如果参考文档中没有相关信息，请直接回答"抱歉，知识库中没有找到相关信息"。
2. 不要编造任何文档中没有的内容。
3. 回答要简洁、准确。

【回答】
"""

    # ========== 5. 调用大模型生成回答 ==========
    response = llm.invoke(prompt)
    print(f"   ✅ 生成完成")
    print(f"\n🤖 回答: {response}")

    # ========== 6. 打印引用来源和原文 ==========
    print("\n📚 参考来源：")
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        print(f"  [{i + 1}] 第{doc.metadata.get('page', '?')}页 (距离: {dist:.2f})")
        print(f"      原文: {doc.page_content[:150]}...")