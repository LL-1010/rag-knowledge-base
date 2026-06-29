import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 1. 加载环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 在 .env 文件中没有找到 API Key")

print("🧠 正在加载向量数据库...")

# 2. 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 3. 加载数据库
db_path = os.path.join(current_dir, "chroma_db")
vector_db = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings
)

# 4. 【新增】检查数据库里到底有多少条记录
try:
    # 尝试获取所有文档的ID，用来判断数据库是否为空
    all_data = vector_db.get()
    print(f"📊 数据库中共有 {len(all_data['ids'])} 条文档片段")
    if len(all_data['ids']) > 0:
        print(f"   第一条ID: {all_data['ids'][0]}")
        print(f"   第一条内容预览: {all_data['documents'][0][:100]}...")
except Exception as e:
    print(f"⚠️ 读取数据库信息时出错: {e}")

# 5. 测试查询
test_queries = [
    "会议审议通过了哪两个议案？",
    "宁德时代",
    "董事会"
]

print("\n" + "="*30)
for query in test_queries:
    print(f"\n🔍 查询: {query}")
    try:
        docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
        print(f"   返回结果数量: {len(docs_with_scores)}")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"  [{i+1}] 距离: {score:.2f}")
            print(f"      内容预览: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"   ❌ 查询出错: {e}")
    print("-"*30)