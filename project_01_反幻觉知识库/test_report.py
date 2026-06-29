import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi

# ===== 配置 =====
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DISTANCE_THRESHOLD = 7500.0

print("🧠 正在加载向量数据库...")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY
)
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
llm = Tongyi(
    model="qwen-plus",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# ===== 测试用例 =====
# 格式: (问题, 预期结果: "有答案" 或 "无答案")
test_cases = [
    # 有答案的问题（从PDF里能找到）
    ("卫光生物的主要业务是什么？", "有答案"),
    ("公司的净利润是多少？", "有答案"),
    ("公司有哪些在售产品？", "有答案"),
    ("本次发行的保荐机构是谁？", "有答案"),
    ("公司的注册地址在哪里？", "有答案"),
    ("公司的股票代码是多少？", "有答案"),

    # 无答案的问题（PDF里没有）
    ("公司创始人的家乡是哪里？", "无答案"),
    ("公司有几个子公司？", "无答案"),
    ("公司的主要竞争对手有哪些？", "无答案"),
    ("公司的注册资本是多少？", "无答案"),
    ("公司什么时候上市的？", "无答案"),
    ("公司的董事长是谁？", "无答案"),
]

print("\n" + "=" * 60)
print("📊 测试报告")
print("=" * 60)

correct = 0
total = len(test_cases)
results = []

for question, expected in test_cases:
    # 1. 检索
    docs_with_scores = vector_db.similarity_search_with_score(question, k=3)
    docs = [doc for doc, _ in docs_with_scores]
    distances = [score for _, score in docs_with_scores]
    best_distance = distances[0] if distances else 9999.0

    # 2. 判断是否拒答
    is_rejected = best_distance > DISTANCE_THRESHOLD
    actual = "无答案" if is_rejected else "有答案"

    # 3. 判断是否正确
    is_correct = (actual == expected)
    if is_correct:
        correct += 1

    status = "✅" if is_correct else "❌"
    results.append({
        "question": question,
        "expected": expected,
        "actual": actual,
        "distance": best_distance,
        "status": status
    })

# 打印详细结果
for r in results:
    print(f"\n{r['status']} 问: {r['question']}")
    print(f"   预期: {r['expected']} | 实际: {r['actual']} | 距离: {r['distance']:.2f}")

# 统计
print("\n" + "=" * 60)
print(f"📈 准确率: {correct}/{total} = {correct / total * 100:.1f}%")
print("=" * 60)

# 额外分析
print("\n📊 详细分析：")
print(f"  - 阈值设置: {DISTANCE_THRESHOLD}")
has_answer_qs = [r for r in results if r['expected'] == '有答案']
no_answer_qs = [r for r in results if r['expected'] == '无答案']
if has_answer_qs:
    print(f"  - 有答案问题平均距离: {sum([r['distance'] for r in has_answer_qs]) / len(has_answer_qs):.2f}")
if no_answer_qs:
    print(f"  - 无答案问题平均距离: {sum([r['distance'] for r in no_answer_qs]) / len(no_answer_qs):.2f}")