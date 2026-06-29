import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 加载API Key
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 请先在 .env 文件中设置 DASHSCOPE_API_KEY")

print("📄 正在加载PDF...")
loader = PyPDFLoader("report.pdf")
documents = loader.load()
print(f"✅ 加载完成，共 {len(documents)} 页")

print("✂️ 正在切片...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 切片完成，共 {len(chunks)} 个片段")

print("🧠 正在创建向量数据库...")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_db.persist()
print("✅ 向量数据库创建成功！保存在 ./chroma_db 目录")