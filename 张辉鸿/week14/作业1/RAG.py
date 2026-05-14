

import os
from pathlib import Path

# ============================================================
# 第一部分：配置
# ============================================================

# API配置（阿里云DashScope，兼容OpenAI接口）
API_KEY = "sk-4fedee4ece6541d3b17a7173f0b3c16f"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-flash"          # 对话模型
EMBEDDING_MODEL = "text-embedding-v2"  # 嵌入模型

# 文档路径
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "knowledge_docs"       # 存放知识文档的目录
VECTOR_STORE_DIR = BASE_DIR / "vector_store"  # FAISS向量存储目录

# 分割参数
CHUNK_SIZE = 500      # 每个文本块的最大字符数
CHUNK_OVERLAP = 50    # 相邻文本块的重叠字符数

# 检索参数
TOP_K = 4  # 每次检索返回的最相关文档块数量

# ============================================================
# 第二部分：初始化模型
# ============================================================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1,  # 低温度保证回答稳定性
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)

print("模型初始化完成")
print(f"  LLM: {LLM_MODEL}")
print(f"  Embedding: {EMBEDDING_MODEL}")

# ============================================================
# 第三部分：文档加载
# ============================================================

from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(docs_dir: Path) -> list:
    """
    从指定目录加载所有 .txt 文档。
    返回 LangChain Document 对象列表，每个 Document 包含 page_content 和 metadata。
    """
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True)
        print(f"已创建文档目录: {docs_dir}")
        return []

    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",           # 匹配所有 .txt 文件
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"文档加载完成，共 {len(docs)} 个文档")
    for doc in docs:
        source = Path(doc.metadata["source"]).name
        print(f"  - {source} ({len(doc.page_content)} 字)")
    return docs


# ============================================================
# 第四部分：文本分割
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs: list) -> list:
    """
    将长文档分割成适合检索的文本块。
    使用 RecursiveCharacterTextSplitter 按中文标点优先级递归分割。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 分割符优先级：段落 > 换行 > 句号 > 感叹号 > 问号 > 分号 > 逗号 > 空格
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"文本分割完成，共 {len(chunks)} 个文本块 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ============================================================
# 第五部分：向量嵌入与存储
# ============================================================

from langchain_community.vectorstores import FAISS


def create_vector_store(chunks: list, save_dir: Path):
    """对文本块进行向量嵌入，创建FAISS向量存储并持久化到磁盘"""
    vector_store = FAISS.from_documents(chunks, embeddings)
    save_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_dir))
    print(f"向量存储已创建并保存到: {save_dir} ({len(chunks)} 条向量)")
    return vector_store


def load_vector_store(save_dir: Path):
    """从磁盘加载已有的FAISS向量存储"""
    index_file = save_dir / "index.faiss"
    if not index_file.exists():
        return None
    print(f"从 {save_dir} 加载已有向量存储...")
    return FAISS.load_local(
        str(save_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # 仅加载自己创建的文件
    )


# ============================================================
# 第六部分：检索器
# ============================================================

def get_retriever(vector_store):
    """
    基于向量存储创建检索器。
    使用余弦相似度（FAISS默认）检索与问题最相关的 TOP_K 个文本块。
    """
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


# ============================================================
# 第七部分：RAG问答链
# ============================================================

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 自定义提示词模板 —— 参考 04-government-advanced-rag 的 BASIC_QA_TEMPLATE
RAG_PROMPT = PromptTemplate.from_template(
    "你是一个专业的智能助手。请根据以下资料回答用户的问题。\n"
    '如果资料中没有相关信息，请如实说明"资料中未找到相关信息"，不要编造内容。\n'
    "\n"
    "资料：\n"
    "{context}\n"
    "\n"
    "问题：{question}\n"
    "\n"
    "回答："
)


def create_rag_chain(retriever):
    """
    构建 RAG 问答链：
    1. 检索器从向量库中检索相关文本块
    2. 将检索结果作为上下文填入提示词
    3. LLM 基于上下文生成回答
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 将所有检索结果拼接到提示词中
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,  # 返回参考来源
    )


# ============================================================
# 第八部分：交互式问答
# ============================================================

def run_qa_loop(rag_chain):
    """交互式问答循环"""
    print("\n" + "=" * 60)
    print("  本地知识库问答系统已就绪")
    print("  输入问题开始问答，输入 quit / exit / q 退出")
    print("=" * 60 + "\n")

    while True:
        question = input("请输入问题：").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        print("正在检索和生成回答...")
        result = rag_chain.invoke({"query": question})

        print(f"\n回答：{result['result']}\n")

        # 显示参考来源
        print("参考来源：")
        for i, doc in enumerate(result["source_documents"], 1):
            source = Path(doc.metadata.get("source", "unknown")).name
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i}] {source} | {preview}...")
        print()


# ============================================================
# 第九部分：主流程
# ============================================================

def main():
    # 1. 尝试加载已有向量存储，不存在则创建
    vector_store = load_vector_store(VECTOR_STORE_DIR)

    if vector_store is None:
        print("\n首次运行，开始构建知识库...")
        docs = load_documents(DOCS_DIR)
        if not docs:
            print(f"\n请将 .txt 文档放入 {DOCS_DIR} 目录后重新运行")
            return
        chunks = split_documents(docs)
        vector_store = create_vector_store(chunks, VECTOR_STORE_DIR)
    else:
        print(f"向量存储加载成功，包含 {vector_store.index.ntotal} 条向量")

    # 2. 创建检索器和RAG链
    retriever = get_retriever(vector_store)
    rag_chain = create_rag_chain(retriever)
    print("RAG链构建完成")

    # 3. 交互式问答
    run_qa_loop(rag_chain)


if __name__ == "__main__":
    main()
