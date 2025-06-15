from typing import Dict
# Langchain imports
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
from LocalFileManager import FileManager
load_dotenv(dotenv_path=".env", override=False)
# LLM API
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL2")

class RAGFlow:
    """RAG流程执行器 - 专注于问答流程"""
    def __init__(self, here_file_manager: FileManager):
        self.file_manager = here_file_manager
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    def query(self, question: str) -> Dict:
        """执行RAG查询流程"""
        if not self.file_manager.has_documents():
            return {
                "answer": "还没有索引文档，请先上传文档。",
                "sources": []
            }
        try:
            # 获取检索器
            retriever = self.file_manager.get_retriever(k=5)
            # 创建QA链
            qa = RetrievalQA.from_chain_type(
                llm=self.model,
                retriever=retriever,
                return_source_documents=True
            )
            # 执行查询
            result = qa.invoke(question)
            answer = result["result"]

            # 提取唯一来源文件
            sources = []
            source_files = set()

            for doc in result["source_documents"]:
                source_file = doc.metadata.get("source_file", "未知")
                if source_file not in source_files:
                    source_files.add(source_file)
                    # 获取内容预览
                    content_preview = doc.page_content[:150] + "..." if len(
                        doc.page_content) > 150 else doc.page_content
                    sources.append({
                        "file": source_file,
                        "content": content_preview
                    })

            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"查询错误: {str(e)}",
                "sources": []
            }
    def query_self(self, question: str) -> Dict:
        """执行详细RAG流程 - 检索与生成分离"""
        if not self.file_manager.has_documents():
            return {"answer": "还没有索引文档，请先上传文档。", "sources": []}

        try:
            # 1. 检索相关文档
            docs = self.file_manager.docsearch.similarity_search(question, k=5)

            # 2. 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])

            # 3. 生成回答
            prompt = f"基于以下上下文回答问题：\n\n上下文：{context}\n\n问题：{question}\n\n回答："
            answer = self.model.invoke(prompt).content

            # 4. 提取来源
            sources = []
            source_files = set()
            for doc in docs:
                source_file = doc.metadata.get("source_file", "未知")
                if source_file not in source_files:
                    source_files.add(source_file)
                    content_preview = doc.page_content[:150] + "..." if len(
                        doc.page_content) > 150 else doc.page_content
                    sources.append({"file": source_file, "content": content_preview})

            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {"answer": f"查询错误: {str(e)}", "sources": []}
# Create an MCP server
mcp = FastMCP("RAGFlow")
# 创建RAGFlow实例
#vector_db = VectorDatabase(persist_directory=os.getenv("VECTOR_DB_PATH"))
file_manager = FileManager(persist_directory=os.getenv("VECTOR_DB_PATH"), chunk_size=int(os.getenv("CHUNK_SIZE")),
                               chunk_overlap=int(os.getenv("CHUNK_OVERLAP")), folder_path=os.getenv("UPLOAD_FOLDER"))
rag_flow = RAGFlow(file_manager)
@mcp.tool()
def rag_query(question: str) -> str:
    """Query documents using RAG"""
    result = rag_flow.query(question)#or query_detail(question)
    response = f"Answer: {result['answer']}\n\n"
    return response
if __name__ == "__main__":
    '''add_document("cooking.txt")'''
    #file_manager.remove_document("Query2doc.pdf")
    #print(rag_query("Query2doc技术讲了什么"))
    '''remove_document("cooking.txt")'''
    mcp.run()