"""
Lab 13: Local Code RAG Assistant (Backend)
==========================================
"""
import os, ast
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

MODEL_LLM = "qwen2.5-coder:3b"  # หรือใช้ qwen2.5-coder:1.5b เพื่อความเร็ว
MODEL_EMBED = "nomic-embed-text"
VECTOR_DB_PATH = "lab13_vector_db"

def extract_function_chunks(filepath: str) -> list[dict]:
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except Exception:
        return []

    chunks = []
    lines = source.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', start + 25)
            code = '\n'.join(lines[start:end])
            chunks.append({
                "name": node.name,
                "code": code,
                "file": str(filepath),
                "line_start": node.lineno
            })
    return chunks

def index_directory(directory_path: str = "restaurant_system") -> int:
    documents = []
    for root, _, files in os.walk(directory_path):
        if any(part.startswith('.') or part.startswith('__') for part in Path(root).parts):
            continue
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                chunks = extract_function_chunks(full_path)
                for c in chunks:
                    doc = Document(
                        page_content=c["code"],
                        metadata={"function": c["name"], "file": c["file"], "line": c["line_start"]}
                    )
                    documents.append(doc)

    if documents:
        embeddings = OllamaEmbeddings(model=MODEL_EMBED)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"[Index Success] Indexed {len(documents)} function chunks into {VECTOR_DB_PATH}")
        return len(documents)
    return 0

def load_vectorstore():
    if not Path(VECTOR_DB_PATH).exists():
        raise FileNotFoundError(f"Vector DB not found at {VECTOR_DB_PATH}")
    embeddings = OllamaEmbeddings(model=MODEL_EMBED)
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def classify_question(question: str) -> str:
    """จำแนกหมวดคำถามด้วยคีย์เวิร์ดภาษาไทยและอังกฤษ เพื่อสลับโหมดอย่างแม่นยำ"""
    graph_keywords = [
        "calls", "called by", "who calls", "depends on", "imports", "affects", "rely",
        "เรียก", "เรียกใช้", "ถูกเรียก", "พึ่งพา", "ขึ้นอยู่กับ", "ใช้งาน", "เชื่อมโยง", "กระทบ"
    ]
    # แปลงเป็นตัวพิมพ์เล็กและตัดช่องว่าง/อักขระพิเศษหัวท้ายออก
    q_lower = question.lower().strip()
    
    if any(kw in q_lower for kw in graph_keywords):
        return "graph"
    return "vector"

def rag_query(question: str, k: int = 3) -> dict:
    vectorstore = load_vectorstore()
    route = classify_question(question)
    docs = vectorstore.similarity_search(question, k=k)
    
    context = "\n\n".join([
        f"# File: {d.metadata.get('file')}\n# Function: {d.metadata.get('function')}\n{d.page_content}" 
        for d in docs
    ])
    
    # System Prompt แบบกำหนดโครงสร้างการตอบภาษาไทยตามโหมด
    system_instruction = (
        "คุณคือ AI ผู้ช่วยวิเคราะห์ซอร์สโค้ดระบบจัดการร้านอาหาร (Restaurant Management System)\n"
        "จงตอบคำถามเป็นภาษาไทยให้กระชับ ชัดเจน และตรงตาม Code Context ที่ให้มาเท่านั้น ห้ามคิดเพิ่มเองเด็ดขาด\n\n"
        "📋 รูปแบบการตอบตามโหมดการสืบค้น:\n\n"
        "1. หากเป็นการอธิบายตรรกะ/สูตรคำนวณ (Vector Mode):\n"
        "   - บรรทัดแรก: อธิบายหน้าที่หลักของฟังก์ชัน/ระบบนั้นสั้นๆ 1 ประโยค\n"
        "   - เนื้อหา: สรุปขั้นตอนหรือเงื่อนไขการทำงานเป็นข้อๆ (1., 2. หรือ •) โดยใช้ชื่อพารามิเตอร์หรือฟังก์ชันแบบ `code` เช่น `is_member` หรือ `VAT_RATE`\n"
        "   - บรรทัดสุดท้าย: สรุปข้อมูลที่รับเข้า (Input) และผลลัพธ์ที่ได้ (Return Output)\n\n"
        "2. หากเป็นการถามความสัมพันธ์/ใครเรียกใคร (Graph Mode):\n"
        "   - บรรทัดแรก: ระบุชื่อฟังก์ชันที่เกี่ยวข้องตรงๆ เช่น 'ฟังก์ชันที่เรียกใช้ `X` คือ `Y`'\n"
        "   - เนื้อหา: อธิบายขั้นตอนว่าฟังก์ชันดังกล่าวเรียกใช้ `X` ในขั้นตอนไหน นำผลลัพธ์ไปทำอะไรต่อ\n"
        "   - บรรทัดสุดท้าย: สรุปภาพรวมความพึ่งพากันของฟังก์ชัน\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Code Context:\n{context}\n\nQuestion: {question}")
    ])
    
    llm = ChatOllama(model=MODEL_LLM, temperature=0)
    chain = prompt | llm
    res = chain.invoke({"context": context, "question": question})
    
    sources = [{"file": d.metadata.get("file"), "function": d.metadata.get("function")} for d in docs]
    return {"answer": res.content, "sources": sources, "route": route}

if __name__ == "__main__":
    index_directory("restaurant_system")