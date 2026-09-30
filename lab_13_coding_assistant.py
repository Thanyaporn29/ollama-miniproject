"""
Lab 13: Local Code RAG Assistant (Backend)
==========================================
"""
import os, ast, sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

MODEL_LLM = "qwen2.5-coder:3b"  # หรือใช้ qwen2.5-coder:7b
MODEL_EMBED = "nomic-embed-text"
VECTOR_DB_PATH = "lab13_vector_db"

CANDIDATE_PATHS = [
    "sample_code/restaurant_system",
    "restaurant_system",
    "restaurant_dataset",
    "raw_codebase",
    "smart_inventory_system"
]

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
            end = getattr(node, 'end_lineno', start + 30)
            code = '\n'.join(lines[start:end])
            chunks.append({
                "name": node.name,
                "code": code,
                "file": str(filepath),
                "line_start": node.lineno
            })
    return chunks

def index_directory(directory_path: str = None) -> int:
    target_path = None

    if directory_path and Path(directory_path).exists():
        target_path = directory_path
    else:
        for path in CANDIDATE_PATHS:
            if Path(path).exists():
                target_path = path
                break

    if not target_path:
        print("[Error] ไม่พบโฟลเดอร์เก็บไฟล์โค้ดในระบบ")
        return 0

    print(f"[Indexing] กำลังอ่านไฟล์โค้ดจาก: '{target_path}'...")
    documents = []

    for root, _, files in os.walk(target_path):
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
        print(f"[Index Success] Indexed {len(documents)} function chunks into '{VECTOR_DB_PATH}'")
        return len(documents)

    print(f"[Warning] ไม่พบไฟล์ .py ในโฟลเดอร์ '{target_path}'")
    return 0

def load_vectorstore():
    if not Path(VECTOR_DB_PATH).exists():
        raise FileNotFoundError(f"Vector DB not found at {VECTOR_DB_PATH}")
    embeddings = OllamaEmbeddings(model=MODEL_EMBED)
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def classify_question(question: str) -> str:
    """จำแนกหมวดคำถามด้วยคีย์เวิร์ดภาษาไทยและอังกฤษ เพื่อสลับโหมดอย่างแม่นยำ"""
    graph_keywords = [
        "calls", "called by", "who calls", "depends on", "dependency", "dependencies", "imports", "affects", "rely",
        "เรียก", "เรียกใช้", "ถูกเรียก", "พึ่งพา", "ขึ้นอยู่กับ", "ใช้งาน", "เชื่อมโยง", "กระทบ", "องค์ประกอบ", "โครงสร้าง"
    ]
    q_lower = question.lower().strip()
    if any(kw in q_lower for kw in graph_keywords):
        return "graph"
    return "vector"

def rag_query(question: str, k: int = 3) -> dict:
    vectorstore = load_vectorstore()
    route = classify_question(question)
    docs = vectorstore.similarity_search(question, k=k)
    
    context = "\n\n".join([
        f"File: {d.metadata.get('file')} | Function: {d.metadata.get('function')}\n{d.page_content}" 
        for d in docs
    ])
    
    system_instruction = (
    "คุณคือ AI ผู้ช่วยวิเคราะห์ซอร์สโค้ดระบบจัดการร้านอาหาร\n"
    "ข้อกำหนดในการตอบ:\n"
    "1. ตอบเป็นภาษาไทย กระชับ ตรงประเด็น และห้ามใช้สัญลักษณ์ [ ] เด็ดขาด\n"
    "2. ชื่อฟังก์ชัน ชื่อไฟล์ หรือตัวแปร ให้แสดงเป็น Inline Code เช่น `process_payment`\n\n"
    "รูปแบบการตอบแบ่งตามโหมด:\n\n"
    "--- กรณี Vector Mode (ถามขั้นตอน/ตรรกะการทำงาน) ---\n"
    "ฟังก์ชัน `ชื่อฟังก์ชัน` ในไฟล์ `ชื่อไฟล์` มีขั้นตอนการคิดเงินและคำนวณดังนี้:\n"
    "1. **ชื่อขั้นตอน:** อธิบายตรรกะหรือฟังก์ชันที่เรียกใช้\n"
    "2. **ชื่อขั้นตอน:** อธิบายสูตรคำนวณหรือเงื่อนไข\n"
    "สรุป: ฟังก์ชันนี้รับพารามิเตอร์อะไร และคืนค่าผลลัพธ์เป็นอะไร\n\n"
    "--- กรณี Graph Mode (ถามความพึ่งพา/สายการเรียก) ---\n"
    "ฟังก์ชัน `ชื่อฟังก์ชัน` ในไฟล์ `ชื่อไฟล์` พึ่งพาการทำงานของฟังก์ชันต่อไปนี้:\n"
    "1. `ชื่อฟังก์ชันที่ถูกเรียก` : ถูกเรียกไปทำหน้าที่...\n"
    "2. `ชื่อฟังก์ชันที่ถูกเรียก` : ถูกเรียกไปทำหน้าที่...\n"
    "สรุป: `ชื่อฟังก์ชันหลัก` ใช้ฟังก์ชันอื่นๆ เพื่อทำอะไรในภาพรวม"
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
    index_directory()