"""
Lab 13: Streamlit Web Application (Restaurant Management Edition)
================================================================
"""
import streamlit as st
from pathlib import Path
from lab_13_coding_assistant import (
    load_vectorstore, 
    rag_query, 
    classify_question, 
    index_directory
)

# ----------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Restaurant System - Coding Assistant",
    page_icon="🍽️",
    layout="wide"
)

# ----------------------------------------------------
# 2. Cache Resource & RAG Loader
# ----------------------------------------------------
@st.cache_resource
def load_rag():
    try:
        return load_vectorstore()
    except Exception as e:
        return None

vectorstore = load_rag()

# ----------------------------------------------------
# 3. Session State Initialization
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ----------------------------------------------------
# 4. Sidebar Setup (ปรับเป็นระบบร้านอาหาร)
# ----------------------------------------------------
with st.sidebar:
    st.title("🍽️ Restaurant AI")
    st.caption("ผู้ช่วยวิเคราะห์โค้ด & ตรรกะระบบจัดการร้านอาหาร")
    st.markdown("---")
    
    # 🔄 ปุ่ม Re-index Codebase
    if st.button("🔄 Re-index Codebase", use_container_width=True):
        with st.spinner("กำลังอ่านและสร้าง Index จากโค้ดระบบร้านอาหาร..."):
            try:
                count = index_directory()
                if count > 0:
                    st.cache_resource.clear()
                    st.success(f"✅ Re-indexed {count} function chunks เรียบร้อย!")
                else:
                    st.warning("⚠️ ไม่พบไฟล์โค้ดในระบบ")
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการ Index: {e}")

    # 🗑️ ปุ่ม Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")
    
    # ⚡ คำถามแนะนำสำหรับพรีเซนต์ (ชุดคำถามสำหรับระบบร้านอาหาร)
    st.subheader("⚡ คำถามแนะนำสำหรับพรีเซนต์:")
    
    st.markdown("**🔍 Vector Mode (ถามตรรกะ/สูตรคำนวณ):**")
    if st.button("How does apply_member_discount work?", use_container_width=True, key="vec_1"):
        st.session_state.pending_prompt = "How does apply_member_discount function calculate discounts?"
        st.rerun()
        
    if st.button("Explain get_recipe_and_prep_instructions", use_container_width=True, key="vec_2"):
        st.session_state.pending_prompt = "What are the cooking instructions in get_recipe_and_prep_instructions?"
        st.rerun()

    st.markdown("**🔗 Graph Mode (ถามความเชื่อมโยงโค้ด):**")
    if st.button("Who calls get_menu_info?", use_container_width=True, key="graph_1"):
        st.session_state.pending_prompt = "Who calls get_menu_info?"
        st.rerun()

    if st.button("Who calls check_allergen_warning?", use_container_width=True, key="graph_2"):
        st.session_state.pending_prompt = "Who calls check_allergen_warning?"
        st.rerun()

# ----------------------------------------------------
# 5. Main Area Header
# ----------------------------------------------------
st.title("🍽️ Restaurant System — Coding Assistant")
st.caption("ระบบวิเคราะห์ซอร์สโค้ดและตรรกะคำนวณของระบบร้านอาหาร ด้วยเทคโนโลยี Hybrid RAG (Vector + Graph Router)")
st.markdown("---")

# ----------------------------------------------------
# 6. Display Chat History
# ----------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "mode" in message:
            st.caption(f"Mode: {message['mode']}")

# ----------------------------------------------------
# 7. Chat Input & Query Handler
# ----------------------------------------------------
prompt_input = st.chat_input("พิมพ์คำถามเกี่ยวกับโค้ดระบบจัดการร้านอาหารที่นี่...")

active_prompt = None
if prompt_input:
    active_prompt = prompt_input
elif st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if active_prompt:
    # บันทึกข้อความผู้ใช้
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # ประมวลผล RAG
    with st.chat_message("assistant"):
        with st.spinner("กำลังวิเคราะห์โครงสร้างโค้ด..."):
            if vectorstore is None:
                vectorstore = load_rag()

            if vectorstore is None:
                answer = "❌ Error: ไม่พบ Vector DB กรุณากดปุ่ม '🔄 Re-index Codebase' ทาง Sidebar ด้านซ้ายก่อนครับ"
                sources = []
                route = "vector"
            else:
                route = classify_question(active_prompt)
                result = rag_query(active_prompt, k=5)
                answer = result["answer"]
                sources = result["sources"]

        # แสดงคำตอบและ Badge โหมดการค้นหา
        st.markdown(answer)
        mode_text = "🔗 Graph" if route == "graph" else "🔍 Vector"
        st.caption(f"Mode: {mode_text}")

        # แสดง Sources
        if sources:
            with st.expander(f"📎 Sources ({len(sources)} chunks analyzed)"):
                for src in sources:
                    st.code(f"File: {src.get('file', '?')}\nFunction: {src.get('function', '?')}")

    # บันทึกประวัติฝั่ง AI
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "mode": mode_text
    })