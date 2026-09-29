"""
Lab 12: Hybrid RAG Engine — Vector + Graph
==========================================
"""
import sys, io, ast, pathlib
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

class QuestionRouter:
    """Route questions to 'vector' or 'graph' retriever based on keywords."""

    GRAPH_KEYWORDS = [
        "calls", "called by", "who calls", "what calls",
        "imports", "depends on", "impact", "affects",
        "uses", "used by", "เรียก", "ใช้", "กระทบ", "ขึ้นกับ"
    ]

    def classify(self, question: str) -> str:
        q_lower = question.lower()
        for kw in self.GRAPH_KEYWORDS:
            if kw in q_lower:
                return "graph"
        return "vector"

    def classify_with_reason(self, question: str) -> dict:
        route = self.classify(question)
        reason = "Matched graph dependency keyword" if route == "graph" else "General code logic/conceptual question"
        return {"route": route, "reason": reason, "question": question}

def extract_function_name(question: str) -> Optional[str]:
    words = question.replace("?", "").replace("()", "").split()
    trigger = {"calls", "call", "called", "uses", "use", "imports", "import"}
    for i, w in enumerate(words):
        if w.lower() in trigger:
            if i > 0 and words[i - 1].lower() not in {"what", "who", "which", "does", "function", "the"}:
                return words[i - 1].strip("'\".,")
            if i + 1 < len(words):
                return words[i + 1].strip("'\".,")
    return None

if __name__ == "__main__":
    router = QuestionRouter()
    print("Testing QuestionRouter:")
    print("Q: Who calls get_menu_item? ->", router.classify_with_reason("Who calls get_menu_item?"))
    print("Q: How does apply_member_discount work? ->", router.classify_with_reason("How does apply_member_discount work?"))