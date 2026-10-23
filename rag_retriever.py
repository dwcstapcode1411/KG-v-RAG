#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
MODULE TRUY XUẤT TÀI LIỆU (DOCUMENT RETRIEVER - VECTOR / TF-IDF RAG)
Học phần: Phát triển các hệ thống thông minh (PTHTTM) - Mục 5.3 & 5.7
=============================================================================
"""

import os
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# BỘ MỞ RỘNG TRUY VẤN (QUERY EXPANSION) ĐA MIỀN
# =============================================================================
class QueryExpander:
    """Mở rộng truy vấn ngôn ngữ tự nhiên bằng cách tiêm từ đồng nghĩa & ngữ cảnh y/tế/tài chính/vàng."""

    # Từ điển đồng nghĩa chuyên ngành tiếng Việt - Anh
    SYNONYM_MAP = {
        # Y tế
        "tiểu đường": "đái tháo đường diabetes glucose HbA1c đường huyết insulin",
        "béo phì": "BMI obese thừa cân overweight cân nặng thể trạng đề kháng insulin",
        "huyết áp": "hypertension tăng huyết áp đồng mắc comorbidity tim mạch",
        "chẩn đoán": "HbA1c tiêu chuẩn ADA glucose xét nghiệm ngưỡng",
        "nguy cơ": "tỷ lệ mắc bệnh risk factor rủi ro phòng ngừa",
        "thuốc": "metformin SGLT2 GLP-1 điều trị liệu pháp dược lý",
        "tuổi": "nhóm tuổi AgeGroup lứa tuổi trung niên",
        # Tài chính
        "khủng hoảng": "crisis sụt giảm crash drawdown sụp đổ biến động",
        "chứng khoán": "stock index thị trường cổ phiếu đầu tư",
        "tương quan": "correlation liên kết lây lan contagion hệ số",
        "tín dụng": "DTI nợ vỡ nợ default rủi ro credit score",
        "lãi suất": "Fed thắt chặt tiền tệ rate hike nới lỏng",
        "đầu tư": "vốn danh mục portfolio lợi nhuận thua lỗ rủi ro",
        # Vàng
        "vàng": "gold giá vàng kim loại quý commodity",
        "rsi": "oversold overbought quá bán quá mua kỹ thuật technical",
        "chiến tranh": "xung đột geopolitical safe-haven trú ẩn an toàn địa chính trị",
        "svb": "Silicon Valley Bank sụp đổ ngân hàng banking crisis",
        "mua vàng": "tích trữ phòng hộ safe-haven demand nhu cầu",
        "bán vàng": "chốt lời take profit điều chỉnh correction",
    }

    # Ánh xạ từ ngôn ngữ tự nhiên sang khái niệm miền
    DOMAIN_CONTEXT = {
        "diabetes": "BMI béo phì thể trạng nguy cơ đái tháo đường cân nặng HbA1c glucose tiểu đường bệnh nhân chẩn đoán",
        "finance": "chứng khoán thị trường chỉ số đầu tư rủi ro tín dụng khủng hoảng lợi suất biến động",
        "gold": "giá vàng thị trường vàng RSI kỹ thuật vĩ mô lãi suất Fed sự kiện địa chính trị"
    }

    @classmethod
    def expand(cls, query: str, domain: str = None) -> str:
        """Mở rộng truy vấn bằng đồng nghĩa + ngữ cảnh miền."""
        q_low = query.lower()
        expansions = []

        # 1. Tiêm đồng nghĩa cho từ khóa khớp
        for trigger, synonyms in cls.SYNONYM_MAP.items():
            if trigger in q_low:
                expansions.append(synonyms)

        # 2. Nhận diện câu hỏi chứa số liệu cá nhân
        has_measurements = bool(
            re.search(r'\d+\s*(?:kg|kí|cân)', q_low) or
            re.search(r'\d+m\d+|cao\s*\d+', q_low) or
            re.search(r'bmi\s*(?:nhỏ|dưới|lớn|trên|<|>|=)', q_low) or
            re.search(r'hba1c\s*(?:là|=|của)', q_low) or
            re.search(r'(?:glucose|đường huyết)\s*\d+', q_low) or
            re.search(r'(?:dti|nợ)\s*\d+', q_low) or
            re.search(r'rsi\s*(?:là|=|đang)?\s*\d+', q_low) or
            re.search(r'\d+\s*(?:triệu|tỷ)', q_low)
        )

        if has_measurements and domain:
            expansions.append(cls.DOMAIN_CONTEXT.get(domain, ""))

        # 3. Ghép truy vấn gốc + mở rộng
        if expansions:
            return query + " " + " ".join(expansions)
        return query


class DocumentRetriever:
    """Bộ truy xuất tài liệu phi cấu trúc sử dụng TF-IDF và Cosine Similarity."""

    def __init__(self, docs_dir: str = "documents"):
        self.docs_dir = docs_dir
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self._load_and_index_documents()

    def _load_and_index_documents(self):
        """Đọc và chia đoạn tài liệu (Chunking) kèm lưu trữ siêu dữ liệu (Metadata)."""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)

        files = [f for f in os.listdir(self.docs_dir) if f.endswith(".txt")]
        all_chunks = []

        for fname in sorted(files):
            fpath = os.path.join(self.docs_dir, fname)
            domain = "diabetes" if "diab" in fname else ("finance" if "fin" in fname or "credit" in fname else "gold")

            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            # Phân tách theo khối tài liệu [DocID: ...]
            raw_sections = re.split(r"(?=\[DocID:\s*[\w\-]+\])", content)

            for sec in raw_sections:
                sec = sec.strip()
                if not sec or sec.startswith("#"):
                    continue

                # Trích xuất mã DocID
                doc_id_match = re.search(r"\[DocID:\s*([\w\-]+)\]", sec)
                doc_id = doc_id_match.group(1) if doc_id_match else f"{fname}_chunk"

                # Trích xuất tiêu đề đề mục
                lines = sec.split("\n")
                title = lines[1].strip() if len(lines) > 1 and lines[1].strip() else lines[0].strip()
                clean_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else sec

                chunk_obj = {
                    "id": doc_id,
                    "title": title,
                    "text": clean_text,
                    "source": fname,
                    "domain": domain
                }
                all_chunks.append(chunk_obj)

        self.chunks = all_chunks

        if self.chunks:
            corpus = [c["text"] for c in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, domain: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Truy xuất Top-k đoạn văn tương đồng với câu hỏi dựa trên độ tương đồng Cosine.
        Hỗ trợ Pre-filtering theo miền nghiệp vụ (domain) và Query Expansion thông minh.
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Mở rộng truy vấn bằng QueryExpander đa miền
        expanded_query = QueryExpander.expand(query, domain)

        query_vec = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        ranked_indices = np.argsort(scores)[::-1]
        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])

            # Lọc theo domain nếu có yêu cầu
            if domain and domain != "all" and chunk["domain"] != domain:
                continue

            results.append({
                "id": chunk["id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "source": chunk["source"],
                "domain": chunk["domain"],
                "score": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results

    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Đóng gói các đoạn bằng chứng thành chuỗi ngữ cảnh DOCUMENT EVIDENCE."""
        if not results:
            return "Không tìm thấy bằng chứng tài liệu liên quan."

        blocks = []
        for r in results:
            blocks.append(f"[{r['id']} | Nguồn: {r['source']} | Score: {r['score']:.3f}]\n{r['text']}")
        return "\n\n".join(blocks)
