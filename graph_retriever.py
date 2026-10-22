#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
MODULE TRUY XUẤT ĐỒ THỊ TRI THỨC (GRAPH RETRIEVER - NEO4J & MOCK LAYER)
Học phần: Phát triển các hệ thống thông minh (PTHTTM) - Mục 5.2, 5.4 - 5.7
=============================================================================
"""

import os
import socket
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


class GraphRetriever:
    """Bộ truy xuất sự kiện Đồ thị tri thức (Knowledge Graph Facts) từ Neo4j hoặc Mock Model."""

    def __init__(self, uri: str = None, username: str = None, password: str = None, database: str = None):
        self.uri = uri or os.getenv("NEO4J_URI") or "bolt://localhost:7687"
        self.username = username or os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
        self.password = password or os.getenv("NEO4J_PASSWORD") or "password"
        self.database = database or os.getenv("NEO4J_DATABASE") or "neo4j"
        self.driver = None
        self.is_connected = False
        self._init_connection()

    def _init_connection(self):
        """Khởi tạo kết nối an toàn với kiểm tra socket nhanh để tránh lag khi offline."""
        if GraphDatabase is None:
            self.is_connected = False
            return

        # Kiểm tra nhanh cổng kết nối bằng socket (timeout 0.5s)
        try:
            parsed = urlparse(self.uri.replace("bolt://", "http://").replace("neo4j+s://", "https://").replace("neo4j://", "http://"))
            host = parsed.hostname or "localhost"
            port = parsed.port or 7687

            # Nếu là localhost, kiểm tra nhanh cổng
            if host in ["localhost", "127.0.0.1", "::1"]:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                result = s.connect_ex((host, port))
                s.close()
                if result != 0:
                    self.is_connected = False
                    return

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                connection_timeout=1.0,
                max_connection_lifetime=60
            )
            self.driver.verify_connectivity()
            self.is_connected = True
        except Exception:
            self.is_connected = False

    def close(self):
        if self.driver:
            self.driver.close()

    def retrieve_facts(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """
        Truy xuất các sự kiện tri thức đồ thị (Triples/Paths) phù hợp với câu hỏi và miền nghiệp vụ.
        Nếu có Neo4j sống, truy vấn trực tiếp; nếu không, sử dụng cơ sở tri thức đồ thị chuẩn hóa.
        """
        query_lower = query.lower()
        facts = []

        if self.is_connected and self.driver:
            try:
                facts = self._query_neo4j(query_lower, domain)
                if facts:
                    return facts
            except Exception:
                pass

        # Dự phòng bằng Đồ thị tri thức định chuẩn (Ground Truth Knowledge Graph)
        return self._query_mock_graph(query_lower, domain)

    def _query_neo4j(self, query_lower: str, domain: str) -> List[Dict[str, Any]]:
        """Thực thi truy vấn Cypher linh hoạt trên Neo4j theo thực thể nhận diện."""
        results = []
        with self.driver.session(database=self.database) as session:
            if domain == "diabetes":
                cypher = """
                MATCH (h:HbA1cBand)-[r:INDICATES]->(c:Condition)
                RETURN h.name AS subject, type(r) AS relation, c.name AS object, r.source AS source, r.threshold AS detail
                UNION
                MATCH (b:BMIBand)-[r:INCREASES_RISK_OF]->(c:Condition)
                RETURN b.name AS subject, type(r) AS relation, c.name AS object, r.source AS source, r.description AS detail
                UNION
                MATCH (c1:Condition)-[r:COMORBID_WITH]->(c2:Condition)
                RETURN c1.name AS subject, type(r) AS relation, c2.name AS object, r.source AS source, r.description AS detail
                """
                records = session.run(cypher).data()
                for rec in records:
                    results.append({
                        "subject": rec["subject"],
                        "relation": rec["relation"],
                        "object": rec["object"],
                        "source": rec.get("source", "Neo4j Graph"),
                        "detail": rec.get("detail", "")
                    })
            elif domain == "finance":
                cypher = """
                MATCH (s1:StockIndex)-[r:CORRELATED_WITH]->(s2:StockIndex)
                WHERE s1.code < s2.code
                RETURN s1.name AS subject, 'CORRELATED_WITH' AS relation, s2.name AS object, 'Correlation Matrix' AS source, 'corr=' + toString(r.corr_value) AS detail
                LIMIT 10
                """
                records = session.run(cypher).data()
                for rec in records:
                    results.append({
                        "subject": rec["subject"],
                        "relation": rec["relation"],
                        "object": rec["object"],
                        "source": rec.get("source", "Neo4j Graph"),
                        "detail": rec.get("detail", "")
                    })
            elif domain == "gold":
                cypher = """
                MATCH (f:Factor)-[r:INFLUENCES]->(c:Commodity)
                RETURN f.name AS subject, 'INFLUENCES' AS relation, c.name AS object, 'Macro Model' AS source, 'Tác động ' + r.direction AS detail
                """
                records = session.run(cypher).data()
                for rec in records:
                    results.append({
                        "subject": rec["subject"],
                        "relation": rec["relation"],
                        "object": rec["object"],
                        "source": rec.get("source", "Neo4j Graph"),
                        "detail": rec.get("detail", "")
                    })
        return results

    def _query_mock_graph(self, q: str, domain: str) -> List[Dict[str, Any]]:
        """Mô hình hóa các bộ ba sự kiện Đồ thị tri thức chính xác của 3 miền bài toán có xếp hạng ngữ nghĩa."""
        facts = []

        if domain == "diabetes":
            kb = [
                {"subject": "HbA1cBand (Diabetes range: >= 6.5%)", "relation": "INDICATES", "object": "Condition (Diabetes)", "source": "ADA Standards of Care", "detail": "Tiêu chuẩn chẩn đoán xác định, độ tin cậy 98%"},
                {"subject": "BMIBand (Obese: BMI >= 30)", "relation": "INCREASES_RISK_OF", "object": "Condition (Diabetes)", "source": "WHO Guidelines", "detail": "Tỷ lệ mắc bệnh thực tế 46.40% (gấp 6.76 lần nhóm Normal)"},
                {"subject": "Condition (Hypertension)", "relation": "COMORBID_WITH", "object": "Condition (Diabetes)", "source": "Clinical Guidelines", "detail": "33.58% bệnh nhân tiểu đường đồng mắc tăng huyết áp"},
                {"subject": "GlucoseBand (High: >= 200 mg/dL)", "relation": "INDICATES", "object": "Condition (Diabetes)", "source": "WHO/ADA Guidelines", "detail": "Đường huyết ngẫu nhiên hoặc sau nghiệm pháp OGTT 2h"},
                {"subject": "GlucoseBand (Elevated: 140 - 199 mg/dL)", "relation": "INDICATES", "object": "Condition (Prediabetes)", "source": "ADA Standards of Care", "detail": "Rối loạn dung nạp glucose (IGT), tỷ lệ mắc bệnh 36.84%"},
                {"subject": "Patient Cohort (Obese + HbA1c >= 6.5%)", "relation": "UNDIAGNOSED_HIGH_RISK", "object": "38 ca tiền lâm sàng", "source": "Multi-hop Graph Reasoning", "detail": "Phát hiện 38 bệnh nhân có nguy cơ bệnh lý cao nhưng chưa có nhãn chẩn đoán"},
                {"subject": "AgeGroup (35-49 & 50-64)", "relation": "PEAK_INCIDENCE_FOR", "object": "Condition (Diabetes)", "source": "Clinical Statistical Layer", "detail": "Tỷ lệ mắc bệnh đỉnh điểm vượt 51.8% - 53.4%"}
            ]
            facts.extend(kb)

        elif domain == "finance":
            kb = [
                {"subject": "StockIndex (^GSPC: S&P 500)", "relation": "CORRELATED_WITH (corr=0.8408)", "object": "StockIndex (^GDAXI: DAX 40)", "source": "Cross-Market Correlation", "detail": "Tương quan chuỗi giá cực cao giữa thị trường Mỹ và Đức"},
                {"subject": "StockIndex (^GSPC: S&P 500)", "relation": "CORRELATED_WITH (corr=0.7665)", "object": "StockIndex (^N225: Nikkei 225)", "source": "Cross-Market Correlation", "detail": "Tương quan cao giữa thị trường Mỹ và Nhật Bản"},
                {"subject": "MarketEvent (Global Financial Crisis 2008)", "relation": "TRIGGERED_MAX_DRAWDOWN", "object": "NIFTY 50 (-78.95%) & SSE (-76.71%)", "source": "GFC 2008 Analysis", "detail": "Thị trường mới nổi chịu tổn thương nặng nề nhất toàn cầu"},
                {"subject": "MarketEvent (Global Financial Crisis 2008)", "relation": "TRIGGERED_MAX_DRAWDOWN", "object": "S&P 500 (-60.73%) & FTSE 100 (-71.36%)", "source": "GFC 2008 Analysis", "detail": "Sụt giảm kỷ lục của các thị trường tài chính phát triển"},
                {"subject": "MarketEvent (COVID-19 Crash 2020)", "relation": "SEVEREST_REGIONAL_IMPACT", "object": "Region (Europe: -49.28%)", "source": "Pandemic Liquidity Shock", "detail": "Châu Âu chịu cú sốc phong tỏa đầu tiên sâu nhất, phục hồi hình chữ V"},
                {"subject": "StockIndex (^GSPC: S&P 500)", "relation": "HISTORICAL_MAX_DRAWDOWN", "object": "Dot-com Bubble 2000 (-75.59%)", "source": "Historical Time Series", "detail": "Vỡ bong bóng định giá công nghệ đi kèm độ biến động 105.21%"},
                {"subject": "Region (Asia-Pacific)", "relation": "LARGEST_INDEX_SHARE", "object": "4 chỉ số (Nikkei, SSE, NIFTY, IDX)", "source": "Geographic Knowledge Layer", "detail": "Khu vực kinh tế chiếm tỷ trọng chỉ số lớn nhất khảo sát"}
            ]
            facts.extend(kb)

        elif domain == "gold":
            kb = [
                {"subject": "Factor (Fed Interest Rates / USD)", "relation": "INFLUENCES (direction: negative)", "object": "Commodity (Gold)", "source": "Macroeconomic Model", "detail": "Lãi suất tăng làm tăng chi phí cơ hội nắm giữ, tạo áp lực giảm giá"},
                {"subject": "Factor (Geopolitical Conflicts)", "relation": "INFLUENCES (direction: positive)", "object": "Commodity (Gold)", "source": "Safe-Haven Model", "detail": "Chiến sự kích hoạt mạnh mẽ dòng vốn phòng vệ vào vàng"},
                {"subject": "GoldEvent (SVB Collapse 2023-03-10)", "relation": "OCCURRED_ON_AND_SURGED", "object": "+5.69% sau 5 phiên", "source": "Banking Contagion Event", "detail": "Bất ổn ngân hàng Mỹ đẩy giá vàng bứt phá mạnh"},
                {"subject": "GoldEvent (Hamas-Israel Conflict 2023-10-09)", "relation": "OCCURRED_ON_AND_SURGED", "object": "+3.75% sau 5 phiên", "source": "Middle East Geopolitical Shock", "detail": "Rủi ro địa chính trị kích hoạt làn sóng mua vàng phòng hộ"},
                {"subject": "GoldEvent (Fed 50bps Rate Cut 2024-09-18)", "relation": "OCCURRED_ON_AND_SURGED", "object": "+3.31% sau 5 phiên", "source": "Monetary Easing Pivot", "detail": "Đảo chiều chu kỳ nới lỏng đẩy vàng lập đỉnh lịch sử mọi thời đại"},
                {"subject": "RSIState (Oversold: RSI < 30)", "relation": "PREDICTS_MEAN_REVERSION", "object": "+0.66% kỳ vọng sau 5 phiên (Win Rate: 61.5%)", "source": "Graph Multi-hop Traversal [:NEXT*5]", "detail": "Quy luật hồi quy trung bình định lượng trên 109 phiên quá bán"},
                {"subject": "RSIState (Overbought) + BandState (Above Upper)", "relation": "SIGNALS_EXTREME_EUPHORIA", "object": "68 phiên căng cứng kỹ thuật", "source": "Technical Indicators Confluence", "detail": "Cảnh báo vùng rủi ro điều chỉnh kỹ thuật ngắn hạn cao"}
            ]
            facts.extend(kb)

        # Xếp hạng và chọn lọc facts liên quan nhất dựa trên truy vấn
        if q:
            scored_facts = []
            tokens = [t.strip() for t in q.lower().split() if len(t.strip()) > 1]
            for item in facts:
                haystack = f"{item['subject']} {item['relation']} {item['object']} {item.get('detail', '')}".lower()
                relevance = sum(2 if t in haystack else 0 for t in tokens)
                scored_facts.append((relevance, item))
            scored_facts.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in scored_facts[:6]]

        return facts[:6]

    def format_graph_facts(self, facts: List[Dict[str, Any]]) -> str:
        """Định dạng danh sách sự kiện đồ thị thành khối văn bản GRAPH FACTS."""
        if not facts:
            return "Không tìm thấy sự kiện đồ thị phù hợp."

        lines = []
        for f in facts:
            lines.append(f"- ({f['subject']}) --[{f['relation']}]--> ({f['object']}) | Nguồn: {f['source']} ({f['detail']})")
        return "\n".join(lines)
