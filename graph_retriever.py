#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
MODULE TRUY XUẤT ĐỒ THỊ TRI THỨC (GRAPH RETRIEVER - NEO4J & MOCK LAYER)
Học phần: Phát triển các hệ thống thông minh (PTHTTM) - Mục 5.2, 5.4 - 5.7
=============================================================================
"""

import os
import re
import socket
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


# =============================================================================
# BỘ PHÂN TÍCH NGÔN NGỮ TỰ NHIÊN ĐA MIỀN (NLU ENTITY EXTRACTOR)
# =============================================================================
class NLUExtractor:
    """Bóc tách thực thể và số liệu từ câu hỏi ngôn ngữ tự nhiên tiếng Việt/Anh cho 3 miền."""

    # ── Miền Y tế ──
    @staticmethod
    def parse_diabetes(q: str) -> Dict[str, Any]:
        """Trích xuất: cân nặng, chiều cao, BMI, HbA1c, glucose, tuổi, huyết áp."""
        q_low = q.lower()
        result = {"weight": None, "height": None, "bmi": None, "bmi_band": None,
                  "hba1c": None, "glucose": None, "age": None, "bp": None,
                  "is_bmi_query": False, "bmi_comparison": None}

        # Cân nặng
        w = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:kg|kí|cân|kilô)', q_low)
        if w:
            result["weight"] = float(w.group(1).replace(',', '.'))

        # Chiều cao
        h = re.search(r'(\d)m(\d{1,2})', q_low)
        if h:
            cm = h.group(2)
            if len(cm) == 1:
                cm += '0'
            result["height"] = float(h.group(1)) + float(cm) / 100.0
        else:
            h2 = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:m\b|mét)', q_low)
            if h2:
                v = float(h2.group(1).replace(',', '.'))
                result["height"] = v if v < 3.0 else v / 100.0
            else:
                h3 = re.search(r'(\d{2,3})\s*cm', q_low)
                if h3:
                    result["height"] = float(h3.group(1)) / 100.0

        # Tính BMI từ cân nặng + chiều cao
        if result["weight"] and result["height"] and result["height"] > 0.5:
            result["bmi"] = round(result["weight"] / (result["height"] ** 2), 2)

        # BMI trực tiếp: "BMI là 28", "BMI của tôi 32"
        bmi_direct = re.search(r'bmi\s*(?:là|=|của\s*(?:tôi|em|bạn))?\s*(\d+(?:[\.,]\d+)?)', q_low)
        if bmi_direct and not result["bmi"]:
            result["bmi"] = float(bmi_direct.group(1).replace(',', '.'))

        # Phân nhóm BMI
        if result["bmi"]:
            b = result["bmi"]
            result["bmi_band"] = "Obese" if b >= 30 else ("Overweight" if b >= 25 else ("Normal" if b >= 18.5 else "Underweight"))

        # So sánh BMI: "BMI nhỏ hơn 25", "BMI dưới 30", "BMI > 30"
        bmi_comp = re.search(r'bmi\s*(?:nhỏ hơn|dưới|bé hơn|thấp hơn|<)\s*(\d+)', q_low)
        if bmi_comp:
            result["is_bmi_query"] = True
            result["bmi_comparison"] = ("lt", float(bmi_comp.group(1)))
        bmi_comp2 = re.search(r'bmi\s*(?:lớn hơn|trên|cao hơn|>=?|vượt)\s*(\d+)', q_low)
        if bmi_comp2:
            result["is_bmi_query"] = True
            result["bmi_comparison"] = ("gte", float(bmi_comp2.group(1)))

        # Từ khóa chung về BMI
        if 'bmi' in q_low or 'cân nặng' in q_low or 'béo' in q_low or 'thể trạng' in q_low:
            result["is_bmi_query"] = True

        # HbA1c: "HbA1c là 7.2", "HbA1c của tôi 6.8%"
        hba1c_m = re.search(r'hba1c\s*(?:là|=|của\s*(?:tôi|em))?\s*(\d+(?:[\.,]\d+)?)\s*%?', q_low)
        if hba1c_m:
            result["hba1c"] = float(hba1c_m.group(1).replace(',', '.'))

        # Glucose: "đường huyết 180", "glucose 145 mg/dl"
        gluc_m = re.search(r'(?:glucose|đường huyết|đường trong máu|blood sugar)\s*(?:là|=|của\s*(?:tôi|em))?\s*(\d+(?:[\.,]\d+)?)', q_low)
        if gluc_m:
            result["glucose"] = float(gluc_m.group(1).replace(',', '.'))

        # Tuổi: "tôi 45 tuổi", "tuổi 60"
        age_m = re.search(r'(?:(\d{2})\s*tuổi|tuổi\s*(?:là|của\s*(?:tôi|em))?\s*(\d{2}))', q_low)
        if age_m:
            result["age"] = int(age_m.group(1) or age_m.group(2))

        # Huyết áp: "huyết áp 140/90", "HA 130/85"
        bp_m = re.search(r'(?:huyết áp|ha|blood pressure|bp)\s*(?:là)?\s*(\d{2,3})[/\\](\d{2,3})', q_low)
        if bp_m:
            result["bp"] = (int(bp_m.group(1)), int(bp_m.group(2)))

        return result

    # ── Miền Tài chính ──
    @staticmethod
    def parse_finance(q: str) -> Dict[str, Any]:
        """Trích xuất: mã chỉ số, năm, tỷ lệ sụt giảm, vốn đầu tư, DTI, lãi suất."""
        q_low = q.lower()
        result = {"year": None, "index_name": None, "investment": None,
                  "dti": None, "interest_rate": None, "loss_pct": None,
                  "is_risk_query": False, "is_correlation_query": False}

        # Năm
        yr = re.search(r'(?:năm\s+)?(\d{4})', q_low)
        if yr and 1990 <= int(yr.group(1)) <= 2030:
            result["year"] = int(yr.group(1))

        # Mã chỉ số
        indices = {"s&p": "S&P 500", "sp500": "S&P 500", "s&p 500": "S&P 500",
                   "nifty": "NIFTY 50", "sse": "SSE Composite", "dax": "DAX 40",
                   "ftse": "FTSE 100", "nikkei": "Nikkei 225", "bovespa": "Bovespa",
                   "vnindex": "VN-Index", "vn-index": "VN-Index", "bist": "BIST 100"}
        for key, name in indices.items():
            if key in q_low:
                result["index_name"] = name
                break

        # Vốn đầu tư: "đầu tư 100 triệu", "tôi có 500 triệu đồng"
        inv = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:triệu|tỷ|tr|tỉ)', q_low)
        if inv:
            val = float(inv.group(1).replace(',', '.'))
            if 'tỷ' in q_low or 'tỉ' in q_low:
                val *= 1000
            result["investment"] = val  # triệu đồng

        # DTI
        dti = re.search(r'(?:dti|nợ trên thu nhập)\s*(?:là|=)?\s*(\d+(?:[\.,]\d+)?)\s*%?', q_low)
        if dti:
            result["dti"] = float(dti.group(1).replace(',', '.'))

        # Thua lỗ phần trăm
        loss = re.search(r'(?:lỗ|mất|giảm|sụt)\s*(\d+(?:[\.,]\d+)?)\s*%', q_low)
        if loss:
            result["loss_pct"] = float(loss.group(1).replace(',', '.'))

        if any(k in q_low for k in ['rủi ro', 'risk', 'thua lỗ', 'mất vốn', 'an toàn', 'nên đầu tư']):
            result["is_risk_query"] = True
        if any(k in q_low for k in ['tương quan', 'corr', 'liên kết', 'lây lan', 'contagion']):
            result["is_correlation_query"] = True

        return result

    # ── Miền Giá vàng ──
    @staticmethod
    def parse_gold(q: str) -> Dict[str, Any]:
        """Trích xuất: giá vàng, RSI, mức giá mua/bán, thời điểm."""
        q_low = q.lower()
        result = {"gold_price": None, "rsi_value": None, "has_event_query": False,
                  "is_buy_signal": False, "is_sell_signal": False, "is_macro_query": False}

        # Giá vàng: "vàng đang ở mức 2300 USD", "giá vàng 2500"
        gp = re.search(r'(?:giá\s*)?(?:vàng|gold)\s*(?:đang|hiện|ở|=|là|khoảng)?\s*(?:mức\s*)?(\d[\d,\.]*)\s*(?:usd|đô|dollar|\$)?', q_low)
        if gp:
            result["gold_price"] = float(gp.group(1).replace(',', ''))

        # RSI: "RSI hiện tại là 25", "RSI = 72"
        rsi = re.search(r'rsi\s*(?:hiện tại|đang|ở|là|=)?\s*(\d+(?:[\.,]\d+)?)', q_low)
        if rsi:
            result["rsi_value"] = float(rsi.group(1).replace(',', '.'))

        # Tín hiệu mua/bán
        if any(k in q_low for k in ['nên mua', 'có nên mua', 'mua vào', 'buy', 'mua vàng', 'tích trữ']):
            result["is_buy_signal"] = True
        if any(k in q_low for k in ['nên bán', 'có nên bán', 'bán ra', 'sell', 'chốt lời', 'bán vàng']):
            result["is_sell_signal"] = True

        # Sự kiện
        if any(k in q_low for k in ['svb', 'silicon valley', 'hamas', 'israel', 'ukraine', 'nga',
                                      'chiến tranh', 'chiến sự', 'xung đột', 'fed cắt', 'hạ lãi suất',
                                      'sự kiện', 'khủng hoảng', 'sụp đổ', 'ngân hàng']):
            result["has_event_query"] = True

        if any(k in q_low for k in ['vĩ mô', 'lãi suất', 'fed', 'usd', 'đô la', 'lạm phát',
                                      'ngân hàng trung ương', 'dự trữ', 'yếu tố']):
            result["is_macro_query"] = True

        return result


# =============================================================================
# LỚP CHÍNH: GRAPH RETRIEVER
# =============================================================================
class GraphRetriever:
    """Bộ truy xuất sự kiện Đồ thị tri thức (Knowledge Graph Facts) từ Neo4j hoặc Mock Model."""

    # Bảng tra cứu tỷ lệ mắc bệnh thực tế theo BMI (từ dữ liệu Neo4j)
    BMI_RISK_TABLE = {
        "Obese":      {"rate": "46.40%", "count": "219/472", "ratio": "gấp 6.76 lần nhóm Normal"},
        "Overweight":  {"rate": "22.35%", "count": "40/179",  "ratio": "gấp 3.26 lần nhóm Normal"},
        "Normal":      {"rate": "6.86%",  "count": "7/102",   "ratio": "nguy cơ thấp nhất"},
        "Underweight": {"rate": "13.33%", "count": "2/15",    "ratio": "nguy cơ do suy dinh dưỡng"}
    }
    HBCA1C_RISK_TABLE = {
        "Diabetes range":    {"rate": "66.83%", "count": "137/205", "threshold": ">= 6.5%"},
        "Prediabetes range": {"rate": "36.84%", "count": "70/190",  "threshold": "5.7% - 6.4%"},
        "Normal":            {"rate": "16.35%", "count": "61/373",  "threshold": "< 5.7%"}
    }
    AGE_RISK_TABLE = {
        "18-34": {"rate": "25.82%"}, "35-49": {"rate": "51.83%"},
        "50-64": {"rate": "53.42%"}, "65+":   {"rate": "25.00%"}
    }

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

        try:
            parsed = urlparse(self.uri.replace("bolt://", "http://").replace("neo4j+s://", "https://").replace("neo4j://", "http://"))
            host = parsed.hostname or "localhost"
            port = parsed.port or 7687

            if host in ["localhost", "127.0.0.1", "::1"]:
                try:
                    s = socket.create_connection((host, port), timeout=0.3)
                    s.close()
                except Exception:
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

    # ── Alias tĩnh cho tương thích ngược ──
    @staticmethod
    def parse_clinical_measurements(q: str) -> Dict[str, Any]:
        """Alias tương thích ngược cho NLUExtractor.parse_diabetes."""
        r = NLUExtractor.parse_diabetes(q)
        return {"weight": r["weight"], "height": r["height"], "bmi": r["bmi"],
                "is_normal": r.get("bmi_comparison") == ("lt", 25) or
                             (r["bmi_band"] == "Normal" if r["bmi_band"] else False) or
                             r.get("is_bmi_query", False) and r["bmi_comparison"] and r["bmi_comparison"][0] == "lt",
                "is_obese": r["bmi_band"] == "Obese" if r["bmi_band"] else False}

    # ── Suy luận Fact thông minh theo miền ──
    def _build_clinical_facts(self, q: str, domain: str) -> List[Dict[str, Any]]:
        """Sinh các Graph Facts có suy luận từ dữ liệu cá nhân người dùng nhập."""
        facts = []

        if domain == "diabetes":
            p = NLUExtractor.parse_diabetes(q)

            # BMI cá nhân
            if p["bmi"] and p["bmi_band"]:
                info = self.BMI_RISK_TABLE[p["bmi_band"]]
                facts.append({
                    "subject": f"Bệnh nhân (Cân nặng {p['weight']}kg, Chiều cao {p['height']}m → BMI = {p['bmi']})",
                    "relation": "CLASSIFIED_AS",
                    "object": f"BMIBand ({p['bmi_band']})",
                    "source": "WHO Clinical Calculator & Neo4j",
                    "detail": f"BMI = {p['weight']} / ({p['height']}²) = {p['bmi']}"
                })
                facts.append({
                    "subject": f"BMIBand ({p['bmi_band']})",
                    "relation": "OBSERVED_DIABETES_RATE",
                    "object": f"Tỷ lệ mắc bệnh thực tế: {info['rate']} ({info['count']} ca)",
                    "source": "Neo4j Clinical Graph (768 bệnh nhân)",
                    "detail": f"{info['ratio']}"
                })

            # So sánh BMI: "BMI nhỏ hơn 25"
            if p["bmi_comparison"] and not p["bmi"]:
                op, threshold = p["bmi_comparison"]
                if op == "lt" and threshold <= 25:
                    band, info = "Normal", self.BMI_RISK_TABLE["Normal"]
                elif op == "lt" and threshold <= 30:
                    band, info = "Overweight", self.BMI_RISK_TABLE["Overweight"]
                elif op == "gte" and threshold >= 30:
                    band, info = "Obese", self.BMI_RISK_TABLE["Obese"]
                else:
                    band, info = "Normal", self.BMI_RISK_TABLE["Normal"]
                facts.append({
                    "subject": f"BMIBand ({band}: BMI {'<' if op == 'lt' else '>='} {threshold})",
                    "relation": "OBSERVED_DIABETES_RATE",
                    "object": f"Tỷ lệ mắc bệnh: {info['rate']} ({info['count']} ca)",
                    "source": "Neo4j Clinical Graph",
                    "detail": f"Phân nhóm WHO chuẩn trên đồ thị – {info['ratio']}"
                })

            # HbA1c cá nhân
            if p["hba1c"]:
                v = p["hba1c"]
                band = "Diabetes range" if v >= 6.5 else ("Prediabetes range" if v >= 5.7 else "Normal")
                info = self.HBCA1C_RISK_TABLE[band]
                facts.append({
                    "subject": f"Chỉ số HbA1c cá nhân = {v}%",
                    "relation": "CLASSIFIED_AS",
                    "object": f"HbA1cBand ({band})",
                    "source": "ADA Standards of Care & Neo4j",
                    "detail": f"Ngưỡng {info['threshold']} – Tỷ lệ mắc bệnh thực tế {info['rate']}"
                })

            # Glucose cá nhân
            if p["glucose"]:
                v = p["glucose"]
                band = "High" if v >= 200 else ("Elevated" if v >= 140 else "Normal")
                desc = "Chẩn đoán xác định (>= 200 mg/dL)" if v >= 200 else ("Rối loạn dung nạp (140-199 mg/dL)" if v >= 140 else "Bình thường (< 140 mg/dL)")
                facts.append({
                    "subject": f"Glucose huyết = {v} mg/dL",
                    "relation": "CLASSIFIED_AS",
                    "object": f"GlucoseBand ({band})",
                    "source": "ADA/WHO Guidelines & Neo4j",
                    "detail": desc
                })

            # Tuổi cá nhân
            if p["age"]:
                a = p["age"]
                band = "18-34" if a < 35 else ("35-49" if a < 50 else ("50-64" if a < 65 else "65+"))
                info = self.AGE_RISK_TABLE[band]
                facts.append({
                    "subject": f"Tuổi bệnh nhân = {a}",
                    "relation": "BELONGS_TO",
                    "object": f"AgeGroup ({band})",
                    "source": "Neo4j Clinical Graph",
                    "detail": f"Tỷ lệ mắc tiểu đường nhóm tuổi này: {info['rate']}"
                })

            # Huyết áp
            if p["bp"]:
                sys, dia = p["bp"]
                has_hyp = sys >= 130 or dia >= 80
                facts.append({
                    "subject": f"Huyết áp = {sys}/{dia} mmHg",
                    "relation": "INDICATES" if has_hyp else "WITHIN_NORMAL",
                    "object": "Condition (Hypertension)" if has_hyp else "Huyết áp bình thường",
                    "source": "ACC/AHA Guidelines",
                    "detail": f"{'Tăng huyết áp (≥130/80)! 33.58% BN tiểu đường đồng mắc THA' if has_hyp else 'Trong ngưỡng kiểm soát tốt'}"
                })

        elif domain == "finance":
            p = NLUExtractor.parse_finance(q)
            if p["investment"] and p["loss_pct"]:
                loss_amount = p["investment"] * p["loss_pct"] / 100
                facts.append({
                    "subject": f"Vốn đầu tư {p['investment']:.0f} triệu",
                    "relation": "SIMULATED_LOSS",
                    "object": f"Mất {loss_amount:.1f} triệu ({p['loss_pct']}%)",
                    "source": "Risk Simulation",
                    "detail": f"Nếu chịu mức sụt giảm {p['loss_pct']}% (tương đương khủng hoảng thị trường)"
                })
            if p["dti"]:
                level = "Rủi ro CAO – Nguy cơ vỡ nợ" if p["dti"] > 45 else ("Cảnh báo – Cần cân nhắc" if p["dti"] > 36 else "An toàn")
                facts.append({
                    "subject": f"Tỷ lệ nợ/thu nhập (DTI) = {p['dti']}%",
                    "relation": "RISK_ASSESSMENT",
                    "object": level,
                    "source": "Credit Policy [FIN-CREDIT-03]",
                    "detail": f"Ngưỡng an toàn ≤ 36%, ngưỡng nguy hiểm > 45%"
                })

        elif domain == "gold":
            p = NLUExtractor.parse_gold(q)
            if p["rsi_value"]:
                v = p["rsi_value"]
                if v < 30:
                    facts.append({
                        "subject": f"RSI hiện tại = {v} (Vùng Quá bán / Oversold)",
                        "relation": "PREDICTS_MEAN_REVERSION",
                        "object": "Kỳ vọng phục hồi +0.66% sau 5 phiên (Win Rate: 61.5%)",
                        "source": "Graph Multi-hop Traversal [:NEXT*5]",
                        "detail": f"RSI={v} < 30 kích hoạt tín hiệu hồi quy trung bình trên 109 phiên lịch sử"
                    })
                elif v > 70:
                    facts.append({
                        "subject": f"RSI hiện tại = {v} (Vùng Quá mua / Overbought)",
                        "relation": "SIGNALS_CAUTION",
                        "object": "Cảnh báo rủi ro điều chỉnh kỹ thuật ngắn hạn cao",
                        "source": "Technical Indicators [TECH-GOLD-04]",
                        "detail": f"RSI={v} > 70 cho thấy thị trường hưng phấn quá đà, nên chờ RSI hạ nhiệt"
                    })
                else:
                    facts.append({
                        "subject": f"RSI hiện tại = {v} (Vùng Trung tính / Neutral)",
                        "relation": "NEUTRAL_SIGNAL",
                        "object": "Không có tín hiệu cực đoan",
                        "source": "Technical Indicators [TECH-GOLD-04]",
                        "detail": f"RSI trong khoảng 30-70 – Thị trường ở trạng thái cân bằng"
                    })

        return facts

    def retrieve_facts(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """
        Truy xuất các sự kiện tri thức đồ thị (Triples/Paths) phù hợp với câu hỏi và miền nghiệp vụ.
        Nếu có Neo4j sống, truy vấn trực tiếp; nếu không, sử dụng cơ sở tri thức đồ thị chuẩn hóa.
        """
        query_lower = query.lower()

        # 1. Suy luận cá nhân từ NLU
        clinical_facts = self._build_clinical_facts(query_lower, domain)

        # 2. Truy vấn đồ thị tri thức (Neo4j hoặc Mock)
        graph_facts = []
        if self.is_connected and self.driver:
            try:
                graph_facts = self._query_neo4j(query_lower, domain)
            except Exception:
                pass

        if not graph_facts:
            graph_facts = self._query_mock_graph(query_lower, domain)

        # 3. Kết hợp: Facts cá nhân luôn nằm đầu, sau đó đến facts đồ thị
        combined = clinical_facts + [f for f in graph_facts if f not in clinical_facts]
        return combined[:8]

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
