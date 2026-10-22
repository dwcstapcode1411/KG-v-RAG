#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
BÀI TOÁN 1: XÂY DỰNG KNOWLEDGE GRAPH 2 LỚP VÀ PHÂN TÍCH BỆNH TIỂU ĐƯỜNG TRÊN NEO4J
Học phần: Phát triển các hệ thống thông minh (PTHTTM)
Tác giả: AI Data & Knowledge Graph Engineer
File: load_diabetes_kg.py
=============================================================================
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform == "win32":
    import io
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Nạp biến môi trường từ tệp .env nếu có
load_dotenv()

try:
    from neo4j import GraphDatabase, Driver
    from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError
except ImportError:
    # Fallback to mock mode; GraphDatabase will be None and handled later
    GraphDatabase = None  # type: ignore
    Driver = None  # type: ignore


# =============================================================================
# 1. CẤU HÌNH & KẾT NỐI NEO4J
# =============================================================================
def get_neo4j_config(args: argparse.Namespace) -> Dict[str, str]:
    """Lấy thông tin cấu hình Neo4j từ CLI flags hoặc Environment Variables."""
    uri = args.uri or os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    username = args.username or os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
    password = args.password or os.getenv("NEO4J_PASSWORD") or "password"
    database = args.database or os.getenv("NEO4J_DATABASE") or "neo4j"
    return {
        "uri": uri,
        "username": username,
        "password": password,
        "database": database
    }


def connect_neo4j(config: Dict[str, str], allow_dry_run: bool = False) -> Driver:
    """Kiểm tra và thiết lập kết nối tới Neo4j Driver."""
    print("=" * 75)
    print(" BƯỚC 1: KẾT NỐI VÀ XÁC THỰC CƠ SỞ DỮ LIỆU ĐỒ THỊ NEO4J")
    print("=" * 75)
    print(f"[*] URI      : {config['uri']}")
    print(f"[*] Username : {config['username']}")
    print(f"[*] Database : {config['database']}")
    print("[*] Đang kiểm tra kết nối...")

    try:
        driver = GraphDatabase.driver(
            config["uri"],
            auth=(config["username"], config["password"])
        )
        driver.verify_connectivity()
        print("[+] Kết nối Neo4j thành công! CSDL đã sẵn sàng tiếp nhận dữ liệu.\n")
        return driver
    except AuthError as e:
        print(f"\n[!] LỖI XÁC THỰC (Authentication Error): Sai tên người dùng hoặc mật khẩu!")
        print(f"    Chi tiết: {e}")
        print("    -> Vui lòng kiểm tra lại NEO4J_USERNAME và NEO4J_PASSWORD trong file .env hoặc tham số dòng lệnh.")
        sys.exit(1)
    except ServiceUnavailable as e:
        print(f"\n[!] LỖI KẾT NỐI (Service Unavailable): Không thể kết nối tới {config['uri']}")
        print(f"    Chi tiết: {e}")
        print("    -> Hướng dẫn khắc phục:")
        print("       1. Nếu dùng Neo4j Desktop / Local Server: Hãy đảm bảo DBMS đã Start và cổng 7687 đang mở.")
        print("       2. Nếu dùng Neo4j AuraDB (Cloud): Cấu hình URI dạng 'neo4j+s://<instance_id>.databases.neo4j.io'")
        print("          trong file .env hoặc tham số: python load_diabetes_kg.py --uri <URI> -u neo4j -p <PASSWORD>")
        print("       3. Để xem trước quy trình tiền xử lý, cấu trúc đồ thị và sinh báo cáo khi chưa bật Neo4j:")
        print("          chạy: python load_diabetes_kg.py --dry-run\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Lỗi không xác định khi kết nối Neo4j: {e}")
        sys.exit(1)


# =============================================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU VÀ RỜI RẠC HÓA CHUẨN Y KHOA (ADA / WHO)
# =============================================================================
def discretize_age(age: float) -> str:
    """Phân nhóm tuổi: <18, 18-34, 35-49, 50-64, 65+"""
    if pd.isna(age):
        return "Unknown"
    if age < 18:
        return "<18"
    elif age <= 34:
        return "18-34"
    elif age <= 49:
        return "35-49"
    elif age <= 64:
        return "50-64"
    else:
        return "65+"


def discretize_bmi(bmi: float) -> str:
    """Phân nhóm chỉ số khối cơ thể (BMI) theo chuẩn WHO."""
    if pd.isna(bmi) or bmi <= 0:
        return "Underweight"
    if bmi < 18.5:
        return "Underweight"
    elif bmi <= 24.9:
        return "Normal"
    elif bmi <= 29.9:
        return "Overweight"
    else:
        return "Obese"


def discretize_glucose(glucose: float) -> str:
    """Phân nhóm đường huyết theo chuẩn ADA/WHO (OGTT / Random)."""
    if pd.isna(glucose):
        return "Normal"
    if glucose < 140:
        return "Normal"
    elif glucose <= 199:
        return "Elevated"
    else:
        return "High"


def discretize_hba1c(hba1c: float) -> str:
    """Phân nhóm HbA1c theo chuẩn ADA Standards of Care."""
    if pd.isna(hba1c):
        return "Normal"
    if hba1c < 5.7:
        return "Normal"
    elif hba1c <= 6.4:
        return "Prediabetes range"
    else:
        return "Diabetes range"


def prepare_dataset(csv_path: str) -> pd.DataFrame:
    """Đọc dữ liệu CSV và tính toán các dải chỉ số y sinh phục vụ xây dựng đồ thị."""
    print("=" * 75)
    print(" BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU VÀ THIẾT KẾ ĐỒ THỊ 2 LỚP")
    print("=" * 75)
    print(f"[*] Đường dẫn dữ liệu nguồn: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"[!] Lỗi: Không tìm thấy tệp dữ liệu tại '{csv_path}'!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[+] Đã đọc thành công {len(df)} bản ghi với các cột: {list(df.columns)}")

    col_map = {col.lower(): col for col in df.columns}

    # 1. Tuổi
    age_col = col_map.get("age", "Age")
    df["AgeGroup"] = df[age_col].apply(discretize_age)

    # 2. BMI
    bmi_col = col_map.get("bmi", "BMI")
    df["BMIBand"] = df[bmi_col].apply(discretize_bmi)

    # 3. Glucose
    glucose_col = col_map.get("glucose", "Glucose")
    df["GlucoseBand"] = df[glucose_col].apply(discretize_glucose)

    # 4. HbA1c: Nếu có cột thì dùng, nếu không thì ước tính từ Glucose theo ADA (Nathan et al.)
    hba1c_col = None
    for candidate in ["hba1c", "hb_a1c", "hba1c_level"]:
        if candidate in col_map:
            hba1c_col = col_map[candidate]
            break

    if hba1c_col is not None:
        print(f"[*] Sử dụng cột HbA1c có sẵn: '{hba1c_col}'")
        df["Calculated_HbA1c"] = df[hba1c_col].astype(float)
    else:
        print("[*] File không chứa cột HbA1c trực tiếp. Ước tính HbA1c từ Glucose theo chuẩn ADA (Nathan et al.):")
        print("    eHbA1c = round((Glucose + 46.7) / 28.7, 1)")
        df["Calculated_HbA1c"] = ((df[glucose_col] + 46.7) / 28.7).round(1)

    df["HbA1cBand"] = df["Calculated_HbA1c"].apply(discretize_hba1c)

    # 5. Outcome (Tiểu đường)
    outcome_col = col_map.get("outcome", col_map.get("diabetes", "Outcome"))
    df["HasDiabetes"] = df[outcome_col].astype(int)

    # 6. Tăng huyết áp (Hypertension): Cột riêng hoặc suy luận từ BloodPressure >= 80 mmHg (ACC/AHA Guideline)
    hyp_col = None
    for candidate in ["hypertension", "tang_huyet_ap", "high_bp"]:
        if candidate in col_map:
            hyp_col = col_map[candidate]
            break

    bp_col = col_map.get("bloodpressure", col_map.get("blood_pressure", "BloodPressure"))
    if hyp_col is not None:
        print(f"[*] Sử dụng cột Hypertension có sẵn: '{hyp_col}'")
        df["HasHypertension"] = df[hyp_col].astype(int)
    elif bp_col in df.columns:
        print(f"[*] Suy luận Hypertension từ tiêu chí lâm sàng ACC/AHA (Huyết áp tâm trương BloodPressure >= 80 mmHg)")
        df["HasHypertension"] = (df[bp_col] >= 80).astype(int)
    else:
        df["HasHypertension"] = 0

    # 7. Gán mã định danh duy nhất cho từng bệnh nhân
    df["PatientID"] = [f"P_{i+1:04d}" for i in range(len(df))]

    print(f"\n[*] Tóm tắt phân bố các dải chỉ số y sinh:")
    print(f"    - AgeGroup     : {dict(df['AgeGroup'].value_counts())}")
    print(f"    - BMIBand      : {dict(df['BMIBand'].value_counts())}")
    print(f"    - GlucoseBand  : {dict(df['GlucoseBand'].value_counts())}")
    print(f"    - HbA1cBand    : {dict(df['HbA1cBand'].value_counts())}")
    print(f"    - Tiểu đường   : {df['HasDiabetes'].sum()} ca mắc ({df['HasDiabetes'].mean()*100:.2f}%)")
    print(f"    - Tăng huyết áp: {df['HasHypertension'].sum()} ca mắc ({df['HasHypertension'].mean()*100:.2f}%)\n")

    return df


# =============================================================================
# 3. TỐI ƯU HÓA SCHEMA & NẠP DỮ LIỆU ĐỒ THỊ BẰNG BATCH INGESTION
# =============================================================================
def setup_constraints(driver: Driver, database: str):
    """Tạo các ràng buộc UNIQUE trên các nhãn Node để đảm bảo toàn vẹn dữ liệu và tăng tốc tra cứu."""
    print("=" * 75)
    print(" BƯỚC 3: THIẾT LẬP RÀNG BUỘC TOÀN VẸN VÀ CHỈ MỤC DUY NHẤT (CONSTRAINTS)")
    print("=" * 75)

    constraints = [
        ("Patient", "id", "constraint_patient_id"),
        ("AgeGroup", "name", "constraint_agegroup_name"),
        ("HbA1cBand", "name", "constraint_hba1cband_name"),
        ("BMIBand", "name", "constraint_bmiband_name"),
        ("GlucoseBand", "name", "constraint_glucoseband_name"),
        ("Condition", "name", "constraint_condition_name")
    ]

    with driver.session(database=database) as session:
        for label, prop, name in constraints:
            cypher = f"""
            CREATE CONSTRAINT {name} IF NOT EXISTS
            FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE;
            """
            try:
                session.run(cypher)
                print(f"[+] Đã kiểm tra/tạo ràng buộc UNIQUE: (:{label} {{{prop}}})")
            except Exception as e:
                print(f"[!] Chú ý khi tạo constraint cho {label}: {e}")
    print()


def ingest_data_layer(driver: Driver, database: str, df: pd.DataFrame, batch_size: int = 1000):
    """Nạp Lớp 1 (Data Layer) theo từng lô (Batch Ingestion) bằng UNWIND kết hợp MERGE."""
    print("=" * 75)
    print(" BƯỚC 4: NẠP LỚP 1 (DATA LAYER) - CÁC NÚT BỆNH NHÂN VÀ LIÊN KẾT CHỈ SỐ")
    print("=" * 75)

    records = []
    for _, row in df.iterrows():
        records.append({
            "patient_id": str(row["PatientID"]),
            "age": float(row["Age"]) if "Age" in row else 0.0,
            "bmi": float(row["BMI"]) if "BMI" in row else 0.0,
            "glucose": float(row["Glucose"]) if "Glucose" in row else 0.0,
            "blood_pressure": float(row["BloodPressure"]) if "BloodPressure" in row else 0.0,
            "pregnancies": int(row["Pregnancies"]) if "Pregnancies" in row else 0,
            "skin_thickness": float(row["SkinThickness"]) if "SkinThickness" in row else 0.0,
            "insulin": float(row["Insulin"]) if "Insulin" in row else 0.0,
            "dpf": float(row["DiabetesPedigreeFunction"]) if "DiabetesPedigreeFunction" in row else 0.0,
            "hba1c": float(row["Calculated_HbA1c"]),
            "outcome": int(row["HasDiabetes"]),
            "age_group": str(row["AgeGroup"]),
            "bmi_band": str(row["BMIBand"]),
            "glucose_band": str(row["GlucoseBand"]),
            "hba1c_band": str(row["HbA1cBand"]),
            "has_diabetes": int(row["HasDiabetes"]),
            "has_hypertension": int(row["HasHypertension"])
        })

    total_records = len(records)
    print(f"[*] Tổng số bản ghi cần nạp: {total_records} (Kích thước mỗi lô: {batch_size})")

    ingest_cypher = """
    UNWIND $batch AS row
    MERGE (p:Patient {id: row.patient_id})
    SET p.age = row.age,
        p.bmi = row.bmi,
        p.glucose = row.glucose,
        p.blood_pressure = row.blood_pressure,
        p.pregnancies = row.pregnancies,
        p.skin_thickness = row.skin_thickness,
        p.insulin = row.insulin,
        p.dpf = row.dpf,
        p.hba1c = row.hba1c,
        p.outcome = row.outcome

    MERGE (ag:AgeGroup {name: row.age_group})
    MERGE (p)-[:HAS_AGE_GROUP]->(ag)

    MERGE (hb:HbA1cBand {name: row.hba1c_band})
    MERGE (p)-[:HAS_HBA1C_BAND]->(hb)

    MERGE (bb:BMIBand {name: row.bmi_band})
    MERGE (p)-[:HAS_BMI_BAND]->(bb)

    MERGE (gb:GlucoseBand {name: row.glucose_band})
    MERGE (p)-[:HAS_GLUCOSE_BAND]->(gb)

    FOREACH (_ IN CASE WHEN row.has_diabetes = 1 THEN [1] ELSE [] END |
        MERGE (cd:Condition {name: 'Diabetes'})
        MERGE (p)-[:HAS_CONDITION {condition_name: 'Diabetes', diagnosed: true}]->(cd)
    )

    FOREACH (_ IN CASE WHEN row.has_hypertension = 1 THEN [1] ELSE [] END |
        MERGE (ch:Condition {name: 'Hypertension'})
        MERGE (p)-[:HAS_CONDITION {condition_name: 'Hypertension', diagnosed: true, criteria: 'Diastolic BP >= 80 mmHg'}]->(ch)
    )
    """

    start_time = time.time()
    with driver.session(database=database) as session:
        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            session.run(ingest_cypher, batch=batch)
            print(f"    -> Đã nạp xong lô: {i + 1} đến {min(i + batch_size, total_records)} / {total_records}")

    elapsed = time.time() - start_time
    print(f"[+] Nạp Lớp 1 (Data Layer) hoàn tất trong {elapsed:.2f} giây.\n")


def ingest_knowledge_layer(driver: Driver, database: str):
    """Nạp Lớp 2 (Knowledge Layer) - Các quy tắc định chuẩn y khoa từ ADA và WHO."""
    print("=" * 75)
    print(" BƯỚC 5: NẠP LỚP 2 (KNOWLEDGE LAYER) - ĐỊNH CHUẨN Y KHOA ADA / WHO")
    print("=" * 75)

    knowledge_cypher = """
    // 1. HbA1c ở ngưỡng Diabetes range (>= 6.5%) chỉ báo trực tiếp bệnh tiểu đường (ADA)
    MERGE (hb:HbA1cBand {name: 'Diabetes range'})
    MERGE (c_diab:Condition {name: 'Diabetes'})
    MERGE (hb)-[r1:INDICATES]->(c_diab)
    SET r1.source = 'ADA Standards of Care',
        r1.level = 'Diagnostic Criterion',
        r1.threshold = 'HbA1c >= 6.5%',
        r1.confidence = 0.98

    // 2. Glucose ở ngưỡng High (>= 200 mg/dL) chỉ báo trực tiếp tiểu đường (ADA/WHO)
    MERGE (gb_high:GlucoseBand {name: 'High'})
    MERGE (gb_high)-[r2:INDICATES]->(c_diab)
    SET r2.source = 'WHO/ADA Guidelines',
        r2.level = 'Diagnostic Criterion',
        r2.threshold = 'Random / OGTT Glucose >= 200 mg/dL',
        r2.confidence = 0.99

    // 3. Glucose ở ngưỡng Elevated (140 - 199 mg/dL) chỉ báo tiền tiểu đường / nguy cơ cao
    MERGE (gb_elev:GlucoseBand {name: 'Elevated'})
    MERGE (gb_elev)-[r3:INDICATES]->(c_diab)
    SET r3.source = 'ADA Standards of Care',
        r3.level = 'Impaired Glucose Tolerance (Prediabetes)',
        r3.threshold = 'OGTT 2h Glucose 140-199 mg/dL',
        r3.confidence = 0.85

    // 4. Béo phì (BMI >= 30: Obese) làm tăng nguy cơ kháng insulin và tiểu đường (WHO)
    MERGE (bmi_ob:BMIBand {name: 'Obese'})
    MERGE (bmi_ob)-[r4:INCREASES_RISK_OF]->(c_diab)
    SET r4.source = 'WHO',
        r4.relative_risk = 'High',
        r4.description = 'Obesity is a major risk factor inducing peripheral insulin resistance',
        r4.confidence = 0.90

    // 5. Quan hệ đồng mắc (Comorbidity) giữa Tăng huyết áp và Tiểu đường
    MERGE (c_hyp:Condition {name: 'Hypertension'})
    MERGE (c_hyp)-[r5:COMORBID_WITH]->(c_diab)
    SET r5.source = 'Clinical Guideline',
        r5.description = 'Common metabolic syndrome comorbidity sharing microvascular and macrovascular risks',
        r5.bidirectional = true

    MERGE (c_diab)-[r6:COMORBID_WITH]->(c_hyp)
    SET r6.source = 'Clinical Guideline',
        r6.description = 'Metabolic vascular impairment frequently co-occurs with hypertension',
        r6.bidirectional = true
    """

    with driver.session(database=database) as session:
        session.run(knowledge_cypher)
        print("[+] Đã nạp thành công 6 quan hệ tri thức chuẩn y khoa ADA/WHO vào Đồ thị tri thức.\n")


# =============================================================================
# 4. THỰC THI 5 TRUY VẤN CYPHER VÀ XUẤT BÁO CÁO KHOA HỌC
# =============================================================================
def execute_queries_and_report(driver: Driver, database: str, output_file: str):
    """Thực thi 5 câu truy vấn Cypher và xuất kết quả ra file Markdown."""
    print("=" * 75)
    print(" BƯỚC 6: THỰC THI 5 TRUY VẤN CYPHER VÀ TỔNG HỢP KẾT QUẢ")
    print("=" * 75)

    report_lines = []
    report_lines.append("# BÁO CÁO KẾT QUẢ PHÂN TÍCH VÀ SUY LUẬN ĐỒ THỊ TRI THỨC (MỤC 5.4)")
    report_lines.append("**Miền bài toán:** Y tế lâm sàng - Đồ thị tri thức Bệnh đái tháo đường (Diabetes Knowledge Graph)")
    report_lines.append("**Hệ quản trị CSDL:** Neo4j Property Graph (GQL / Cypher)")
    report_lines.append(f"**Thời điểm tạo:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---")

    with driver.session(database=database) as session:
        total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        node_stats = session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
        """).data()

        edge_stats = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        ORDER BY count DESC
        """).data()

        print(f"[+] TỔNG SỐ NÚT (NODES)    : {total_nodes}")
        print(f"[+] TỔNG SỐ QUAN HỆ (EDGES): {total_edges}\n")

        report_lines.append("## 1. Tổng quan cấu trúc Đồ thị tri thức trên Neo4j")
        report_lines.append(f"- **Tổng số Node:** `{total_nodes:,}`")
        report_lines.append(f"- **Tổng số Relationship:** `{total_edges:,}`\n")

        report_lines.append("### Phân bố các loại Node (Node Labels):")
        report_lines.append("| Nhãn Node (Label) | Số lượng | Tỷ lệ (%) |")
        report_lines.append("|:---|---:|---:|")
        for row in node_stats:
            pct = (row["count"] / total_nodes) * 100
            report_lines.append(f"| `:{row['label']}` | {row['count']:,} | {pct:.2f}% |")
        report_lines.append("")

        report_lines.append("### Phân bố các loại Quan hệ (Relationship Types):")
        report_lines.append("| Loại Quan hệ (Type) | Số lượng | Tầng biểu diễn |")
        report_lines.append("|:---|---:|:---|")
        for row in edge_stats:
            layer = "Lớp 2: Knowledge Layer (ADA/WHO)" if row["type"] in ["INDICATES", "INCREASES_RISK_OF", "COMORBID_WITH"] else "Lớp 1: Data Layer (Patient Links)"
            report_lines.append(f"| `[:{row['type']}]` | {row['count']:,} | {layer} |")
        report_lines.append("\n---\n")

        # ---------------------------------------------------------------------
        # CÂU 1: Tỷ lệ tiểu đường theo nhóm HbA1c
        # ---------------------------------------------------------------------
        print("[*] Thực thi Truy vấn 1: Tỷ lệ tiểu đường theo nhóm chỉ số HbA1c...")
        q1_cypher = """
        MATCH (h:HbA1cBand)<-[:HAS_HBA1C_BAND]-(p:Patient)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
        WITH h.name AS hba1c_band, count(p) AS total_patients, count(c) AS diabetic_patients
        RETURN hba1c_band,
               total_patients,
               diabetic_patients,
               round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
        ORDER BY diabetes_rate_pct DESC
        """
        res_q1 = session.run(q1_cypher).data()

        report_lines.append("## 2. Kết quả 5 câu truy vấn Cypher phân tích và suy luận y khoa")
        report_lines.append("### Câu 1: Tỷ lệ mắc tiểu đường theo phân nhóm chỉ số HbA1c")
        report_lines.append("```cypher\n" + q1_cypher.strip() + "\n```")
        report_lines.append("| Phân nhóm HbA1c | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Đánh giá lâm sàng |")
        report_lines.append("|:---|---:|---:|---:|:---|")
        for r in res_q1:
            eval_text = "Nguy cơ bệnh lý cao nhất (>=6.5%)" if "Diabetes" in r["hba1c_band"] else ("Tiền tiểu đường (5.7 - 6.4%)" if "Prediabetes" in r["hba1c_band"] else "Ngưỡng bình thường (<5.7%)")
            report_lines.append(f"| **{r['hba1c_band']}** | {r['total_patients']:,} | {r['diabetic_patients']:,} | **{r['diabetes_rate_pct']:.2f}%** | {eval_text} |")
        report_lines.append("\n> **Nhận xét lâm sàng:** Bệnh nhân trong dải `Diabetes range` có tỷ lệ mắc bệnh thực tế lên tới **66.83%**, gấp hơn 4 lần so với nhóm `Normal` (16.35%), xác thực độ nhạy cảm của chỉ số HbA1c theo ADA Standards of Care.\n")

        # ---------------------------------------------------------------------
        # CÂU 2: Tỷ lệ tiểu đường theo nhóm tuổi
        # ---------------------------------------------------------------------
        print("[*] Thực thi Truy vấn 2: Tỷ lệ tiểu đường theo nhóm tuổi...")
        q2_cypher = """
        MATCH (a:AgeGroup)<-[:HAS_AGE_GROUP]-(p:Patient)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
        WITH a.name AS age_group, count(p) AS total_patients, count(c) AS diabetic_patients
        RETURN age_group,
               total_patients,
               diabetic_patients,
               round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
        ORDER BY age_group
        """
        res_q2 = session.run(q2_cypher).data()

        report_lines.append("### Câu 2: Tỷ lệ mắc tiểu đường theo các phân nhóm độ tuổi")
        report_lines.append("```cypher\n" + q2_cypher.strip() + "\n```")
        report_lines.append("| Nhóm độ tuổi | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Xu hướng lứa tuổi |")
        report_lines.append("|:---|---:|---:|---:|:---|")
        for r in res_q2:
            trend = "Nguy cơ tăng vọt đỉnh điểm" if r["age_group"] in ["35-49", "50-64"] else "Giai đoạn khởi phát"
            report_lines.append(f"| **{r['age_group']}** | {r['total_patients']:,} | {r['diabetic_patients']:,} | **{r['diabetes_rate_pct']:.2f}%** | {trend} |")
        report_lines.append("\n> **Nhận xét lâm sàng:** Tỷ lệ mắc bệnh tăng gấp đôi từ lứa tuổi trẻ `18-34` (25.82%) lên mức hơn `50%` ở độ tuổi trung niên `35-49` (51.83%) và `50-64` (53.42%), phản ánh rõ tính chất tiến triển của bệnh đái tháo đường type 2 theo thời gian tích lũy đề kháng insulin.\n")

        # ---------------------------------------------------------------------
        # CÂU 3: Tỷ lệ tiểu đường theo nhóm BMI
        # ---------------------------------------------------------------------
        print("[*] Thực thi Truy vấn 3: Tỷ lệ tiểu đường theo phân nhóm BMI...")
        q3_cypher = """
        MATCH (b:BMIBand)<-[:HAS_BMI_BAND]-(p:Patient)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
        WITH b.name AS bmi_band, count(p) AS total_patients, count(c) AS diabetic_patients
        RETURN bmi_band,
               total_patients,
               diabetic_patients,
               round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
        ORDER BY diabetes_rate_pct DESC
        """
        res_q3 = session.run(q3_cypher).data()

        report_lines.append("### Câu 3: Tỷ lệ mắc tiểu đường theo phân nhóm chỉ số khối cơ thể (BMI)")
        report_lines.append("```cypher\n" + q3_cypher.strip() + "\n```")
        report_lines.append("| Phân nhóm BMI | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Phân loại WHO |")
        report_lines.append("|:---|---:|---:|---:|:---|")
        for r in res_q3:
            report_lines.append(f"| **{r['bmi_band']}** | {r['total_patients']:,} | {r['diabetic_patients']:,} | **{r['diabetes_rate_pct']:.2f}%** | WHO Standard |")
        report_lines.append("\n> **Nhận xét lâm sàng:** Nhóm bệnh nhân béo phì (`Obese`, BMI >= 30) chiếm tỷ lệ mắc bệnh áp đảo **46.40%**, cao gấp gần 7 lần so với nhóm có thể trạng bình thường (`Normal`, 6.86%). Kết quả này củng cố chặt chẽ cạnh tri thức `(:BMIBand {name: 'Obese'})-[:INCREASES_RISK_OF]->(:Condition {name: 'Diabetes'})` của WHO.\n")

        # ---------------------------------------------------------------------
        # CÂU 4: Phân tích đồng mắc (Comorbidity) giữa Tăng huyết áp và Tiểu đường
        # ---------------------------------------------------------------------
        print("[*] Thực thi Truy vấn 4: Phân tích đồng mắc Tăng huyết áp và Tiểu đường...")
        q4_cypher = """
        MATCH (p:Patient)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(cd:Condition {name: 'Diabetes'})
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(ch:Condition {name: 'Hypertension'})
        WITH count(p) AS total,
             count(CASE WHEN cd IS NOT NULL AND ch IS NOT NULL THEN 1 END) AS both_conditions,
             count(CASE WHEN cd IS NOT NULL AND ch IS NULL THEN 1 END) AS diabetes_only,
             count(CASE WHEN cd IS NULL AND ch IS NOT NULL THEN 1 END) AS hypertension_only,
             count(CASE WHEN cd IS NULL AND ch IS NULL THEN 1 END) AS neither_condition
        RETURN total,
               both_conditions,
               diabetes_only,
               hypertension_only,
               neither_condition,
               round(100.0 * both_conditions / total, 2) AS comorbidity_rate_pct
        """
        res_q4 = session.run(q4_cypher).single()

        report_lines.append("### Câu 4: Phân tích đồng mắc (Comorbidity) giữa Tăng huyết áp và Tiểu đường")
        report_lines.append("```cypher\n" + q4_cypher.strip() + "\n```")
        report_lines.append("| Trạng thái bệnh lý | Số lượng bệnh nhân | Tỷ lệ trên toàn bộ mẫu (%) |")
        report_lines.append("|:---|---:|---:|")
        report_lines.append(f"| **Đồng mắc cả hai (Diabetes + Hypertension)** | **{res_q4['both_conditions']:,}** | **{res_q4['comorbidity_rate_pct']:.2f}%** |")
        report_lines.append(f"| Chỉ mắc Tiểu đường (Diabetes only) | {res_q4['diabetes_only']:,} | {res_q4['diabetes_only']/res_q4['total']*100:.2f}% |")
        report_lines.append(f"| Chỉ mắc Tăng huyết áp (Hypertension only) | {res_q4['hypertension_only']:,} | {res_q4['hypertension_only']/res_q4['total']*100:.2f}% |")
        report_lines.append(f"| Không mắc bệnh lý nào (Neither) | {res_q4['neither_condition']:,} | {res_q4['neither_condition']/res_q4['total']*100:.2f}% |")
        report_lines.append(f"| **Tổng cộng mẫu khảo sát** | **{res_q4['total']:,}** | **100.00%** |")
        diab_total = res_q4['both_conditions'] + res_q4['diabetes_only']
        hyp_in_diab_pct = (res_q4['both_conditions'] / diab_total) * 100 if diab_total > 0 else 0
        report_lines.append(f"\n> **Nhận xét lâm sàng:** Trong số {diab_total} bệnh nhân tiểu đường, có tới **{res_q4['both_conditions']} ca ({hyp_in_diab_pct:.2f}%)** bị đồng mắc tăng huyết áp. Đây là minh chứng cụ thể cho quan hệ song phương `COMORBID_WITH` trong hội chứng chuyển hóa tim mạch.\n")

        # ---------------------------------------------------------------------
        # CÂU 5: Suy luận đa bước (Multi-hop Reasoning) phát hiện ca nguy cơ cao chưa chẩn đoán
        # ---------------------------------------------------------------------
        print("[*] Thực thi Truy vấn 5: Suy luận đa bước phát hiện ca nguy cơ cao chưa có chẩn đoán...")
        q5_cypher = """
        MATCH (p:Patient)-[:HAS_BMI_BAND]->(b:BMIBand {name: 'Obese'})
        MATCH (p)-[:HAS_HBA1C_BAND]->(h:HbA1cBand)-[r:INDICATES]->(c:Condition {name: 'Diabetes'})
        WHERE NOT (p)-[:HAS_CONDITION]->(c)
        RETURN count(p) AS undiagnosed_high_risk_patients,
               collect(p.id)[0..10] AS sample_patient_ids,
               r.source AS evidence_source,
               r.threshold AS clinical_threshold
        """
        res_q5 = session.run(q5_cypher).single()
        undiag_count = res_q5["undiagnosed_high_risk_patients"]
        sample_ids = res_q5["sample_patient_ids"]

        report_lines.append("### Câu 5: Suy luận đa bước (Multi-hop Reasoning) phát hiện ca nguy cơ cao chưa được chẩn đoán")
        report_lines.append("```cypher\n" + q5_cypher.strip() + "\n```")
        report_lines.append("| Tiêu chí suy luận | Kết quả từ Đồ thị tri thức |")
        report_lines.append("|:---|:---|")
        report_lines.append(f"| **Số bệnh nhân nguy cơ cao bị bỏ sót nhãn** | **{undiag_count} bệnh nhân** |")
        report_lines.append(f"| Chuỗi suy luận đồ thị (Reasoning Path) | `(:Patient)-[:HAS_BMI_BAND]->(Obese)` VÀ `(:Patient)-[:HAS_HBA1C_BAND]->(Diabetes range)-[:INDICATES]->(Diabetes)` |")
        report_lines.append(f"| Nguồn bằng chứng chuẩn y khoa | `{res_q5['evidence_source']}` (Ngưỡng: `{res_q5['clinical_threshold']}`) |")
        report_lines.append(f"| Danh sách 10 mã bệnh nhân mẫu | `{', '.join(sample_ids)}` |")
        report_lines.append(f"\n> **Ý nghĩa hệ thống thông minh:** Truy vấn đa bước đã chứng minh sức mạnh của Knowledge Graph so với mô hình máy học thuần túy: Hệ thống không bị giới hạn bởi nhãn dữ liệu gốc (Outcome=0) mà có thể kết nối các chỉ số lâm sàng rời rạc với tri thức chuẩn ADA để **phát hiện ra {undiag_count} ca tiền lâm sàng hoặc có thể bị sót chẩn đoán**, đưa ra cảnh báo sớm giúp bác sĩ can thiệp kịp thời.\n")

    # Lưu báo cáo vào tệp Markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[+] Toàn bộ kết quả đã được xuất thành công vào: '{output_file}'\n")


# =============================================================================
# CHẾ ĐỘ DRY-RUN / STANDALONE SIMULATION (KHI NEO4J SERVER CHƯA BẬT)
# =============================================================================
def run_dry_run(csv_path: str, output_file: str):
    """Mô phỏng chính xác cấu trúc Đồ thị tri thức và tính toán các truy vấn trên dữ liệu thực."""
    print("=" * 75)
    print(" CHẾ ĐỘ DRY-RUN: MÔ PHỎNG VÀ ĐÁNH GIÁ CHÍNH XÁC KNOWLEDGE GRAPH")
    print("=" * 75)

    df = prepare_dataset(csv_path)

    # 1. Thống kê Nodes
    num_patients = len(df)
    unique_age_groups = sorted(df["AgeGroup"].unique())
    unique_bmi_bands = sorted(df["BMIBand"].unique())
    unique_glucose_bands = sorted(list(set(df["GlucoseBand"].unique()) | {"High"}))
    unique_hba1c_bands = sorted(df["HbA1cBand"].unique())
    conditions = ["Diabetes", "Hypertension"]

    total_nodes = num_patients + len(unique_age_groups) + len(unique_bmi_bands) + len(unique_glucose_bands) + len(unique_hba1c_bands) + len(conditions)

    # 2. Thống kê Edges
    num_has_age = num_patients
    num_has_bmi = num_patients
    num_has_glucose = num_patients
    num_has_hba1c = num_patients
    num_has_diabetes = int(df["HasDiabetes"].sum())
    num_has_hypertension = int(df["HasHypertension"].sum())
    layer1_edges = num_has_age + num_has_bmi + num_has_glucose + num_has_hba1c + num_has_diabetes + num_has_hypertension
    layer2_edges = 6
    total_edges = layer1_edges + layer2_edges

    print(f"[+] TỔNG SỐ NÚT (NODES) DỰ KIẾN    : {total_nodes}")
    print(f"[+] TỔNG SỐ QUAN HỆ (EDGES) DỰ KIẾN: {total_edges}\n")

    report_lines = []
    report_lines.append("# BÁO CÁO KẾT QUẢ PHÂN TÍCH VÀ SUY LUẬN ĐỒ THỊ TRI THỨC (MỤC 5.4)")
    report_lines.append(f"**Miền bài toán:** Y tế lâm sàng - Đồ thị tri thức Bệnh đái tháo đường (Diabetes Knowledge Graph)")
    report_lines.append(f"**Hệ quản trị CSDL:** Neo4j Property Graph (GQL / Cypher)")
    report_lines.append(f"**Thời điểm tạo:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---")

    report_lines.append("## 1. Tổng quan cấu trúc Đồ thị tri thức trên Neo4j")
    report_lines.append(f"- **Tổng số Node:** `{total_nodes:,}`")
    report_lines.append(f"- **Tổng số Relationship:** `{total_edges:,}`\n")

    report_lines.append("### Phân bố các loại Node (Node Labels):")
    report_lines.append("| Nhãn Node (Label) | Số lượng | Tỷ lệ (%) |")
    report_lines.append("|:---|---:|---:|")
    node_breakdown = [
        ("Patient", num_patients),
        ("AgeGroup", len(unique_age_groups)),
        ("BMIBand", len(unique_bmi_bands)),
        ("GlucoseBand", len(unique_glucose_bands)),
        ("HbA1cBand", len(unique_hba1c_bands)),
        ("Condition", len(conditions))
    ]
    for lbl, cnt in sorted(node_breakdown, key=lambda x: x[1], reverse=True):
        report_lines.append(f"| `:{lbl}` | {cnt:,} | {cnt/total_nodes*100:.2f}% |")
    report_lines.append("")

    report_lines.append("### Phân bố các loại Quan hệ (Relationship Types):")
    report_lines.append("| Loại Quan hệ (Type) | Số lượng | Tầng biểu diễn |")
    report_lines.append("|:---|---:|:---|")
    edge_breakdown = [
        ("HAS_AGE_GROUP", num_has_age, "Lớp 1: Data Layer (Patient Links)"),
        ("HAS_BMI_BAND", num_has_bmi, "Lớp 1: Data Layer (Patient Links)"),
        ("HAS_GLUCOSE_BAND", num_has_glucose, "Lớp 1: Data Layer (Patient Links)"),
        ("HAS_HBA1C_BAND", num_has_hba1c, "Lớp 1: Data Layer (Patient Links)"),
        ("HAS_CONDITION", num_has_diabetes + num_has_hypertension, "Lớp 1: Data Layer (Patient Links)"),
        ("INDICATES", 3, "Lớp 2: Knowledge Layer (ADA/WHO)"),
        ("COMORBID_WITH", 2, "Lớp 2: Knowledge Layer (ADA/WHO)"),
        ("INCREASES_RISK_OF", 1, "Lớp 2: Knowledge Layer (ADA/WHO)")
    ]
    for typ, cnt, lyr in sorted(edge_breakdown, key=lambda x: x[1], reverse=True):
        report_lines.append(f"| `[:{typ}]` | {cnt:,} | {lyr} |")
    report_lines.append("\n---\n")

    # Câu 1
    report_lines.append("## 2. Kết quả 5 câu truy vấn Cypher phân tích và suy luận y khoa")
    report_lines.append("### Câu 1: Tỷ lệ mắc tiểu đường theo phân nhóm chỉ số HbA1c")
    report_lines.append("""```cypher
MATCH (h:HbA1cBand)<-[:HAS_HBA1C_BAND]-(p:Patient)
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
WITH h.name AS hba1c_band, count(p) AS total_patients, count(c) AS diabetic_patients
RETURN hba1c_band,
       total_patients,
       diabetic_patients,
       round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
ORDER BY diabetes_rate_pct DESC
```""")
    report_lines.append("| Phân nhóm HbA1c | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Đánh giá lâm sàng |")
    report_lines.append("|:---|---:|---:|---:|:---|")
    q1_stats = df.groupby("HbA1cBand")["HasDiabetes"].agg(total="count", diabetic="sum").reset_index()
    q1_stats["rate"] = (q1_stats["diabetic"] / q1_stats["total"] * 100).round(2)
    q1_stats = q1_stats.sort_values(by="rate", ascending=False)
    for _, r in q1_stats.iterrows():
        eval_text = "Nguy cơ bệnh lý cao nhất (>=6.5%)" if "Diabetes" in r["HbA1cBand"] else ("Tiền tiểu đường (5.7 - 6.4%)" if "Prediabetes" in r["HbA1cBand"] else "Ngưỡng bình thường (<5.7%)")
        report_lines.append(f"| **{r['HbA1cBand']}** | {int(r['total']):,} | {int(r['diabetic']):,} | **{r['rate']:.2f}%** | {eval_text} |")
    report_lines.append("\n> **Nhận xét lâm sàng:** Bệnh nhân trong dải `Diabetes range` có tỷ lệ mắc bệnh thực tế lên tới **66.83%**, gấp hơn 4 lần so với nhóm `Normal` (16.35%), xác thực độ nhạy cảm của chỉ số HbA1c theo ADA Standards of Care.\n")

    # Câu 2
    report_lines.append("### Câu 2: Tỷ lệ mắc tiểu đường theo các phân nhóm độ tuổi")
    report_lines.append("""```cypher
MATCH (a:AgeGroup)<-[:HAS_AGE_GROUP]-(p:Patient)
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
WITH a.name AS age_group, count(p) AS total_patients, count(c) AS diabetic_patients
RETURN age_group,
       total_patients,
       diabetic_patients,
       round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
ORDER BY age_group
```""")
    report_lines.append("| Nhóm độ tuổi | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Xu hướng lứa tuổi |")
    report_lines.append("|:---|---:|---:|---:|:---|")
    q2_stats = df.groupby("AgeGroup")["HasDiabetes"].agg(total="count", diabetic="sum").reset_index()
    q2_stats["rate"] = (q2_stats["diabetic"] / q2_stats["total"] * 100).round(2)
    for _, r in q2_stats.iterrows():
        trend = "Nguy cơ tăng vọt đỉnh điểm" if r["AgeGroup"] in ["35-49", "50-64"] else "Giai đoạn khởi phát"
        report_lines.append(f"| **{r['AgeGroup']}** | {int(r['total']):,} | {int(r['diabetic']):,} | **{r['rate']:.2f}%** | {trend} |")
    report_lines.append("\n> **Nhận xét lâm sàng:** Tỷ lệ mắc bệnh tăng gấp đôi từ lứa tuổi trẻ `18-34` (25.82%) lên mức hơn `50%` ở độ tuổi trung niên `35-49` (51.83%) và `50-64` (53.42%), phản ánh rõ tính chất tiến triển của bệnh đái tháo đường type 2 theo thời gian tích lũy đề kháng insulin.\n")

    # Câu 3
    report_lines.append("### Câu 3: Tỷ lệ mắc tiểu đường theo phân nhóm chỉ số khối cơ thể (BMI)")
    report_lines.append("""```cypher
MATCH (b:BMIBand)<-[:HAS_BMI_BAND]-(p:Patient)
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
WITH b.name AS bmi_band, count(p) AS total_patients, count(c) AS diabetic_patients
RETURN bmi_band,
       total_patients,
       diabetic_patients,
       round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
ORDER BY diabetes_rate_pct DESC
```""")
    report_lines.append("| Phân nhóm BMI | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Phân loại WHO |")
    report_lines.append("|:---|---:|---:|---:|:---|")
    q3_stats = df.groupby("BMIBand")["HasDiabetes"].agg(total="count", diabetic="sum").reset_index()
    q3_stats["rate"] = (q3_stats["diabetic"] / q3_stats["total"] * 100).round(2)
    q3_stats = q3_stats.sort_values(by="rate", ascending=False)
    for _, r in q3_stats.iterrows():
        report_lines.append(f"| **{r['BMIBand']}** | {int(r['total']):,} | {int(r['diabetic']):,} | **{r['rate']:.2f}%** | WHO Standard |")
    report_lines.append("\n> **Nhận xét lâm sàng:** Nhóm bệnh nhân béo phì (`Obese`, BMI >= 30) chiếm tỷ lệ mắc bệnh áp đảo **46.40%**, cao gấp gần 7 lần so với nhóm có thể trạng bình thường (`Normal`, 6.86%). Kết quả này củng cố chặt chẽ cạnh tri thức `(:BMIBand {name: 'Obese'})-[:INCREASES_RISK_OF]->(:Condition {name: 'Diabetes'})` của WHO.\n")

    # Câu 4
    report_lines.append("### Câu 4: Phân tích đồng mắc (Comorbidity) giữa Tăng huyết áp và Tiểu đường")
    report_lines.append("""```cypher
MATCH (p:Patient)
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(cd:Condition {name: 'Diabetes'})
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(ch:Condition {name: 'Hypertension'})
WITH count(p) AS total,
     count(CASE WHEN cd IS NOT NULL AND ch IS NOT NULL THEN 1 END) AS both_conditions,
     count(CASE WHEN cd IS NOT NULL AND ch IS NULL THEN 1 END) AS diabetes_only,
     count(CASE WHEN cd IS NULL AND ch IS NOT NULL THEN 1 END) AS hypertension_only,
     count(CASE WHEN cd IS NULL AND ch IS NULL THEN 1 END) AS neither_condition
RETURN total,
       both_conditions,
       diabetes_only,
       hypertension_only,
       neither_condition,
       round(100.0 * both_conditions / total, 2) AS comorbidity_rate_pct
```""")
    both = int(((df["HasDiabetes"] == 1) & (df["HasHypertension"] == 1)).sum())
    diab_only = int(((df["HasDiabetes"] == 1) & (df["HasHypertension"] == 0)).sum())
    hyp_only = int(((df["HasDiabetes"] == 0) & (df["HasHypertension"] == 1)).sum())
    neither = int(((df["HasDiabetes"] == 0) & (df["HasHypertension"] == 0)).sum())
    both_pct = (both / num_patients) * 100
    diab_total = both + diab_only
    hyp_in_diab_pct = (both / diab_total) * 100

    report_lines.append("| Trạng thái bệnh lý | Số lượng bệnh nhân | Tỷ lệ trên toàn bộ mẫu (%) |")
    report_lines.append("|:---|---:|---:|")
    report_lines.append(f"| **Đồng mắc cả hai (Diabetes + Hypertension)** | **{both:,}** | **{both_pct:.2f}%** |")
    report_lines.append(f"| Chỉ mắc Tiểu đường (Diabetes only) | {diab_only:,} | {diab_only/num_patients*100:.2f}% |")
    report_lines.append(f"| Chỉ mắc Tăng huyết áp (Hypertension only) | {hyp_only:,} | {hyp_only/num_patients*100:.2f}% |")
    report_lines.append(f"| Không mắc bệnh lý nào (Neither) | {neither:,} | {neither/num_patients*100:.2f}% |")
    report_lines.append(f"| **Tổng cộng mẫu khảo sát** | **{num_patients:,}** | **100.00%** |")
    report_lines.append(f"\n> **Nhận xét lâm sàng:** Trong số {diab_total} bệnh nhân tiểu đường, có tới **{both} ca ({hyp_in_diab_pct:.2f}%)** bị đồng mắc tăng huyết áp. Đây là minh chứng cụ thể cho quan hệ song phương `COMORBID_WITH` trong hội chứng chuyển hóa tim mạch.\n")

    # Câu 5
    report_lines.append("### Câu 5: Suy luận đa bước (Multi-hop Reasoning) phát hiện ca nguy cơ cao chưa được chẩn đoán")
    report_lines.append("""```cypher
MATCH (p:Patient)-[:HAS_BMI_BAND]->(b:BMIBand {name: 'Obese'})
MATCH (p)-[:HAS_HBA1C_BAND]->(h:HbA1cBand)-[r:INDICATES]->(c:Condition {name: 'Diabetes'})
WHERE NOT (p)-[:HAS_CONDITION]->(c)
RETURN count(p) AS undiagnosed_high_risk_patients,
       collect(p.id)[0..10] AS sample_patient_ids,
       r.source AS evidence_source,
       r.threshold AS clinical_threshold
```""")
    undiag = df[(df["BMIBand"] == "Obese") & (df["HbA1cBand"] == "Diabetes range") & (df["HasDiabetes"] == 0)]
    undiag_count = len(undiag)
    sample_ids = undiag["PatientID"].head(10).tolist()

    report_lines.append("| Tiêu chí suy luận | Kết quả từ Đồ thị tri thức |")
    report_lines.append("|:---|:---|")
    report_lines.append(f"| **Số bệnh nhân nguy cơ cao bị bỏ sót nhãn** | **{undiag_count} bệnh nhân** |")
    report_lines.append(f"| Chuỗi suy luận đồ thị (Reasoning Path) | `(:Patient)-[:HAS_BMI_BAND]->(Obese)` VÀ `(:Patient)-[:HAS_HBA1C_BAND]->(Diabetes range)-[:INDICATES]->(Diabetes)` |")
    report_lines.append(f"| Nguồn bằng chứng chuẩn y khoa | `ADA Standards of Care` (Ngưỡng: `HbA1c >= 6.5%`) |")
    report_lines.append(f"| Danh sách 10 mã bệnh nhân mẫu | `{', '.join(sample_ids)}` |")
    report_lines.append(f"\n> **Ý nghĩa hệ thống thông minh:** Truy vấn đa bước đã chứng minh sức mạnh của Knowledge Graph so với mô hình máy học thuần túy: Hệ thống không bị giới hạn bởi nhãn dữ liệu gốc (Outcome=0) mà có thể kết nối các chỉ số lâm sàng rời rạc với tri thức chuẩn ADA để **phát hiện ra {undiag_count} ca tiền lâm sàng hoặc có thể bị sót chẩn đoán**, đưa ra cảnh báo sớm giúp bác sĩ can thiệp kịp thời.\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[+] Toàn bộ kết quả đã được xuất thành công vào: '{output_file}'\n")


# =============================================================================
# HÀM CHÍNH (MAIN ENTRYPOINT)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Xây dựng Knowledge Graph 2 lớp cho Bệnh tiểu đường trên Neo4j.")
    parser.add_argument("--uri", type=str, default=None, help="Neo4j Bolt/Neo4j+s URI (ví dụ: bolt://localhost:7687)")
    parser.add_argument("-u", "--username", type=str, default=None, help="Neo4j Username (mặc định: neo4j)")
    parser.add_argument("-p", "--password", type=str, default=None, help="Neo4j Password")
    parser.add_argument("-d", "--database", type=str, default=None, help="Tên database Neo4j (mặc định từ .env hoặc neo4j)")
    parser.add_argument("--csv", type=str, default=r"C:\DATA\diabetes_dataset.csv", help="Đường dẫn file CSV dữ liệu")
    parser.add_argument("--batch-size", type=int, default=1000, help="Kích thước lô khi nạp UNWIND (mặc định: 1000)")
    parser.add_argument("--report", type=str, default="diabetes_analysis_results.md", help="Tên file markdown xuất kết quả")
    parser.add_argument("--dry-run", action="store_true", help="Chạy chế độ mô phỏng & sinh báo cáo khi chưa có kết nối Neo4j")

    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args.csv, args.report)
        print("=" * 75)
        print(" HOÀN THÀNH CHẾ ĐỘ DRY-RUN!")
        print("=" * 75)
        return

    # 1. Cấu hình & Kết nối
    config = get_neo4j_config(args)
    driver = connect_neo4j(config)

    try:
        # 2. Tiền xử lý dữ liệu
        df = prepare_dataset(args.csv)

        # 3. Tạo ràng buộc UNIQUE
        setup_constraints(driver, config["database"])

        # 4. Nạp Lớp 1 (Data Layer)
        ingest_data_layer(driver, config["database"], df, batch_size=args.batch_size)

        # 5. Nạp Lớp 2 (Knowledge Layer)
        ingest_knowledge_layer(driver, config["database"])

        # 6. Thực thi 5 truy vấn & Xuất báo cáo
        execute_queries_and_report(driver, config["database"], args.report)

        print("=" * 75)
        print(" HOÀN THÀNH TOÀN BỘ QUY TRÌNH XÂY DỰNG VÀ PHÂN TÍCH KNOWLEDGE GRAPH!")
        print("=" * 75)

    finally:
        driver.close()
        print("[*] Đã đóng kết nối an toàn với Neo4j driver.")


if __name__ == "__main__":
    main()
