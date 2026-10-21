# BÁO CÁO KẾT QUẢ PHÂN TÍCH VÀ SUY LUẬN ĐỒ THỊ TRI THỨC (MỤC 5.4)
**Miền bài toán:** Y tế lâm sàng - Đồ thị tri thức Bệnh đái tháo đường (Diabetes Knowledge Graph)
**Hệ quản trị CSDL:** Neo4j Property Graph (GQL / Cypher)
**Thời điểm tạo:** 2026-10-21 21:29:24

---
## 1. Tổng quan cấu trúc Đồ thị tri thức trên Neo4j
- **Tổng số Node:** `784`
- **Tổng số Relationship:** `3,551`

### Phân bố các loại Node (Node Labels):
| Nhãn Node (Label) | Số lượng | Tỷ lệ (%) |
|:---|---:|---:|
| `:Patient` | 768 | 97.96% |
| `:AgeGroup` | 4 | 0.51% |
| `:BMIBand` | 4 | 0.51% |
| `:GlucoseBand` | 3 | 0.38% |
| `:HbA1cBand` | 3 | 0.38% |
| `:Condition` | 2 | 0.26% |

### Phân bố các loại Quan hệ (Relationship Types):
| Loại Quan hệ (Type) | Số lượng | Tầng biểu diễn |
|:---|---:|:---|
| `[:HAS_AGE_GROUP]` | 768 | Lớp 1: Data Layer (Patient Links) |
| `[:HAS_BMI_BAND]` | 768 | Lớp 1: Data Layer (Patient Links) |
| `[:HAS_GLUCOSE_BAND]` | 768 | Lớp 1: Data Layer (Patient Links) |
| `[:HAS_HBA1C_BAND]` | 768 | Lớp 1: Data Layer (Patient Links) |
| `[:HAS_CONDITION]` | 473 | Lớp 1: Data Layer (Patient Links) |
| `[:INDICATES]` | 3 | Lớp 2: Knowledge Layer (ADA/WHO) |
| `[:COMORBID_WITH]` | 2 | Lớp 2: Knowledge Layer (ADA/WHO) |
| `[:INCREASES_RISK_OF]` | 1 | Lớp 2: Knowledge Layer (ADA/WHO) |

---

## 2. Kết quả 5 câu truy vấn Cypher phân tích và suy luận y khoa
### Câu 1: Tỷ lệ mắc tiểu đường theo phân nhóm chỉ số HbA1c
```cypher
MATCH (h:HbA1cBand)<-[:HAS_HBA1C_BAND]-(p:Patient)
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
WITH h.name AS hba1c_band, count(p) AS total_patients, count(c) AS diabetic_patients
RETURN hba1c_band,
       total_patients,
       diabetic_patients,
       round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
ORDER BY diabetes_rate_pct DESC
```
| Phân nhóm HbA1c | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Đánh giá lâm sàng |
|:---|---:|---:|---:|:---|
| **Diabetes range** | 205 | 137 | **66.83%** | Nguy cơ bệnh lý cao nhất (>=6.5%) |
| **Prediabetes range** | 190 | 70 | **36.84%** | Tiền tiểu đường (5.7 - 6.4%) |
| **Normal** | 373 | 61 | **16.35%** | Ngưỡng bình thường (<5.7%) |

> **Nhận xét lâm sàng:** Bệnh nhân trong dải `Diabetes range` có tỷ lệ mắc bệnh thực tế lên tới **66.83%**, gấp hơn 4 lần so với nhóm `Normal` (16.35%), xác thực độ nhạy cảm của chỉ số HbA1c theo ADA Standards of Care.

### Câu 2: Tỷ lệ mắc tiểu đường theo các phân nhóm độ tuổi
```cypher
MATCH (a:AgeGroup)<-[:HAS_AGE_GROUP]-(p:Patient)
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
WITH a.name AS age_group, count(p) AS total_patients, count(c) AS diabetic_patients
RETURN age_group,
       total_patients,
       diabetic_patients,
       round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
ORDER BY age_group
```
| Nhóm độ tuổi | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Xu hướng lứa tuổi |
|:---|---:|---:|---:|:---|
| **18-34** | 488 | 126 | **25.82%** | Giai đoạn khởi phát |
| **35-49** | 191 | 99 | **51.83%** | Nguy cơ tăng vọt đỉnh điểm |
| **50-64** | 73 | 39 | **53.42%** | Nguy cơ tăng vọt đỉnh điểm |
| **65+** | 16 | 4 | **25.00%** | Giai đoạn khởi phát |

> **Nhận xét lâm sàng:** Tỷ lệ mắc bệnh tăng gấp đôi từ lứa tuổi trẻ `18-34` (25.82%) lên mức hơn `50%` ở độ tuổi trung niên `35-49` (51.83%) và `50-64` (53.42%), phản ánh rõ tính chất tiến triển của bệnh đái tháo đường type 2 theo thời gian tích lũy đề kháng insulin.

### Câu 3: Tỷ lệ mắc tiểu đường theo phân nhóm chỉ số khối cơ thể (BMI)
```cypher
MATCH (b:BMIBand)<-[:HAS_BMI_BAND]-(p:Patient)
OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition {name: 'Diabetes'})
WITH b.name AS bmi_band, count(p) AS total_patients, count(c) AS diabetic_patients
RETURN bmi_band,
       total_patients,
       diabetic_patients,
       round(100.0 * diabetic_patients / total_patients, 2) AS diabetes_rate_pct
ORDER BY diabetes_rate_pct DESC
```
| Phân nhóm BMI | Tổng số bệnh nhân | Số ca tiểu đường | Tỷ lệ mắc bệnh (%) | Phân loại WHO |
|:---|---:|---:|---:|:---|
| **Obese** | 472 | 219 | **46.40%** | WHO Standard |
| **Overweight** | 179 | 40 | **22.35%** | WHO Standard |
| **Underweight** | 15 | 2 | **13.33%** | WHO Standard |
| **Normal** | 102 | 7 | **6.86%** | WHO Standard |

> **Nhận xét lâm sàng:** Nhóm bệnh nhân béo phì (`Obese`, BMI >= 30) chiếm tỷ lệ mắc bệnh áp đảo **46.40%**, cao gấp gần 7 lần so với nhóm có thể trạng bình thường (`Normal`, 6.86%). Kết quả này củng cố chặt chẽ cạnh tri thức `(:BMIBand {name: 'Obese'})-[:INCREASES_RISK_OF]->(:Condition {name: 'Diabetes'})` của WHO.

### Câu 4: Phân tích đồng mắc (Comorbidity) giữa Tăng huyết áp và Tiểu đường
```cypher
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
```
| Trạng thái bệnh lý | Số lượng bệnh nhân | Tỷ lệ trên toàn bộ mẫu (%) |
|:---|---:|---:|
| **Đồng mắc cả hai (Diabetes + Hypertension)** | **90** | **11.72%** |
| Chỉ mắc Tiểu đường (Diabetes only) | 178 | 23.18% |
| Chỉ mắc Tăng huyết áp (Hypertension only) | 115 | 14.97% |
| Không mắc bệnh lý nào (Neither) | 385 | 50.13% |
| **Tổng cộng mẫu khảo sát** | **768** | **100.00%** |

> **Nhận xét lâm sàng:** Trong số 268 bệnh nhân tiểu đường, có tới **90 ca (33.58%)** bị đồng mắc tăng huyết áp. Đây là minh chứng cụ thể cho quan hệ song phương `COMORBID_WITH` trong hội chứng chuyển hóa tim mạch.

### Câu 5: Suy luận đa bước (Multi-hop Reasoning) phát hiện ca nguy cơ cao chưa được chẩn đoán
```cypher
MATCH (p:Patient)-[:HAS_BMI_BAND]->(b:BMIBand {name: 'Obese'})
MATCH (p)-[:HAS_HBA1C_BAND]->(h:HbA1cBand)-[r:INDICATES]->(c:Condition {name: 'Diabetes'})
WHERE NOT (p)-[:HAS_CONDITION]->(c)
RETURN count(p) AS undiagnosed_high_risk_patients,
       collect(p.id)[0..10] AS sample_patient_ids,
       r.source AS evidence_source,
       r.threshold AS clinical_threshold
```
| Tiêu chí suy luận | Kết quả từ Đồ thị tri thức |
|:---|:---|
| **Số bệnh nhân nguy cơ cao bị bỏ sót nhãn** | **38 bệnh nhân** |
| Chuỗi suy luận đồ thị (Reasoning Path) | `(:Patient)-[:HAS_BMI_BAND]->(Obese)` VÀ `(:Patient)-[:HAS_HBA1C_BAND]->(Diabetes range)-[:INDICATES]->(Diabetes)` |
| Nguồn bằng chứng chuẩn y khoa | `ADA Standards of Care` (Ngưỡng: `HbA1c >= 6.5%`) |
| Danh sách 10 mã bệnh nhân mẫu | `P_0041, P_0055, P_0059, P_0096, P_0145, P_0149, P_0154, P_0167, P_0179, P_0212` |

> **Ý nghĩa hệ thống thông minh:** Truy vấn đa bước đã chứng minh sức mạnh của Knowledge Graph so với mô hình máy học thuần túy: Hệ thống không bị giới hạn bởi nhãn dữ liệu gốc (Outcome=0) mà có thể kết nối các chỉ số lâm sàng rời rạc với tri thức chuẩn ADA để **phát hiện ra 38 ca tiền lâm sàng hoặc có thể bị sót chẩn đoán**, đưa ra cảnh báo sớm giúp bác sĩ can thiệp kịp thời.
