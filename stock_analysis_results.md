# BÁO CÁO KẾT QUẢ PHÂN TÍCH VÀ SUY LUẬN ĐỒ THỊ TRI THỨC (MỤC 5.5)
**Miền bài toán:** Thị trường tài chính & Chỉ số chứng khoán toàn cầu (Stock Market Knowledge Graph)
**Hệ quản trị CSDL:** Neo4j Property Graph (GQL / Cypher)
**Thời điểm tạo:** 2026-10-22 10:30:03

---
## 1. Tổng quan cấu trúc Đồ thị tri thức Chứng khoán trên Neo4j
- **Tổng số Node:** `1,076`
- **Tổng số Relationship:** `3,905`

### Phân bố các loại Node (Node Labels):
| Nhãn Node (Label) | Số lượng | Tỷ lệ (%) | Ý nghĩa biểu diễn |
|:---|---:|---:|:---|
| `:Patient` | 768 | 71.38% | Sự kiện khủng hoảng kinh tế |
| `:YearStat` | 260 | 24.16% | Thống kê chỉ số theo năm |
| `:Country` | 10 | 0.93% | Quốc gia niêm yết |
| `:StockIndex` | 10 | 0.93% | Chỉ số chứng khoán |
| `:Region` | 6 | 0.56% | Khu vực kinh tế |
| `:MarketEvent` | 6 | 0.56% | Sự kiện khủng hoảng kinh tế |
| `:AgeGroup` | 4 | 0.37% | Sự kiện khủng hoảng kinh tế |
| `:BMIBand` | 4 | 0.37% | Sự kiện khủng hoảng kinh tế |
| `:HbA1cBand` | 3 | 0.28% | Sự kiện khủng hoảng kinh tế |
| `:GlucoseBand` | 3 | 0.28% | Sự kiện khủng hoảng kinh tế |
| `:Condition` | 2 | 0.19% | Sự kiện khủng hoảng kinh tế |

### Phân bố các loại Quan hệ (Relationship Types):
| Loại Quan hệ (Type) | Số lượng | Tầng biểu diễn | Ý nghĩa liên kết |
|:---|---:|:---|:---|
| `[:HAS_AGE_GROUP]` | 768 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:HAS_HBA1C_BAND]` | 768 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:HAS_BMI_BAND]` | 768 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:HAS_GLUCOSE_BAND]` | 768 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:HAS_CONDITION]` | 473 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:HAS_YEAR_STAT]` | 260 | Lớp 1: Data Layer | Chỉ số liên kết dữ liệu theo năm |
| `[:AFFECTED_BY]` | 38 | Lớp 2: Knowledge Layer | Bị ảnh hưởng bởi khủng hoảng |
| `[:CORRELATED_WITH]` | 36 | Lớp 1: Data Layer | Tương quan cao (|corr| >= 0.6) |
| `[:IN_REGION]` | 10 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:LISTED_IN]` | 10 | Lớp 1: Data Layer | Chỉ số thuộc quốc gia |
| `[:INDICATES]` | 3 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:COMORBID_WITH]` | 2 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |
| `[:INCREASES_RISK_OF]` | 1 | Lớp 1: Data Layer | Quốc gia thuộc khu vực |

---

## 2. Kết quả 5 câu truy vấn Cypher phân tích thị trường tài chính
### Câu 1: Phân bố số lượng chỉ số chứng khoán theo khu vực địa lý (Region)
```cypher
MATCH (s:StockIndex)-[:LISTED_IN]->(c:Country)-[:IN_REGION]->(r:Region)
        RETURN r.name AS region,
               count(s) AS num_indices,
               collect(s.code) AS index_codes,
               collect(s.name) AS index_names
        ORDER BY num_indices DESC
```
| Khu vực địa lý (Region) | Số lượng chỉ số | Danh sách mã chỉ số | Tên chỉ số tiêu biểu |
|:---|---:|:---|:---|
| **Asia-Pacific** | 4 | `^N225, 000001.SS, ^NSEI, ^JKSE` | Nikkei 225, SSE Composite, NIFTY 50, IDX Composite |
| **Europe** | 2 | `^FTSE, ^GDAXI` | FTSE 100, DAX 40 |
| **North America** | 1 | `^GSPC` | S&P 500 |
| **Latin America** | 1 | `^BVSP` | Bovespa |
| **Eurasia** | 1 | `XU100.IS` | BIST 100 |
| **Middle East** | 1 | `^TASI.SR` | Tadawul All Share |

> **Nhận xét phân tích:** Khu vực Châu Á - Thái Bình Dương (`Asia-Pacific`) chiếm số lượng đông đảo nhất (4 chỉ số: Nikkei 225, SSE Composite, NIFTY 50, IDX Composite), đại diện cho các nền kinh tế đang phát triển và phát triển năng động nhất toàn cầu.

### Câu 2: Xếp hạng các chỉ số sụt giảm sâu nhất (Max Drawdown) trong Khủng hoảng tài chính toàn cầu 2008
```cypher
MATCH (m:MarketEvent {name: 'Global Financial Crisis'})<-[:AFFECTED_BY]-(y:YearStat)<-[:HAS_YEAR_STAT]-(s:StockIndex)
        RETURN s.code AS index_code,
               s.name AS index_name,
               s.region AS region,
               y.year AS year,
               round(y.ret * 100, 2) AS annual_return_pct,
               round(y.max_drawdown * 100, 2) AS max_drawdown_pct,
               round(y.vol * 100, 2) AS volatility_pct
        ORDER BY y.max_drawdown ASC
```
| Hạng | Mã chỉ số | Tên chỉ số | Khu vực | Lợi suất năm (%) | Mức sụt giảm tối đa (%) | Độ biến động quy năm (%) |
|:---:|:---|:---|:---|---:|---:|---:|
| 1 | `^NSEI` | **NIFTY 50** | Asia-Pacific | -74.81% | **-78.95%** | 64.01% |
| 2 | `000001.SS` | **SSE Composite** | Asia-Pacific | -67.75% | **-76.71%** | 71.09% |
| 3 | `^FTSE` | **FTSE 100** | Europe | -49.01% | **-71.36%** | 80.67% |
| 4 | `^GSPC` | **S&P 500** | North America | -56.93% | **-60.73%** | 48.62% |
| 5 | `^GDAXI` | **DAX 40** | Europe | -45.09% | **-56.62%** | 48.38% |
| 6 | `^N225` | **Nikkei 225** | Asia-Pacific | -9.37% | **-52.65%** | 64.28% |
| 7 | `^BVSP` | **Bovespa** | Latin America | -27.44% | **-43.35%** | 43.31% |
| 8 | `^JKSE` | **IDX Composite** | Asia-Pacific | -10.39% | **-35.68%** | 32.01% |
| 9 | `XU100.IS` | **BIST 100** | Eurasia | -23.53% | **-31.82%** | 50.08% |

> **Nhận xét phân tích:** Trong năm 2008, toàn bộ các thị trường chứng khoán lớn đều chứng kiến mức sụt giảm nghiêm trọng từ -31% đến gần -79%. Trong đó, thị trường mới nổi NIFTY 50 (Ấn Độ) và SSE Composite (Trung Quốc) chịu sụt giảm sâu nhất với Max Drawdown lần lượt là **-78.95%** và **-76.71%**, đi kèm độ biến động cực đại trên 64% - 71%.

### Câu 3: So sánh mức sụt giảm trung bình của các khu vực trong cú sốc COVID-19 (năm 2020)
```cypher
MATCH (m:MarketEvent {name: 'COVID-19 Crash'})<-[:AFFECTED_BY]-(y:YearStat)<-[:HAS_YEAR_STAT]-(s:StockIndex)-[:LISTED_IN]->(:Country)-[:IN_REGION]->(r:Region)
        RETURN r.name AS region,
               count(s) AS num_indices,
               round(avg(y.max_drawdown) * 100, 2) AS avg_drawdown_pct,
               round(avg(y.vol) * 100, 2) AS avg_volatility_pct,
               round(avg(y.ret) * 100, 2) AS avg_annual_return_pct
        ORDER BY avg_drawdown_pct ASC
```
| Khu vực địa lý (Region) | Số chỉ số | Mức sụt giảm TB (Max Drawdown %) | Độ biến động TB (%) | Lợi suất TB cả năm 2020 (%) | Đánh giá mức độ tổn thương |
|:---|---:|---:|---:|---:|:---|
| **Europe** | 2 | **-49.28%** | 48.23% | 6.17% | Tổn thương nặng nề nhất |
| **Middle East** | 1 | **-45.03%** | 32.09% | -12.25% | Tổn thương nặng nề nhất |
| **Latin America** | 1 | **-42.55%** | 46.42% | -16.32% | Chịu tác động lớn |
| **Asia-Pacific** | 4 | **-35.71%** | 43.02% | 82.66% | Chịu tác động lớn |
| **Eurasia** | 1 | **-35.68%** | 36.73% | 8.52% | Chịu tác động lớn |
| **North America** | 1 | **-29.70%** | 36.63% | 38.75% | Hồi phục nhanh và kiên cường |

> **Nhận xét phân tích:** Khu vực Châu Âu chịu cú sốc phong tỏa đầu tiên và nặng nề nhất với sụt giảm trung bình **-49.27%** (FTSE 100 và DAX 40). Tuy nhiên, nhờ các gói kích thích tài khóa chưa từng có của ngân hàng trung ương, các thị trường chứng khoán toàn cầu đã phục hồi hình chữ V ngoạn mục vào cuối năm 2020.

### Câu 4: Thống kê các năm chỉ số S&P 500 (`^GSPC`) chịu mức sụt giảm nghiêm trọng nhất lịch sử
```cypher
MATCH (s:StockIndex {code: '^GSPC'})-[:HAS_YEAR_STAT]->(y:YearStat)
        WHERE y.return_band = 'Severe Drop (<-20%)' OR y.max_drawdown <= -0.30
        OPTIONAL MATCH (y)-[:AFFECTED_BY]->(m:MarketEvent)
        RETURN y.year AS year,
               round(y.ret * 100, 2) AS annual_return_pct,
               round(y.max_drawdown * 100, 2) AS max_drawdown_pct,
               round(y.vol * 100, 2) AS volatility_pct,
               y.return_band AS return_band,
               coalesce(m.name, 'Market Volatility') AS historical_event
        ORDER BY y.max_drawdown ASC
```
| Năm | Lợi suất năm (%) | Mức sụt giảm tối đa (%) | Độ biến động (%) | Phân nhóm lợi suất | Sự kiện lịch sử gắn liền |
|:---:|---:|---:|---:|:---|:---|
| **2000** | -23.95% | **-75.59%** | 105.21% | `Severe Drop (<-20%)` | **Dot-com Bubble** |
| **2001** | -43.97% | **-72.09%** | 68.21% | `Severe Drop (<-20%)` | **Market Volatility** |
| **2002** | -38.56% | **-70.90%** | 61.85% | `Severe Drop (<-20%)` | **Market Volatility** |
| **2008** | -56.93% | **-60.73%** | 48.62% | `Severe Drop (<-20%)` | **Global Financial Crisis** |
| **2004** | -16.27% | **-47.52%** | 38.43% | `Negative (<0%)` | **Market Volatility** |
| **2011** | -16.60% | **-44.09%** | 48.81% | `Negative (<0%)` | **Market Volatility** |
| **2003** | 52.80% | **-42.67%** | 44.91% | `Positive (>0%)` | **Market Volatility** |
| **2009** | 91.36% | **-36.27%** | 41.03% | `Positive (>0%)` | **Market Volatility** |
| **2023** | -6.66% | **-36.04%** | 26.84% | `Negative (<0%)` | **Market Volatility** |
| **2025** | 2.62% | **-34.79%** | 31.66% | `Positive (>0%)` | **Market Volatility** |
| **2006** | 10.44% | **-30.81%** | 28.58% | `Positive (>0%)` | **Market Volatility** |

> **Nhận xét phân tích:** Hai cuộc sụp đổ lớn nhất của S&P 500 trong thế kỷ 21 là giai đoạn vỡ bong bóng Dot-com (2000–2002) với mức sụt giảm đỉnh điểm lên đến **-75.59%** và Khủng hoảng tài chính toàn cầu 2008 với mức sụt giảm **-60.73%**.

### Câu 5: Suy luận đa bước (Multi-hop Reasoning) phát hiện các cặp chỉ số tương quan cao cùng chịu biến cố Khủng hoảng toàn cầu 2008
```cypher
MATCH (s1:StockIndex)-[r:CORRELATED_WITH]->(s2:StockIndex)
        WHERE s1.code < s2.code
        MATCH (s1)-[:HAS_YEAR_STAT]->(y1:YearStat)-[:AFFECTED_BY]->(m:MarketEvent {name: 'Global Financial Crisis'})
        MATCH (s2)-[:HAS_YEAR_STAT]->(y2:YearStat)-[:AFFECTED_BY]->(m)
        RETURN s1.name AS index_1,
               s1.region AS region_1,
               s2.name AS index_2,
               s2.region AS region_2,
               r.corr_value AS correlation,
               round(y1.max_drawdown * 100, 2) AS mdd_index_1_pct,
               round(y2.max_drawdown * 100, 2) AS mdd_index_2_pct,
               m.name AS shared_crisis
        ORDER BY r.corr_value DESC
```
| Chỉ số 1 | Khu vực 1 | Chỉ số 2 | Khu vực 2 | Hệ số tương quan | Sụt giảm 2008 (CS 1) | Sụt giảm 2008 (CS 2) | Biến cố chung |
|:---|:---|:---|:---|:---:|---:|---:|:---|
| **SSE Composite** | Asia-Pacific | **Bovespa** | Latin America | **0.8710** | -76.71% | -43.35% | `Global Financial Crisis` |
| **S&P 500** | North America | **NIFTY 50** | Asia-Pacific | **0.8582** | -60.73% | -78.95% | `Global Financial Crisis` |
| **DAX 40** | Europe | **S&P 500** | North America | **0.8408** | -56.62% | -60.73% | `Global Financial Crisis` |
| **Bovespa** | Latin America | **DAX 40** | Europe | **0.8141** | -43.35% | -56.62% | `Global Financial Crisis` |
| **DAX 40** | Europe | **Nikkei 225** | Asia-Pacific | **0.8002** | -56.62% | -52.65% | `Global Financial Crisis` |
| **SSE Composite** | Asia-Pacific | **DAX 40** | Europe | **0.7974** | -76.71% | -56.62% | `Global Financial Crisis` |
| **BIST 100** | Eurasia | **NIFTY 50** | Asia-Pacific | **0.7958** | -31.82% | -78.95% | `Global Financial Crisis` |
| **S&P 500** | North America | **Nikkei 225** | Asia-Pacific | **0.7665** | -60.73% | -52.65% | `Global Financial Crisis` |
| **DAX 40** | Europe | **IDX Composite** | Asia-Pacific | **0.7578** | -56.62% | -35.68% | `Global Financial Crisis` |
| **Bovespa** | Latin America | **S&P 500** | North America | **0.6904** | -43.35% | -60.73% | `Global Financial Crisis` |
| **SSE Composite** | Asia-Pacific | **S&P 500** | North America | **0.6729** | -76.71% | -60.73% | `Global Financial Crisis` |
| **BIST 100** | Eurasia | **S&P 500** | North America | **0.6492** | -31.82% | -60.73% | `Global Financial Crisis` |
| **Bovespa** | Latin America | **IDX Composite** | Asia-Pacific | **0.6350** | -43.35% | -35.68% | `Global Financial Crisis` |
| **IDX Composite** | Asia-Pacific | **Nikkei 225** | Asia-Pacific | **0.6067** | -35.68% | -52.65% | `Global Financial Crisis` |

> **Ý nghĩa suy luận đồ thị:** Truy vấn liên kết đa bước: `(:StockIndex)-[:CORRELATED_WITH]->(:StockIndex)` kết hợp cùng `(:YearStat)-[:AFFECTED_BY]->(:MarketEvent)` đã khám phá hiện tượng lây lan khủng hoảng tài chính xuyên lục địa (Financial Contagion). Các cặp chỉ số có tương quan cao như S&P 500 (Mỹ) và DAX 40 (Đức) ($corr = 0.8408$) hay S&P 500 và Nikkei 225 (Nhật Bản) ($corr = 0.7665$) đều sụt giảm đồng thời hơn 50% - 60% trong năm 2008, chứng minh tính toàn cầu hóa sâu sắc của thị trường vốn.
