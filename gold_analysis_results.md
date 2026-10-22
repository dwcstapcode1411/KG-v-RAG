# BÁO CÁO KẾT QUẢ PHÂN TÍCH VÀ SUY LUẬN ĐỒ THỊ TRI THỨC (MỤC 5.6)
**Miền bài toán:** Thị trường hàng hóa & Dự báo phân tích Giá Vàng (Gold Market Knowledge Graph)
**Hệ quản trị CSDL:** Neo4j Property Graph (GQL / Cypher)
**Thời điểm tạo:** 2026-10-22 10:30:28

---
## 1. Tổng quan cấu trúc Đồ thị tri thức Thị trường Vàng trên Neo4j
- **Tổng số Node:** `2,355`
- **Tổng số Relationship:** `8,950`

### Phân bố các loại Node (Node Labels):
| Nhãn Node (Label) | Số lượng | Tỷ lệ (%) | Ý nghĩa biểu diễn |
|:---|---:|---:|:---|
| `:GoldDay` | 1,259 | 53.46% | Phiên giao dịch theo ngày |
| `:Patient` | 768 | 32.61% | Sự kiện địa chính trị/lãi suất |
| `:YearStat` | 260 | 11.04% | Sự kiện địa chính trị/lãi suất |
| `:Country` | 10 | 0.42% | Sự kiện địa chính trị/lãi suất |
| `:StockIndex` | 10 | 0.42% | Sự kiện địa chính trị/lãi suất |
| `:Region` | 6 | 0.25% | Sự kiện địa chính trị/lãi suất |
| `:MarketEvent` | 6 | 0.25% | Sự kiện địa chính trị/lãi suất |
| `:Factor` | 5 | 0.21% | Yếu tố kinh tế vĩ mô |
| `:GoldEvent` | 5 | 0.21% | Sự kiện địa chính trị/lãi suất |
| `:AgeGroup` | 4 | 0.17% | Sự kiện địa chính trị/lãi suất |
| `:BMIBand` | 4 | 0.17% | Sự kiện địa chính trị/lãi suất |
| `:HbA1cBand` | 3 | 0.13% | Sự kiện địa chính trị/lãi suất |
| `:GlucoseBand` | 3 | 0.13% | Sự kiện địa chính trị/lãi suất |
| `:RSIState` | 3 | 0.13% | Trạng thái RSI |
| `:BandState` | 3 | 0.13% | Vị thế dải Bollinger |
| `:TrendState` | 3 | 0.13% | Xu hướng kỹ thuật |
| `:Condition` | 2 | 0.08% | Sự kiện địa chính trị/lãi suất |
| `:Commodity` | 1 | 0.04% | Hàng hóa kim loại quý |

### Phân bố các loại Quan hệ (Relationship Types):
| Loại Quan hệ (Type) | Số lượng | Tầng biểu diễn | Ý nghĩa liên kết |
|:---|---:|:---|:---|
| `[:HAS_RSI_STATE]` | 1,259 | Lớp 1: Time Series & Technical Layer | Liên kết chỉ báo RSI |
| `[:HAS_BAND_POSITION]` | 1,259 | Lớp 1: Time Series & Technical Layer | Liên kết dải Bollinger |
| `[:HAS_TREND]` | 1,259 | Lớp 1: Time Series & Technical Layer | Liên kết xu hướng SMA |
| `[:NEXT]` | 1,258 | Lớp 1: Time Series & Technical Layer | Chuỗi thời gian liên tiếp |
| `[:HAS_AGE_GROUP]` | 768 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:HAS_HBA1C_BAND]` | 768 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:HAS_BMI_BAND]` | 768 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:HAS_GLUCOSE_BAND]` | 768 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:HAS_CONDITION]` | 473 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:HAS_YEAR_STAT]` | 260 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:AFFECTED_BY]` | 38 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:CORRELATED_WITH]` | 36 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:IN_REGION]` | 10 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:LISTED_IN]` | 10 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:INFLUENCES]` | 5 | Lớp 2: Macro & Knowledge Layer | Tác động vĩ mô lên vàng |
| `[:OCCURRED_ON]` | 5 | Lớp 2: Macro & Knowledge Layer | Sự kiện gắn với phiên giao dịch |
| `[:INDICATES]` | 3 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:COMORBID_WITH]` | 2 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |
| `[:INCREASES_RISK_OF]` | 1 | Lớp 1: Time Series & Technical Layer | Sự kiện gắn với phiên giao dịch |

---

## 2. Kết quả 5 câu truy vấn Cypher phân tích kỹ thuật và suy luận vĩ mô
### Câu 1: Truy xuất các yếu tố vĩ mô và chiều tác động đối với giá vàng (INFLUENCES)
```cypher
MATCH (f:Factor)-[r:INFLUENCES]->(c:Commodity {name: 'Gold'})
        RETURN f.name AS factor,
               r.direction AS direction,
               r.confidence AS confidence,
               f.description AS description
        ORDER BY r.direction DESC, f.name ASC
```
| Yếu tố vĩ mô (Factor) | Chiều tác động (Direction) | Độ tin cậy (Confidence) | Cơ chế tác động kinh tế học |
|:---|:---:|:---:|:---|
| **Banking Instability / Liquidity Crisis** | **Tích cực (Dương)** | 94% | Bất ổn hệ thống ngân hàng thương mại kích hoạt tâm lý hoảng loạn và tháo chạy vốn vào kim loại quý nhằm tránh rủi ro đối tác. |
| **Central Bank Gold Accumulation** | **Tích cực (Dương)** | 92% | Các ngân hàng trung ương toàn cầu (PBoC, RBI...) đẩy mạnh đa dạng hóa dự trữ ngoại hối tạo lực cầu mua nền tảng vững chắc. |
| **Geopolitical Conflicts / Warfare** | **Tích cực (Dương)** | 98% | Xung đột địa chính trị và rủi ro chiến sự thúc đẩy mạnh mẽ dòng vốn phòng vệ vào tài sản trú ẩn an toàn (Safe-haven demand). |
| **Global Inflation Hedging** | **Tích cực (Dương)** | 90% | Lạm phát cao bào mòn sức mua của tiền pháp định, thúc đẩy nhu cầu tích trữ vàng vật chất để bảo toàn giá trị vốn dài hạn. |
| **Fed Interest Rates / US Dollar Strength** | **Tiêu cực (Âm)** | 95% | Lãi suất Fed tăng làm tăng chi phí cơ hội nắm giữ vàng không sinh lãi, củng cố đồng USD và tạo áp lực giảm giá vàng. |

> **Nhận xét kinh tế học:** Lãi suất USD tăng tạo ra lực cản tiêu cực lớn nhất đối với giá vàng do chi phí cơ hội nắm giữ tăng cao. Ngược lại, xung đột địa chính trị và bất ổn ngân hàng kích hoạt dòng tiền trú ẩn an toàn cực mạnh đẩy giá vàng bứt phá.

### Câu 2: Thống kê số ngày và lợi suất ngày trung bình theo từng trạng thái xu hướng (TrendState)
```cypher
MATCH (t:TrendState)<-[:HAS_TREND]-(d:GoldDay)
        RETURN t.name AS trend_state,
               count(d) AS num_days,
               round(avg(d.daily_return) * 100, 3) AS avg_daily_return_pct,
               round(avg(d.close), 2) AS avg_close_price
        ORDER BY avg_daily_return_pct DESC
```
| Trạng thái xu hướng | Số phiên giao dịch | Tỷ lệ phiên (%) | Lợi suất ngày TB (%) | Giá đóng cửa TB ($/oz) | Đánh giá kỹ thuật |
|:---|---:|---:|---:|---:|:---|
| **Uptrend** | 672 | 53.4% | **+0.251%** | $2,897.57 | Động lượng tăng vượt trội |
| **Sideways** | 265 | 21.0% | **-0.002%** | $2,194.99 | Giai đoạn tích lũy dao động hẹp |
| **Downtrend** | 322 | 25.6% | **-0.228%** | $2,795.61 | Áp lực bán chiếm ưu thế |

> **Nhận xét phân tích:** Trong giai đoạn 2021–2026, giá vàng nằm trong trạng thái `Uptrend` chiếm tới **53.4%** thời gian với lợi suất ngày trung bình **+0.251%**, chứng minh xu hướng tăng giá mang tính cấu trúc của kim loại quý trong chu kỳ lạm phát và căng thẳng địa chính trị.

### Câu 3: Duyệt chuỗi thời gian đa bước (Multi-hop Traversal): Lợi suất sau 5 phiên khi RSI ở vùng Quá bán (Oversold)
```cypher
MATCH (d:GoldDay)-[:HAS_RSI_STATE]->(:RSIState {name: 'Oversold'})
        MATCH (d)-[:NEXT*5]->(f:GoldDay)
        WITH d, f, ((f.close - d.close) / d.close) AS return_5d
        RETURN count(d) AS oversold_occurrences,
               round(avg(return_5d) * 100, 2) AS avg_5d_forward_return_pct,
               round(min(return_5d) * 100, 2) AS min_5d_return_pct,
               round(max(return_5d) * 100, 2) AS max_5d_return_pct,
               round(100.0 * count(CASE WHEN return_5d > 0 THEN 1 END) / count(d), 1) AS win_rate_pct
```
| Chỉ số thống kê kiểm thử | Kết quả từ Đồ thị tri thức chuỗi thời gian |
|:---|:---|
| **Số lần RSI chạm ngưỡng Quá bán (RSI < 30)** | **109 phiên** |
| **Lợi suất kỳ vọng sau 5 phiên giao dịch (T+5)** | **+0.66%** |
| **Tỷ lệ phiên sinh lời (Win Rate %)** | **61.5%** |
| Lợi suất thấp nhất sau 5 phiên | -4.98% |
| Lợi suất cao nhất sau 5 phiên | 6.93% |
| Đường đi truy vấn trên đồ thị (Traversal Path) | `(:GoldDay {RSI < 30})-[:NEXT*5]->(:GoldDay)` |

> **Ý nghĩa kỹ thuật:** Khả năng duyệt đồ thị theo đường đi độ dài cố định `[:NEXT*5]` cho phép kiểm thử trực tiếp chiến lược giao dịch hồi quy về giá trị trung bình (Mean-reversion). Khi RSI rớt vào vùng quá bán, xác suất giá vàng phục hồi sau 5 phiên đạt **61.5%** với lợi suất dương kỳ vọng, cung cấp tín hiệu bắt đáy định lượng tin cậy.

### Câu 4: Phản ứng biến động của giá vàng trước và sau các sự kiện địa chính trị / lãi suất (GoldEvent)
```cypher
MATCH (e:GoldEvent)-[:OCCURRED_ON]->(d:GoldDay)
        OPTIONAL MATCH (prev:GoldDay)-[:NEXT*5]->(d)
        OPTIONAL MATCH (d)-[:NEXT*5]->(next:GoldDay)
        WITH e, d,
             round(((d.close - prev.close) / prev.close) * 100, 2) AS return_pre_5d,
             round(((next.close - d.close) / d.close) * 100, 2) AS return_post_5d
        RETURN e.name AS event_name,
               e.type AS event_type,
               d.date AS event_date,
               d.close AS event_close_price,
               return_pre_5d,
               return_post_5d,
               e.description AS description
        ORDER BY d.date ASC
```
| Tên sự kiện vĩ mô | Phân loại | Ngày diễn ra | Giá đóng cửa ($/oz) | Biến động T-5 trước sự kiện (%) | Biến động T+5 sau sự kiện (%) | Tác động thị trường |
|:---|:---|:---:|---:|---:|---:|:---|
| **Russia-Ukraine War Breakout** | `Geopolitical Shock` | 2022-02-24 | $1,926.30 | +2.93% | **+0.50%** | Nga mở chiến dịch quân sự tại Ukraine kích hoạt nhu cầu trú ẩn an toàn cực đại trên toàn cầu. |
| **Fed Aggressive Rate Hike Cycle** | `Monetary Policy` | 2022-03-16 | $1,909.20 | -3.97% | **+1.47%** | Fed nâng lãi suất lần đầu tiên trong chu kỳ thắt chặt mạnh mẽ nhất 40 năm để kiềm chế lạm phát phi mã. |
| **Silicon Valley Bank (SVB) Collapse** | `Banking Crisis` | 2023-03-10 | $1,867.20 | +0.68% | **+5.69%** | Ngân hàng SVB phá sản đột ngột làm dấy lên nỗi lo sợ khủng hoảng thanh khoản hệ thống ngân hàng Mỹ. |
| **Hamas-Israel Conflict Eruption** | `Geopolitical Shock` | 2023-10-09 | $1,864.30 | +0.93% | **+3.75%** | Chiến sự bất ngờ bùng nổ tại Trung Đông kích hoạt làn sóng mua vàng phòng hộ rủi ro lan rộng chiến sự. |
| **Fed 50bps Rate Cut Pivot** | `Monetary Easing` | 2024-09-18 | $2,598.60 | +2.21% | **+3.31%** | Fed chính thức đảo chiều nới lỏng chính sách tiền tệ với mức cắt giảm lãi suất 50 điểm cơ bản, đẩy vàng lập đỉnh lịch sử. |

> **Nhận xét phân tích:** Giá vàng phản ứng cực kỳ nhạy bén với các cú sốc thanh khoản và chiến sự. Nổi bật nhất là vụ sụp đổ ngân hàng SVB (**+5.69%** sau 5 ngày), xung đột Trung Đông (**+3.75%** sau 5 ngày), và quyết định hạ lãi suất 50 điểm cơ bản của Fed năm 2024 (**+3.31%** sau 5 ngày).

### Câu 5: Tìm các phiên giao dịch đồng thời Quá mua (RSI > 70) và Vượt dải trên Bollinger Bands (Above upper band)
```cypher
MATCH (d:GoldDay)-[:HAS_RSI_STATE]->(:RSIState {name: 'Overbought'})
        MATCH (d)-[:HAS_BAND_POSITION]->(:BandState {name: 'Above upper band'})
        RETURN d.date AS date,
               d.close AS close_price,
               d.rsi AS rsi_value,
               d.bb_upper AS bb_upper,
               round(((d.close - d.bb_upper) / d.bb_upper) * 100, 2) AS pct_above_band
        ORDER BY d.date DESC
        LIMIT 10
```
| Phiên giao dịch | Giá đóng cửa ($/oz) | Chỉ số RSI (14) | Dải trên Bollinger Bands | Mức vượt dải (%) | Trạng thái kỹ thuật cực đoan |
|:---:|---:|---:|---:|---:|:---|
| **2026-08-11** | **$4,441.10** | 73.39 | $4,433.40 | **+0.17%** | Căng cứng kỹ thuật (Overextended) |
| **2026-08-10** | **$4,419.70** | 75.53 | $4,380.27 | **+0.90%** | Căng cứng kỹ thuật (Overextended) |
| **2026-08-07** | **$4,399.70** | 76.92 | $4,323.84 | **+1.75%** | Căng cứng kỹ thuật (Overextended) |
| **2026-08-06** | **$4,299.60** | 72.81 | $4,260.32 | **+0.92%** | Căng cứng kỹ thuật (Overextended) |
| **2026-08-05** | **$4,305.20** | 74.59 | $4,223.66 | **+1.93%** | Căng cứng kỹ thuật (Overextended) |
| **2026-01-28** | **$5,303.60** | 93.97 | $5,225.43 | **+1.50%** | Căng cứng kỹ thuật (Overextended) |
| **2026-01-26** | **$5,082.50** | 88.78 | $5,032.92 | **+0.99%** | Căng cứng kỹ thuật (Overextended) |
| **2026-01-23** | **$4,979.70** | 89.04 | $4,940.74 | **+0.79%** | Căng cứng kỹ thuật (Overextended) |
| **2026-01-22** | **$4,913.40** | 86.78 | $4,863.45 | **+1.03%** | Căng cứng kỹ thuật (Overextended) |
| **2026-01-21** | **$4,837.50** | 80.19 | $4,791.21 | **+0.97%** | Căng cứng kỹ thuật (Overextended) |

> **Ý nghĩa cảnh báo rủi ro:** Sự hội tụ của hai chỉ báo kỹ thuật độc lập (`RSI > 70` và `Close > BB Upper`) đánh dấu những thời điểm giá vàng rơi vào trạng thái hưng phấn tột độ (Euphoria). Đây là vùng rủi ro điều chỉnh kỹ thuật cao, cảnh báo nhà giao dịch không nên mở vị thế mua đuổi mà cần chốt lời từng phần.
