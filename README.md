# Instamart Customer Intelligence & Support Copilot

## Overview
Capstone project on synthetic, fictional Instamart data (all amounts in INR). It has four parts: A cleans the raw exports and stores them in SQLite, B segments customers and predicts lapse, C triages support tickets with guardrails, D builds a policy assistant and a refund agent. `generate_data.py` and `mock_llm.py` are supplied and unmodified. No paid service or API key is used.

## Reproduce
1. `python -m venv .venv`
2. `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python src/part_a_clean.py`
6. `python src/validate.py`
7. `python src/run_sql.py`
8. `python src/part_a_pandas.py`

## Part A

### Audit of raw data
| Check | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| pending rows | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| raw city spellings | 24 |
| raw gender spellings (non-blank) | 10 |
| blank gender | 30 |

Missing values: customers.age 61 (5.08%), customers.gender 30 (2.50%), orders.delivery_minutes 2,541 (19.42%). All other columns have none.

### Cleaning rules
**Customers**
1. City: strip spaces and title-case, leaving 6 cities.
2. Gender: mapped to male, female or unknown (blank becomes unknown).
3. Age: values outside 10-100 (6) set to missing, then all missing ages filled with the city median (groupby + transform).
4. signup_date converted to datetime.

**Orders** (in this order)
1. order_ts converted to datetime.
2. Exact duplicates removed: 119 rows.
3. For each order_id, kept the record with the latest order_ts: 656 older (pending) rows removed, leaving 12,309 rows.
4. order_amount converted to numeric; 112 unparseable values (in the rows that remained) filled with round(price_inr x quantity x (1 - discount_pct/100), 2).
5. Negative delivery_minutes set to missing: 105 values. Missing delivery_minutes of completed orders filled with the product-category median of completed orders: 774 values. Cancelled orders left missing (1,199), because a cancelled order was never delivered.

### Result and validation
12,309 orders (11,110 completed, 1,199 cancelled); completed revenue INR 2,998,668.80. Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders: female 610, male 560, unknown 30. Ages 18 to 60, none missing. `src/validate.py` checks 23 checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL queries (`sql/queries.sql`)
| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average spend of INR 2,552.06 |
| Q6 window ranking | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 running total | 183 daily rows, last cumulative INR 2,998,668.80 |

RANK differs from ROW_NUMBER for 8 rows because customers with exactly equal spend in a city share a rank in RANK, while ROW_NUMBER always gives them different numbers.

### pandas cross-check
The pivot table (margins=True) grand total is INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory analysis of completed order_amount
Mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, lower fence -283.50 (0 outliers), upper fence 645.30 (897 outliers above).

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

Largest absolute correlation: order_amount and discount_pct (-0.095).

**Answers (max 200 words):**

This is six months (April to September 2025) of synthetic Instamart orders: 12,309 orders from 1,200 customers across six cities and 40 products in five categories. Order amounts are strongly right-skewed. The mean (INR 269.91) is about twice the median (INR 135.00) and the skewness is 4.61, because a few expensive items such as electronics pull the average up. Q1 is 64.80 and Q3 is 297.00. There are 897 outliers above the upper fence of 645.30 and none below it. The raw data also had anomalies: duplicate rows, pending records, negative delivery times, unparseable amounts and impossible ages. Among order_amount, discount_pct, quantity and delivery_minutes, the pair with the largest absolute correlation is order_amount and discount_pct at -0.095, just ahead of order_amount and quantity at +0.087. Both are very weak, so the variables are almost independent, and a correlation of this size does not establish causation.

## Part B
TODO

## Part C
TODO

## Part D
TODO

## Limitations and Next Steps
TODO