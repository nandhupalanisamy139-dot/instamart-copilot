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
9. `python src/part_b_segments.py`
10. `python src/part_b_lapse.py`
11. `python src/part_c_triage.py`

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

### RFM (snapshot 2025-10-01)
1,175 customers with a completed order. Recency median 19 days (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

### K-means (StandardScaler, n_init=10, random_state=42)
| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** Inertia drops sharply from K=3 to K=4 (1268 to 743) and much more slowly afterwards, so the elbow is at 4, and K=4 also has the best silhouette (0.526). Four segments are also a number the retention team can run campaigns for. The inertia of K=8 is always lower than K=2 because more centres always bring every point closer to a centre (with K equal to the number of customers it would be zero), so inertia alone would always pick the largest K; the elbow and silhouette are needed to judge whether the extra clusters are meaningful.

### Segments (cluster numbers are only labels with no ordering)
| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium Big Spenders (is_premium) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics offers, priority support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points and subscription for daily grocery |
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Bundle offers to raise order frequency |
| 3 | At-Risk Customers | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon before they are lost |

PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. Output: `outputs/customer_segments.csv`.

### Lapse problem
Cutoff 2025-09-01. Cohort: 1,156 customers with a completed order before the cutoff; 443 lapsed (38.32%) because they have no completed order in September 2025. Split: 924 train and 232 test rows (`test_size=0.2`, stratified, `random_state=42`). Features (orders before the cutoff only): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days. The segment labels were not used as features, and no data from 2025-09-01 onwards entered the features.

### Preprocessing without leakage
A scikit-learn Pipeline with StandardScaler for numeric columns and OneHotEncoder(drop="first") for city, gender and favourite category. No encoder, scaler or resampler saw test rows. No feature column has a missing value (checked in code). The scaler learned a frequency mean of 8.115 (training rows only) versus 8.029 for the full cohort; they differ because the scaler sees only the 924 training rows, not the test rows.

### Models
Baseline (always "not lapsed"): accuracy 61.6%, lapsed-class F1 0.

| Model | Best parameters | Best CV F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| LogisticRegression | C=100 | 0.651 | 0.644 | 0.652 | 0.648 |
| RandomForest | max_depth=8, min_samples_leaf=10 | 0.687 | 0.640 | 0.640 | 0.640 |

Confusion matrices on the test set (rows actual 0/1, columns predicted 0/1): LR [[111, 32], [31, 58]]; RF [[111, 32], [32, 57]].

**Selected model: RandomForest.** It had the higher cross-validated F1 (0.687 versus 0.651), and the two models are almost tied on the test set, so the choice was made on the tuning scores before looking at the test set. Missing a lapsing customer means losing all their future orders, while wrongly targeting a loyal customer only costs one discount or message, so recall of the lapsed class matters more than precision here. The final F1 of 0.64 is above the required 0.55.

### Class-balance experiment
Only the training set was over-sampled (duplicates of lapsed rows, `random_state=42`) to 570 and 570 rows, and the best logistic regression was refitted. On the unchanged test set:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| LR unbalanced | 0.644 | 0.652 | 0.648 |
| LR balanced | 0.589 | 0.854 | 0.697 |

Recall moved most (0.652 to 0.854) while precision did not rise (0.644 to 0.589). For the retention team this means they would catch many more customers who are about to lapse, but a larger share of the people they contact would not have lapsed anyway.

### Interpretation
Top 5 random-forest features: monetary (0.278), frequency (0.277), recency (0.111), average delivery minutes (0.062), average discount (0.053). Logistic regression coefficients: frequency is negative (-1.698): customers who order often rarely lapse; recency is positive (0.040): the longer since the last order, the higher the lapse risk; monetary is positive (0.285) once frequency is held fixed, meaning high spend per order with few orders is a risk sign.

### Saved model and retention list
`outputs/lapse_model.joblib` was reloaded with `joblib.load` and reproduces the original test predictions exactly. `outputs/retention_list.csv` has the 10 Premium customers (out of 40 Premium customers in the test set) with the highest predicted lapse probability; 1 of the 10 actually lapsed. The Premium flag comes from whole-window RFM, which includes the September window that defines the lapse label, so this retention list is illustrative and not a leakage-free evaluation.

## Part C

Pipeline in `src/part_c_triage.py`: input guardrails, PII redaction, retries with backoff, output validation, token budget and routing. Every model call goes through the offline `mock_llm.py`.

### Input guardrails
Empty or whitespace-only tickets get `skipped_empty` and long tickets (est_tokens > 400) get `flagged_long`; neither is ever sent to the model. Found: 4 empty and 4 long tickets (each long ticket estimates to 734 tokens).

### Redaction (email, then phone, then order id)
| Type | Substitutions |
|---|---|
| EMAIL | 55 |
| PHONE | 91 |
| ORDER_ID | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches. The user message sent to the model is exactly the redacted ticket text, and the code asserts that no email, phone or order id appears in any prompt.

### Prompts
Zero-shot prompt: about 37 estimated tokens. Few-shot prompt: about 130 estimated tokens, with an `EXAMPLES:` block containing four labelled examples (billing, delivery, technical, fraud). Both contain the word CLASSIFY, list the four categories and require JSON-only replies.

### Retries
Each model call is wrapped in a retry function: at most 3 retries, wait = 0.05 x 2^n x uniform(0.9, 1.15) seconds, one log line per retry (`outputs/retry_log.txt`). A full run over the 156 eligible tickets had exactly 19 TransientLLMErrors, all recovered by a retry; every wait lay between 0.045 and 0.0575 seconds (all were retry number 0) and no ticket ended with a permanent error.

### Output validation
JSON replies with an allowed category are `classified_json`. Otherwise the lowercase words are extracted with `re.findall(r"[a-z]+", text.lower())` and, if exactly one allowed category word is present, the ticket is `classified_keyword`. Everything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review.

| Status (full run) | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot (no budget)
| Run | Tickets classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 of 147) | 10,180 | 69.3 |
| Few-shot | 147 | 100% (147 of 147) | 24,688 | 167.9 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

All 21 zero-shot errors are fraud tickets predicted as billing (11) or technical tickets predicted as billing (10), because the zero-shot prompt makes the mock check billing words such as "charged" or "payment" first. The few-shot prompt costs about 99 more tokens per classified ticket (167.9 versus 69.3, 14,508 tokens more over the run, roughly 2.4 times), but it fixed all 21 errors, which is about 690 extra tokens per corrected ticket. This is worth paying when a wrong label is expensive, as for fraud tickets that would be sent to the normal billing queue instead of the security team. For low-risk labelling where 85 to 86% accuracy is enough or a human re-checks every label, the cheaper zero-shot prompt is the better choice.

### Token budget
Zero-shot run with TOKEN_BUDGET = 6000 (`outputs/triage_budget_demo.csv`): the budget stopped the run, giving 64 tickets with status `budget_exhausted` (80 classified_json, 7 classified_keyword, 5 flagged_invalid_output, 4 skipped_empty, 4 flagged_long). The check is made before each call, so no call is made once the running total is at or above the budget.

### Routing and final output
`outputs/triage_results.csv` (164 rows) comes from a few-shot run with TOKEN_BUDGET = 100000 and the segment file from Part B. Rules: skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted go to `human_review` (NORMAL); otherwise fraud goes to `security_team` (HIGH); otherwise Premium customers go to `premium_desk` (HIGH); otherwise `support_queue` (NORMAL). A customer without a row in the segment file counts as not premium.

| Route | Tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 (4 empty, 4 long, 9 invalid output) |
| premium_desk | 11 |

Priority: 119 NORMAL, 45 HIGH.

### Rule-validated versus human
147 tickets were classified and rule-validated (JSON schema check or single-keyword extraction) and routed automatically. 17 tickets went to humans: 4 empty, 4 too long and 9 whose model output could not be validated. Nothing was guessed.

## Part D
TODO

## Limitations and Next Steps
TODO