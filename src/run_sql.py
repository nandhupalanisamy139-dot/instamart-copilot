import re, sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"(?m)^-- (Q\d):[^\n]*\n", text)
queries = dict(zip(parts[1::2], parts[2::2]))

con = sqlite3.connect("data/instamart.db")
res = {q: pd.read_sql_query(sql.strip(), con) for q, sql in queries.items()}
con.close()

print("Q1 rows (30):", len(res["Q1"]))
print("Q2 rows (30):", len(res["Q2"]), "| total (2998668.8):", round(res["Q2"]["revenue"].sum(), 2))
print("Q2 largest cell:\n", res["Q2"].head(1).to_string(index=False), "(Bengaluru / Electronics 233909.45)")
print("Q3 rows (11):", len(res["Q3"]))
print("Q4 rows (25):", len(res["Q4"]))
print("Q5 rows (550):", len(res["Q5"]), "| average (2552.06):", res["Q5"]["avg_total"].iloc[0])
q6 = res["Q6"]
print("Q6 top customer per city:\n", q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER (8):", int((q6["rnk"] != q6["row_num"]).sum()))
print("Q7 rows (183):", len(res["Q7"]), "| last cumulative (2998668.8):", res["Q7"]["cumulative_revenue"].iloc[-1])