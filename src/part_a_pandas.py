import numpy as np
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")

comp = (o[o["status"] == "completed"]
        .merge(c[["customer_id", "city"]], on="customer_id")
        .merge(p[["product_id", "category"]], on="product_id"))

# (a) pivot table with margins, compare with SQL total
pv = pd.pivot_table(comp, index="city", columns="category", values="order_amount",
                    aggfunc="sum", margins=True)
print("== PIVOT (revenue by city x category)")
print(pv.round(2))
print("pivot grand total (SQL total 2998668.8):", round(pv.loc["All", "All"], 2))

# (a) per-city named aggregation
city = comp.groupby("city").agg(total=("order_amount", "sum"),
                                mean=("order_amount", "mean"),
                                count=("order_amount", "count")).round(2)
print("\n== PER CITY (Bengaluru 523643.40 / 276.47 / 1894, Mumbai 479058.50 / 297.00 / 1613)")
print(city)

# (b) distribution of completed order_amount
x = comp["order_amount"]
q1, med, q3 = x.quantile(0.25), x.median(), x.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\n== order_amount STATS (completed)")
print("mean (269.91):", round(x.mean(), 2), "| median (135.00):", round(med, 2))
print("skewness (about 4.61):", round(x.skew(), 2))
print("Q1 (64.80):", round(q1, 2), "| Q3 (297.00):", round(q3, 2))
print("lower fence:", round(lo, 2), "| upper fence (645.30):", round(hi, 2))
print("outliers below:", int((x < lo).sum()), "(0) | above (897):", int((x > hi).sum()))

# (b) correlation matrix
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = comp[cols].corr()
print("\n== CORRELATION MATRIX")
print(corr.round(3))
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
best = upper.abs().idxmax()
print("largest |corr| pair (order_amount, discount_pct about -0.095):", best, round(upper[best], 3))