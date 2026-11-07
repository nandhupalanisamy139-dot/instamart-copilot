import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)

# ---------- RFM ----------
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = o[o["status"] == "completed"]
snapshot = pd.Timestamp("2025-10-01 00:00:00")
g = comp.groupby("customer_id")
rfm = pd.DataFrame({"last": g["order_ts"].max(),
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum()}).reset_index()
rfm["recency"] = (snapshot - rfm["last"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("== RFM")
print("customers (1175):", len(rfm))
print("recency median (19) / min (0) / max (180):", rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max())
print("frequency mean (9.46) / max (33):", round(rfm["frequency"].mean(), 2), rfm["frequency"].max())
print("monetary mean (2552.06) / max (14766.55):", round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))

# ---------- K-means table ----------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n== K-MEANS (K, inertia, silhouette)")
print("ref: 2:1820.8/0.465  3:1268.2/0.445  4:743.1/0.526  5:617.4/0.481  6:503.2/0.482  7:408.5/0.416  8:366.5/0.412")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3))

# ---------- Final K = 4 ----------
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     recency=("recency", "mean"),
                                     frequency=("frequency", "mean"),
                                     monetary=("monetary", "mean")).round(2)

# naming rule: highest monetary = Premium, highest frequency = Loyal Regulars,
# of the remaining two the higher recency (longer silent) = At-Risk, the other = Occasional Buyers
names = {}
prem = prof["monetary"].idxmax()
names[prem] = "Premium Big Spenders"
rest = [i for i in prof.index if i != prem]
loyal = prof.loc[rest, "frequency"].idxmax()
names[loyal] = "Loyal Regulars"
rest = [i for i in rest if i != loyal]
risk = prof.loc[rest, "recency"].idxmax()
names[risk] = "At-Risk Customers"
for i in rest:
    if i != risk:
        names[i] = "Occasional Buyers"
prof["segment_name"] = prof.index.map(names)
print("\n== SEGMENT PROFILE (ref sizes about 436/138/419/182; the 138 one: monetary about 7130, freq about 6.8; the 419 one: freq about 19.4, recency about 8.8)")
print(prof)

# ---------- PCA ----------
pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print("\n== PCA ratios (ref 0.634 / 0.218 / total 0.852):", r.round(3), round(r.sum(), 3))

# ---------- Save ----------
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == prem
rfm["monetary"] = rfm["monetary"].round(2)
rfm.to_csv("outputs/customer_segments.csv", index=False)
print("\nSaved outputs/customer_segments.csv | premium customers:", int(rfm["is_premium"].sum()))