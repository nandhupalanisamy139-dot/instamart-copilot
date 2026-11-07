import os, sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN = "data/raw", "data/clean"
os.makedirs(CLEAN, exist_ok=True)

# ---------- LOAD ----------
cust = pd.read_csv(f"{RAW}/customers.csv")
prod = pd.read_csv(f"{RAW}/products.csv")
# orders-a text-a padikkirom, appo dhaan "N/A" maari values maraiyaadhu
orders = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)

# ---------- AUDIT ----------
def missing_table(df):
    d = df.replace("", np.nan)
    return pd.DataFrame({"dtype": df.dtypes.astype(str),
                         "missing": d.isna().sum(),
                         "missing_pct": (d.isna().mean() * 100).round(2)})

for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
    print(f"\n== {name}  shape={df.shape}")
    print(missing_table(df))

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
del_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
audit = {
    "order rows (13084)": len(orders),
    "exact duplicate rows (119)": int(orders.duplicated().sum()),
    "pending rows (656)": int((orders["status"] == "pending").sum()),
    "distinct order_id (12309)": orders["order_id"].nunique(),
    "non-numeric amounts (120)": int(amt_num.isna().sum()),
    "negative delivery (106)": int((del_num < 0).sum()),
    "blank delivery (2541)": int((orders["delivery_minutes"] == "").sum()),
    "blank age (61)": int(cust["age"].isna().sum()),
    "age outside 10-100 (6)": int(((cust["age"] < 10) | (cust["age"] > 100)).sum()),
    "raw city spellings (24)": cust["city"].nunique(),
    "raw gender spellings (10)": cust["gender"].dropna().nunique(),
    "blank gender (30)": int(cust["gender"].isna().sum()),
}
print("\n== AUDIT")
for k, v in audit.items():
    print(f"{k}: {v}")

# ---------- CLEAN CUSTOMERS ----------
c = cust.copy()
c["city"] = c["city"].str.strip().str.title()
g = c["gender"].fillna("").str.strip().str.lower()
c["gender"] = g.map({"m": "male", "male": "male", "f": "female", "female": "female"}).fillna("unknown")
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = np.nan
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])
print("\n== CLEAN CUSTOMERS")
print("cities:", c["city"].value_counts().to_dict())
print("genders:", c["gender"].value_counts().to_dict())
print("missing age:", int(c["age"].isna().sum()), "| age min/max:", c["age"].min(), c["age"].max())

# ---------- CLEAN ORDERS ----------
o = orders.copy()
o["order_ts"] = pd.to_datetime(o["order_ts"])                      # rule 1
o["quantity"] = o["quantity"].astype(int)
o["discount_pct"] = o["discount_pct"].astype(int)
print("\n== CLEAN ORDERS")

n = len(o)
o = o.drop_duplicates()                                            # rule 2
print(f"rule 2: exact duplicates removed = {n - len(o)}")

n = len(o)
o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last").reset_index(drop=True)   # rule 3
print(f"rule 3: older records removed = {n - len(o)}  -> rows now {len(o)}")

o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")   # rule 4
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss = o["order_amount"].isna()
o.loc[miss, "order_amount"] = (o.loc[miss, "price_inr"] * o.loc[miss, "quantity"]
                               * (1 - o.loc[miss, "discount_pct"] / 100)).round(2)
print(f"rule 4: amounts filled from price = {int(miss.sum())}")

o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # rule 5
neg = o["delivery_minutes"] < 0
o.loc[neg, "delivery_minutes"] = np.nan
print(f"rule 5a: negative delivery set to missing = {int(neg.sum())}")
comp = o["status"] == "completed"
med = o[comp].groupby("category")["delivery_minutes"].transform("median").reindex(o.index)
fill = comp & o["delivery_minutes"].isna()
o.loc[fill, "delivery_minutes"] = med[fill]
print(f"rule 5b: delivery filled for completed = {int(fill.sum())}")

o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

print("\n== ORDERS CHECK")
print("rows:", len(o), "(12309)")
print("status:", o["status"].value_counts().to_dict(), "(11110 completed, 1199 cancelled)")
print("missing amount:", int(o["order_amount"].isna().sum()), "(0)")
print("missing delivery, completed:", int(o.loc[o.status == "completed", "delivery_minutes"].isna().sum()), "(0)")
print("missing delivery, cancelled:", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()), "(1199)")
print("completed amount sum:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2), "(2998668.8)")

# ---------- STORE ----------
o["order_ts"] = o["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c["signup_date"] = c["signup_date"].dt.strftime("%Y-%m-%d")
c.to_csv(f"{CLEAN}/customers.csv", index=False)
prod.to_csv(f"{CLEAN}/products.csv", index=False)
o.to_csv(f"{CLEAN}/orders.csv", index=False)
with sqlite3.connect("data/instamart.db") as con:
    c.to_sql("customers", con, if_exists="replace", index=False)
    prod.to_sql("products", con, if_exists="replace", index=False)
    o.to_sql("orders", con, if_exists="replace", index=False)
print("\nSaved: data/clean/*.csv and data/instamart.db")
