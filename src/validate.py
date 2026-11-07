import sqlite3
import pandas as pd

raw = pd.read_csv("data/raw/orders.csv", dtype=str, keep_default_na=False)
con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
con.close()

comp = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

checks = [
    ("raw order rows", len(raw), 13084),
    ("exact duplicates", int(raw.duplicated().sum()), 119),
    ("pending rows", int((raw["status"] == "pending").sum()), 656),
    ("distinct raw order_id", raw["order_id"].nunique(), 12309),
    ("non-numeric amounts", int(pd.to_numeric(raw["order_amount"], errors="coerce").isna().sum()), 120),
    ("negative delivery", int((pd.to_numeric(raw["delivery_minutes"], errors="coerce") < 0).sum()), 106),
    ("blank delivery", int((raw["delivery_minutes"] == "").sum()), 2541),
    ("clean order rows", len(o), 12309),
    ("unique order_id after cleaning", o["order_id"].nunique(), 12309),
    ("completed orders", len(comp), 11110),
    ("cancelled orders", len(canc), 1199),
    ("pending after cleaning", int((o["status"] == "pending").sum()), 0),
    ("missing order_amount", int(o["order_amount"].isna().sum()), 0),
    ("missing delivery (completed)", int(comp["delivery_minutes"].isna().sum()), 0),
    ("missing delivery (cancelled)", int(canc["delivery_minutes"].isna().sum()), 1199),
    ("completed amount sum", round(float(comp["order_amount"].sum()), 2), 2998668.80),
    ("distinct cities", c["city"].nunique(), 6),
    ("Pune customers", int((c["city"] == "Pune").sum()), 227),
    ("female customers", int((c["gender"] == "female").sum()), 610),
    ("unknown gender", int((c["gender"] == "unknown").sum()), 30),
    ("missing age", int(c["age"].isna().sum()), 0),
    ("min age", float(c["age"].min()), 18.0),
    ("max age", float(c["age"].max()), 60.0),
]

failed = 0
for name, actual, expected in checks:
    ok = actual == expected
    failed += not ok
    print(("OK   " if ok else "FAIL ") + f"{name}: {actual} (expected {expected})")

assert failed == 0, f"{failed} checkpoint(s) failed"
print("ALL CHECKPOINTS PASSED")