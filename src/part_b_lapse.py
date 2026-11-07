import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id", how="left")

# ---------- FEATURES: only orders BEFORE the cutoff ----------
before = o[o["order_ts"] < CUTOFF]
comp = before[before["status"] == "completed"]
g = comp.groupby("customer_id")
feat = pd.DataFrame({"last": g["order_ts"].max(),
                     "frequency": g["order_id"].nunique(),
                     "monetary": g["order_amount"].sum(),
                     "avg_discount": g["discount_pct"].mean(),
                     "avg_delivery": g["delivery_minutes"].mean()})
feat["recency"] = (CUTOFF - feat["last"]).dt.days
feat = feat.drop(columns="last")
cancelled = before[before["status"] == "cancelled"].groupby("customer_id").size()
final = before.groupby("customer_id").size()
feat["cancel_rate"] = cancelled.reindex(feat.index).fillna(0) / final.reindex(feat.index)
cnt = (comp.groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id"))
feat["fav_category"] = cnt["category"]
feat = feat.join(c.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

# ---------- LABEL: no completed order in September ----------
sept = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < END) & (o["status"] == "completed")]
y = (~feat.index.isin(set(sept["customer_id"]))).astype(int)
y = pd.Series(y, index=feat.index, name="lapsed")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["fav_category", "city", "gender"]
X = feat[NUM + CAT]
assert not X.isna().any().any(), "missing values found"
print("no missing values in features: OK")

print("\n== COHORT (ref 1156 customers, 443 lapsed = 38.32%)")
print("cohort:", len(X), "| lapsed:", int(y.sum()), f"({y.mean() * 100:.2f}%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test (ref 924/232):", len(X_train), len(X_test))

def metrics(yt, yp):
    return (precision_score(yt, yp, zero_division=0), recall_score(yt, yp, zero_division=0),
            f1_score(yt, yp, zero_division=0))

# ---------- BASELINE ----------
base = np.zeros(len(y_test), dtype=int)
print("\n== BASELINE always 'not lapsed' (ref accuracy 61.6%, F1 0)")
print("accuracy:", round(accuracy_score(y_test, base) * 100, 1), "| lapsed F1:", f1_score(y_test, base, zero_division=0))

def pipe(clf):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", pre), ("clf", clf)])

# ---------- TUNE ----------
lr = GridSearchCV(pipe(LogisticRegression(max_iter=1000)), {"clf__C": [0.01, 0.1, 1, 10, 100]},
                  cv=5, scoring="f1").fit(X_train, y_train)
rf = GridSearchCV(pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                  {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                  cv=5, scoring="f1").fit(X_train, y_train)
print("\n== TUNING")
print("LR best:", lr.best_params_, "cv F1:", round(lr.best_score_, 3), "(ref C=1)")
print("RF best:", rf.best_params_, "cv F1:", round(rf.best_score_, 3), "(ref depth 8, leaf 10)")

# scaler learned mean vs cohort mean (training rows only vs everyone)
sc = lr.best_estimator_.named_steps["prep"].named_transformers_["num"]
print("scaler mean of frequency (train only):", round(sc.mean_[NUM.index("frequency")], 3),
      "| full-cohort mean:", round(X["frequency"].mean(), 3))

# ---------- TEST EVALUATION ----------
print("\n== TEST SET (lapsed class) (ref LR F1 about 0.62, RF about 0.65)")
for name, m in [("LogisticRegression", lr), ("RandomForest", rf)]:
    pred = m.predict(X_test)
    pr, rc, f1 = metrics(y_test, pred)
    print(name, "| precision", round(pr, 3), "recall", round(rc, 3), "F1", round(f1, 3))
    print(confusion_matrix(y_test, pred))

best = rf if rf.best_score_ >= lr.best_score_ else lr
best_name = "RandomForest" if best is rf else "LogisticRegression"
print("\nselected model (higher cross-validated F1):", best_name)

# ---------- CLASS-BALANCE EXPERIMENT (training set only) ----------
need = int((y_train == 0).sum() - (y_train == 1).sum())
extra = pd.Series(y_train[y_train == 1].index).sample(n=need, replace=True, random_state=42).values
X_bal = pd.concat([X_train, X_train.loc[extra]])
y_bal = pd.concat([y_train, y_train.loc[extra]])
print("\n== BALANCED TRAIN counts (ref 570 each):", int((y_bal == 0).sum()), int((y_bal == 1).sum()))
lr_bal = clone(lr.best_estimator_).fit(X_bal, y_bal)
for name, m in [("LR unbalanced", lr.best_estimator_), ("LR balanced  ", lr_bal)]:
    pr, rc, f1 = metrics(y_test, m.predict(X_test))
    print(name, "| precision", round(pr, 3), "recall", round(rc, 3), "F1", round(f1, 3))
print("(ref: recall about 0.64 -> 0.85, precision stays near 0.6)")

# ---------- INTERPRET ----------
names = rf.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\n== TOP 5 RF FEATURE IMPORTANCES")
print(imp.head(5).round(3))
coef = pd.Series(lr.best_estimator_.named_steps["clf"].coef_[0], index=names)
print("\n== LR COEFFICIENTS (numeric features; + means higher lapse risk)")
print(coef[[f"num__{n}" for n in NUM]].round(3))

# ---------- PERSIST ----------
joblib.dump(best.best_estimator_, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(re.predict(X_test), best.best_estimator_.predict(X_test))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

# ---------- RETENTION LIST (illustrative) ----------
seg = pd.read_csv("outputs/customer_segments.csv")
prem = set(seg.loc[seg["is_premium"], "customer_id"])
t = X_test[X_test.index.isin(prem)].copy()
t["lapse_probability"] = re.predict_proba(t)[:, 1]
t["actually_lapsed"] = y_test.loc[t.index]
out = (t.sort_values("lapse_probability", ascending=False).head(10)
       .reset_index().rename(columns={"index": "customer_id"})
       [["customer_id", "lapse_probability", "actually_lapsed"]])
out["lapse_probability"] = out["lapse_probability"].round(4)
out.to_csv("outputs/retention_list.csv", index=False)
print("\n== RETENTION LIST (premium customers in test set:", len(t), ")")
print(out)
print("truly lapsed among them:", int(out["actually_lapsed"].sum()), "of", len(out))