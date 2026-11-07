import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm

POLICY_DIR = os.path.join(ROOT, "data", "policies")
EVAL_CSV = os.path.join(ROOT, "data", "raw", "policy_eval.csv")


def chunk_words(text, size=40, overlap=10):
    """Windows of `size` words, each starting `size - overlap` words after the previous one.
    Stops after the first chunk whose end reaches or passes the last word."""
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def build_index(overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for ch in chunk_words(text, 40, overlap):
            chunks.append({"source": os.path.basename(path), "text": ch})
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vec": vec, "mat": mat}


INDEX = build_index(10)


def retrieve(query, k=3, index=None):
    index = index or INDEX
    sims = cosine_similarity(index["vec"].transform([query]), index["mat"])[0]
    order = np.argsort(-sims, kind="stable")[:k]
    return [{"source": index["chunks"][i]["source"], "score": float(sims[i]),
             "text": index["chunks"][i]["text"]} for i in order]


SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied context. "
          "If the answer is not in the context, say that you could not find the information.")


def answer(question, index=None):
    hits = retrieve(question, 3, index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]


def evaluate(index):
    qa = pd.read_csv(EVAL_CSV, dtype=str, keep_default_na=False)
    ans = qa[qa["expected_doc"] != "NONE"]
    unans = qa[qa["expected_doc"] == "NONE"]
    doc_ok = fact_ok = fb_ok = 0
    broken = []
    for _, r in ans.iterrows():
        text, sources = answer(r["question"], index)
        d = r["expected_doc"] in sources
        f = r["expected_fact"] in text
        doc_ok += d
        fact_ok += f
        if not (d and f):
            broken.append((r["question"], d, f, text))
    for _, r in unans.iterrows():
        text, _ = answer(r["question"], index)
        fb_ok += text.startswith("I could not find")
    return len(ans), doc_ok, fact_ok, len(unans), fb_ok, broken


if __name__ == "__main__":
    print("== CHUNKS (ref 15 total: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")
    counts = pd.Series([c["source"] for c in INDEX["chunks"]]).value_counts().sort_index()
    print(counts.to_dict(), "| total:", len(INDEX["chunks"]))

    print("\n== RETRIEVE 'refund policy late delivery' (ref: top 3 all refund_policy.txt)")
    for h in retrieve("refund policy late delivery"):
        print(h["source"], round(h["score"], 3))

    print("\n== EXAMPLE ANSWER")
    print(answer("Who must approve a refund above INR 1000?"))

    for label, ov in [("OVERLAP 10", 10), ("OVERLAP 0", 0)]:
        idx = INDEX if ov == 10 else build_index(0)
        n_a, doc_ok, fact_ok, n_u, fb_ok, broken = evaluate(idx)
        print(f"\n== EVALUATION, {label}: {len(idx['chunks'])} chunks")
        print(f"answerable: expected_doc in top 3 = {doc_ok}/{n_a} | expected_fact in answer = {fact_ok}/{n_a}"
              f" | unanswerable fallback = {fb_ok}/{n_u}  (ref overlap 10: 12/12, 12/12, 2/2)")
        for q, d, f, text in broken:
            print("  BREAKS:", q, "| doc ok:", d, "| fact ok:", f)
            print("    answer was:", text)