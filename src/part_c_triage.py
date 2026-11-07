import json, os, random, re, sys, time
# mock_llm.py is in the repo root, so put the root on the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import mock_llm
from mock_llm import est_tokens

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
PII = [("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
       ("PHONE", r"\b[6-9]\d{9}\b", "[PHONE]"),
       ("ORDER_ID", r"ORD-\d{6}", "[ORDER_ID]")]
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "failed_after_retries"}

os.makedirs("outputs", exist_ok=True)
LOG = open("outputs/retry_log.txt", "w", encoding="utf-8")

# ---------- PROMPTS ----------
ZERO = ('You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. '
        'Reply with JSON only, in the form {"category": "...", "reason": "..."}. Do not add any other text.')
FEW = (ZERO + '\nEXAMPLES:\n'
       'Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}\n'
       'Ticket: My order [ORDER_ID] has not arrived and the rider is not picking up calls. -> {"category": "delivery", "reason": "late order"}\n'
       'Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app crash"}\n'
       'Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "unauthorized use"}')
assert "CLASSIFY" in ZERO and "EXAMPLES:" not in ZERO
assert "CLASSIFY" in FEW and "EXAMPLES:" in FEW

# ---------- HELPERS ----------
class BudgetExhausted(Exception):
    pass

def redact(text):
    counts = {}
    for name, pat, placeholder in PII:
        text, n = re.subn(pat, placeholder, text)
        counts[name] = n
    return text, counts

def assert_no_pii(messages):
    for m in messages:
        for name, pat, _ in PII:
            assert not re.search(pat, m["content"]), f"{name} found in prompt!"

def call_with_retry(tid, messages, state, budget, label):
    for n in range(MAX_RETRIES + 1):
        if budget is not None and state["tokens"] >= budget:
            raise BudgetExhausted()
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            state["transient"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            state["waits"].append((n, wait))
            LOG.write(f"{label} | ticket={tid} | retry={n} | wait={wait:.4f}s\n")
            time.sleep(wait)

def validate(text):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""

tickets = pd.read_csv("data/raw/tickets.csv", dtype=str, keep_default_na=False)

def run(label, system_prompt, token_budget=None):
    mock_llm.reset()
    state = {"tokens": 0, "transient": 0, "waits": []}
    results = []
    for _, t in tickets.iterrows():
        tid, text = t["ticket_id"], t["text"]
        row = {"ticket_id": tid, "customer_id": t["customer_id"], "true_category": t["true_category"],
               "status": "", "category": "", "tokens_used": 0}
        if text.strip() == "":
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        else:
            clean, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            assert_no_pii(messages)
            try:
                resp = call_with_retry(tid, messages, state, token_budget, label)
                used = resp.usage.total_tokens
                state["tokens"] += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(resp.choices[0].message.content)
            except BudgetExhausted:
                row["status"] = "budget_exhausted"
            except mock_llm.TransientLLMError:
                row["status"] = "failed_after_retries"
        results.append(row)
    return results, state

def summarize(label, results, state):
    counts = pd.Series([r["status"] for r in results]).value_counts().to_dict()
    classified = [r for r in results if r["status"] in ("classified_json", "classified_keyword")]
    correct = sum(r["category"] == r["true_category"] for r in classified)
    print(f"\n== {label}")
    print("status counts:", counts)
    print("TransientLLMErrors:", state["transient"], "| retries logged:", len(state["waits"]))
    for n, w in state["waits"]:
        assert BASE_DELAY * 2 ** n * 0.9 <= w <= BASE_DELAY * 2 ** n * 1.15, "wait outside jitter range"
    if state["waits"]:
        ws = [w for _, w in state["waits"]]
        print("wait range seconds:", round(min(ws), 4), "to", round(max(ws), 4))
    if classified:
        print(f"classified: {len(classified)} | accuracy: {correct}/{len(classified)} = {correct / len(classified) * 100:.1f}%")
        print(f"total tokens: {state['tokens']} | tokens per classified ticket: {state['tokens'] / len(classified):.1f}")
    return classified

seg = pd.read_csv("outputs/customer_segments.csv")
SEG_NAME = dict(zip(seg["customer_id"], seg["segment_name"]))
PREM = dict(zip(seg["customer_id"], seg["is_premium"].astype(str) == "True"))

def write_csv(results, path):
    rows = []
    for r in results:
        cid = r["customer_id"]
        if r["status"] in HUMAN:
            route, prio = "human_review", "NORMAL"
        elif r["category"] == "fraud":
            route, prio = "security_team", "HIGH"
        elif PREM.get(cid, False):
            route, prio = "premium_desk", "HIGH"
        else:
            route, prio = "support_queue", "NORMAL"
        rows.append({"ticket_id": r["ticket_id"], "customer_id": cid, "status": r["status"],
                     "category": r["category"], "segment_name": SEG_NAME.get(cid, ""),
                     "priority": prio, "route": route, "tokens_used": r["tokens_used"]})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

# ---------- 1. GUARDRAIL COUNTS ----------
empty = int((tickets["text"].str.strip() == "").sum())
long_ = int(tickets["text"].map(lambda s: s.strip() != "" and est_tokens(s) > MAX_INPUT_TOKENS).sum())
long_tok = [est_tokens(s) for s in tickets["text"] if est_tokens(s) > MAX_INPUT_TOKENS]
print("== GUARDRAILS (ref 4 empty, 4 long at 734 tokens)")
print("empty:", empty, "| long:", long_, "| long token estimates:", sorted(set(long_tok)))

# ---------- 2. REDACTION ----------
tot = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
left = 0
for text in tickets["text"]:
    clean, c = redact(text)
    for k in tot:
        tot[k] += c[k]
    left += sum(len(re.findall(pat, clean)) for _, pat, _ in PII)
print("\n== REDACTION (ref 55 emails, 91 phones, 132 order ids, 278 total)")
print(tot, "| total:", sum(tot.values()), "| left after re-applying patterns (0):", left)
print("prompt sizes (estimated tokens): zero-shot", est_tokens(ZERO), "| few-shot", est_tokens(FEW))

# ---------- 3. ZERO-SHOT vs FEW-SHOT ----------
zr, zs = run("zero-shot", ZERO)
zc = summarize("ZERO-SHOT full run (ref 135/12/9, accuracy 85.7%, 19 transient errors)", zr, zs)
fr, fs = run("few-shot", FEW)
summarize("FEW-SHOT full run (ref accuracy 100%)", fr, fs)

cm = pd.crosstab(pd.Series([r["true_category"] for r in zc], name="true"),
                 pd.Series([r["category"] for r in zc], name="predicted"))
print("\n== ZERO-SHOT CONFUSION TABLE (ref errors: fraud->billing 11, technical->billing 10)")
print(cm)

# ---------- 4. BUDGET DEMO ----------
BUDGET_DEMO = 6000
br, bs = run("zero-shot budget demo", ZERO, BUDGET_DEMO)
summarize(f"BUDGET DEMO zero-shot, TOKEN_BUDGET={BUDGET_DEMO}", br, bs)
write_csv(br, "outputs/triage_budget_demo.csv")
print("budget_exhausted tickets:", sum(r["status"] == "budget_exhausted" for r in br), "(must be at least 1)")

# ---------- 5. FINAL RESULT ----------
rr, rs = run("few-shot final", FEW, 100000)
summarize("FINAL few-shot run, TOKEN_BUDGET=100000", rr, rs)
final = write_csv(rr, "outputs/triage_results.csv")
print("\nrows in triage_results.csv (164):", len(final))
print(final["route"].value_counts().to_dict())
print(final["priority"].value_counts().to_dict())
LOG.close()
print("\nretry log lines are in outputs/retry_log.txt")