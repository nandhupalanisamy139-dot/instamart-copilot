import glob, json, os, sys, time
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import mock_llm
from part_d_rag import retrieve

DB = os.path.join(ROOT, "data", "instamart.db")
OUT = os.path.join(ROOT, "outputs")
RUNS = os.path.join(OUT, "agent_runs.jsonl")
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")
STATE_DIR = os.path.join(ROOT, "state")
os.makedirs(OUT, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
_CTX = {"run_id": None}


# ---------------- TOOLS ----------------
def get_order(order_id):
    import sqlite3
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, minutes = row
    return {"order_id": oid, "status": status,
            "order_amount": None if amount is None else float(amount),
            "delivery_minutes": None if minutes is None else float(minutes)}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and order_amount is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "run_id": _CTX["run_id"], "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up one order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate the refund amount from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue a refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                      "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                      "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools to look up the order, check the policy, "
          "calculate the refund, then issue it or escalate it. Finish with one line that starts with "
          "REFUND, ESCALATE, DENIED or NOT_FOUND.")


class SimulatedCrash(Exception):
    pass


# ---------------- LOOP ----------------
def _checkpoint(st):
    path = os.path.join(STATE_DIR, f"{st['run_id']}.json")
    keys = ("run_id", "scenario_id", "messages", "tokens_used", "step", "tools_executed", "steps", "llm_calls")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: st[k] for k in keys}, f, default=str)


def _finish(st, status, final):
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final,
            "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}


def _fallback(st, status, reason):
    done = ", ".join(s["tool"] for s in st["steps"]) or "none"
    return _finish(st, status, f"Handing over to a human agent: {reason}. Tools completed so far: {done}.")


def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    _CTX["run_id"] = st["run_id"]
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds} seconds reached")
        if st["llm_calls"] >= max_iterations:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls")
        resp = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit of {token_limit} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content or ""
            return _finish(st, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        args = dict(tc.arguments)
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        obs = TOOL_MAP[tc.name](**args)
        st["steps"].append({"tool": tc.name, "arguments": args, "observation": obs})
        st["tools_executed"].append(tc.name)
        st["step"] += 1
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        _checkpoint(st)
        if crash_after_tools and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool number {crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}_{int(time.time() * 1000)}", "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "llm_calls": 0}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st, max_iterations, token_limit, time_limit_seconds)


def refund_of(res):
    for s in reversed(res["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0


if __name__ == "__main__":
    for p in (RUNS, LEDGER):
        if os.path.exists(p):
            os.remove(p)
    for f in glob.glob(os.path.join(STATE_DIR, "*.json")):
        os.remove(f)

    sc = pd.read_csv(os.path.join(ROOT, "data", "raw", "agent_scenarios.csv"), dtype=str, keep_default_na=False)
    print("== SEVEN SCENARIOS")
    print("| scenario | status | refund INR | model calls | tokens |")
    print("|---|---|---|---|---|")
    all_ok = True
    for _, r in sc.iterrows():
        res = run_agent(r["message"], r["scenario_id"], run_id=f"{r['scenario_id']}_main")
        amt = refund_of(res)
        ok = res["status"].startswith(r["expected_outcome"]) and abs(amt - float(r["expected_refund_inr"])) <= 0.01
        all_ok = all_ok and ok
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"scenario_id": r["scenario_id"], "status": res["status"],
                                "final_answer": res["final_answer"], "llm_calls": res["llm_calls"],
                                "tokens_used": res["tokens_used"], "steps": res["steps"]}, default=str) + "\n")
        print(f"| {r['scenario_id']} | {res['status']} | {amt:.2f} | {res['llm_calls']} | {res['tokens_used']} |"
              f"  expected {r['expected_outcome']} {float(r['expected_refund_inr']):.2f} -> {'OK' if ok else 'FAIL'}")
        if r["scenario_id"] == "S7":
            print("   S7 must be FALLBACK_MAX_ITERATIONS with 6 model calls:", res["status"], res["llm_calls"])
    print("all scenarios match expected outcome and refund:", all_ok)

    s1 = sc.iloc[0]["message"]
    print("\n== STOP-SAFELY DEMOS")
    d1 = run_agent(s1, "S1", token_limit=200, run_id="S1_token_limit_demo")
    print("S1 with token_limit=200 ->", d1["status"], "| calls", d1["llm_calls"], "| tokens", d1["tokens_used"])
    print("  ", d1["final_answer"])
    d2 = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")
    print("S1 with time_limit_seconds=0 ->", d2["status"], "| calls", d2["llm_calls"])
    print("  ", d2["final_answer"])

    print("\n== CRASH AND RESUME (S1)")
    try:
        run_agent(s1, "S1", run_id="S1_crash_demo", crash_after_tools=2)
    except SimulatedCrash as e:
        print("crash:", e)
    with open(os.path.join(STATE_DIR, "S1_crash_demo.json"), encoding="utf-8") as f:
        print("tools executed before the crash:", json.load(f)["tools_executed"])
    res = resume_agent("S1_crash_demo")
    with open(os.path.join(STATE_DIR, "S1_crash_demo.json"), encoding="utf-8") as f:
        tools = json.load(f)["tools_executed"]
    print("tools executed across crash and resume:", tools)
    print("final:", res["status"], refund_of(res), "| four tools, none repeated:",
          tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"])

    print("\nledger lines:", sum(1 for _ in open(LEDGER, encoding="utf-8")))
    print("checkpoint files in state/:", len(glob.glob(os.path.join(STATE_DIR, "*.json"))))