#!/usr/bin/env python3
"""Phase E 3조건 비교 실험"""
import json, time, re, os, sys, argparse
sys.path.insert(0, '.')
from biomni.agent import A1

EVAL1_FILE = "trajectory_output/deduped_r0_eval1.jsonl"
LAB_FILE = "gdpo_labbench/gdpo_B_gpt_teaches_r0.jsonl"
PROMPT_FILE = "trajectory_collection/improved_system_prompt.txt"
OUTPUT_DIR = "phase_e_results"

EVAL1_IDS = ["eval1_gwas_variant_prioritization_134","eval1_screen_gene_retrieval_190","eval1_lab_bench_seqqa_533"]
LAB_IDS = ["lab_DbQA_train_73","lab_SeqQA_train_521","lab_SuppQA_train_57","lab_LitQA2_train_21","lab_TableQA_train_25","lab_ProtocolQA_train_24","lab_FigQA_train_36","lab_CloningScenarios_train_32"]

COND = {
    "base_r0": {"port":8003,"model":"biomni-r0","label":"Base R0"},
    "gdpo": {"port":8002,"model":"biomni-gdpo","label":"GDPO R0"},
    "gdpo_unc": {"port":8002,"model":"biomni-gdpo","label":"GDPO+Unc"},
}

def load_tasks():
    tasks = []
    with open(EVAL1_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] in EVAL1_IDS:
                tasks.append(d)
    with open(LAB_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] in LAB_IDS:
                tasks.append(d)
    print(f"태스크 로드: {len(tasks)}건")
    return tasks

def extract_answer(raw, task_type=""):
    if not raw: return ""
    raw = str(raw)
    sol = re.search(r"<solution>(.*?)</solution>", raw, re.DOTALL)
    if sol: raw = sol.group(1).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"[#*]+", "", raw).strip()
    fa = re.search(r"(?:Final Answer|Answer)[:\s]+([A-Za-z0-9_-]+)", raw, re.IGNORECASE)
    if fa: return fa.group(1).strip()
    if "variant" in task_type.lower():
        rs = re.search(r"(rs\d+)", raw)
        if rs: return rs.group(1)
    if "gene" in task_type.lower() or "causal" in task_type.lower():
        g = re.search(r"\b([A-Z][A-Z0-9]{1,15})\b", raw)
        if g: return g.group(1)
    return raw.split("\n")[0].strip()[:500]

def score_answer(pred, gt):
    gt, pred = str(gt).strip().lower(), str(pred).strip().lower()
    if gt == pred: return 1.0
    if gt in pred or pred in gt: return 0.8
    return 0.0

def run_condition(cond_name, tasks, extra):
    cfg = COND[cond_name]
    url = f"http://localhost:{cfg['port']}/v1"
    out_file = os.path.join(OUTPUT_DIR, f"phase_e_{cond_name}.jsonl")
    done = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                if line.strip(): done.add(json.loads(line)["task_id"])
    remaining = [t for t in tasks if t["task_id"] not in done]
    print(f"\n{'='*55}")
    print(f"조건: {cfg['label']} (port {cfg['port']})")
    print(f"  전체: {len(tasks)} | 완료: {len(done)} | 남은: {len(remaining)}")
    print(f"{'='*55}")
    if not remaining:
        print("  모든 태스크 완료!")
        return
    agent = A1(llm=cfg["model"], source="OpenAI", api_key="not-needed", base_url=url)
    success, total = 0, 0
    for i, task in enumerate(remaining):
        tid, ttype = task["task_id"], task.get("task_type","unknown")
        gt, question = task.get("ground_truth",""), task["question"]
        unc_prompt = ""
        if cond_name == "gdpo_unc":
            unc_prompt = "\n\nIMPORTANT: Before giving your final answer, assess your confidence. If uncertain, state uncertainty level (low/medium/high) and try an alternative approach before finalizing."
        full = question + "\n\n" + extra + unc_prompt
        print(f"\n  [{i+1}/{len(remaining)}] {tid} ({ttype})")
        t0 = time.time()
        try:
            conv, answer = agent.go(full)
            elapsed = time.time() - t0
            traj = []
            if conv:
                for j, item in enumerate(conv):
                    step = {"step": j+1}
                    if isinstance(item, dict):
                        step["role"] = item.get("role","unknown")
                        step["content"] = str(item.get("content",""))[:3000]
                    else:
                        step["role"] = "unknown"
                        step["content"] = str(item)[:3000]
                    traj.append(step)
            clean = extract_answer(answer, ttype)
            sc = score_answer(clean, gt)
            ok = sc > 0.5
            if ok: success += 1
            total += 1
            st = "OK" if ok else "FAIL"
            print(f"    {st} | ans={clean[:50]} | gt={gt} | steps={len(traj)} | {elapsed:.1f}s")
            record = {"task_id":tid,"task_type":ttype,"condition":cond_name,"answer":clean,"ground_truth":str(gt),"score":sc,"success":ok,"steps":len(traj),"elapsed_sec":round(elapsed,1),"trajectory":traj,"raw_answer":str(answer)[:2000]}
            with open(out_file,"a") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
        except Exception as e:
            elapsed = time.time() - t0
            total += 1
            print(f"    ERROR: {e} ({elapsed:.1f}s)")
            record = {"task_id":tid,"task_type":ttype,"condition":cond_name,"answer":"","ground_truth":str(gt),"score":0.0,"success":False,"steps":0,"elapsed_sec":round(elapsed,1),"error":str(e)}
            with open(out_file,"a") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
    print(f"\n  결과: {success}/{total} ({success/max(total,1)*100:.1f}%)")

def print_comparison():
    files = {c: os.path.join(OUTPUT_DIR,f"phase_e_{c}.jsonl") for c in ["base_r0","gdpo","gdpo_unc"]}
    if not all(os.path.exists(f) for f in files.values()):
        print("\n3조건 모두 완료 후 비교 가능합니다.")
        return
    print(f"\n\n{'='*60}")
    print("Phase E 3조건 비교 결과")
    print(f"{'='*60}")
    results = {}
    for c, f in files.items():
        results[c] = [json.loads(l) for l in open(f) if l.strip()]
    print(f"\n{'조건':<20} {'성공률':>8} {'평균steps':>10} {'평균시간':>10}")
    print("-"*55)
    for c, data in results.items():
        n = len(data)
        succ = sum(1 for d in data if d.get("success"))
        avg_s = sum(d.get("steps",0) for d in data)/max(n,1)
        avg_t = sum(d.get("elapsed_sec",0) for d in data)/max(n,1)
        print(f"  {COND[c]['label']:<18} {succ/max(n,1)*100:>6.1f}% {avg_s:>10.1f} {avg_t:>8.1f}s")
    print(f"\n{'태스크':<40} {'Base':>6} {'GDPO':>6} {'G+Unc':>6}")
    print("-"*60)
    all_tids = set()
    for data in results.values():
        for d in data: all_tids.add(d["task_id"])
    for tid in sorted(all_tids):
        scores = {}
        for c, data in results.items():
            m = [d for d in data if d["task_id"]==tid]
            scores[c] = m[0]["score"] if m else -1
        def fmt(s): return "O" if s>0.5 else ("X" if s>=0 else "-")
        print(f"  {tid:<38} {fmt(scores.get('base_r0',-1)):>6} {fmt(scores.get('gdpo',-1)):>6} {fmt(scores.get('gdpo_unc',-1)):>6}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["base_r0","gdpo","gdpo_unc","all_gdpo","compare"], default="all_gdpo")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.condition == "compare":
        print_comparison()
        return
    tasks = load_tasks()
    extra = open(PROMPT_FILE).read()
    if args.condition == "all_gdpo":
        run_condition("gdpo", tasks, extra)
        run_condition("gdpo_unc", tasks, extra)
        print("\n다음: Base R0를 port 8003에 로드 후 --condition base_r0 실행")
    else:
        run_condition(args.condition, tasks, extra)
    print_comparison()

if __name__ == "__main__":
    main()
