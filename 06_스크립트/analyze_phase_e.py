#!/usr/bin/env python3
"""Phase E 결과 자동 분석"""
import json, os
from collections import defaultdict

OUTPUT_DIR = "phase_e_results"
CONDS = {"base_r0": "Base R0", "gdpo": "GDPO R0", "gdpo_unc": "GDPO+Unc"}

def load():
    results = {}
    for c, label in CONDS.items():
        f = os.path.join(OUTPUT_DIR, f"phase_e_{c}.jsonl")
        if os.path.exists(f):
            data = [json.loads(l) for l in open(f) if l.strip()]
            results[c] = data
            print(f"  {label}: {len(data)}건")
    return results

def analyze(results):
    print(f"\n{'='*65}")
    print("Phase E 3조건 비교 결과")
    print(f"{'='*65}")
    print(f"\n  {'조건':<20} {'성공률':>8} {'성공':>6} {'Steps':>8} {'시간':>8}")
    print(f"  {'─'*55}")
    summary = {}
    for c in ["base_r0","gdpo","gdpo_unc"]:
        if c not in results: continue
        d = results[c]
        n = len(d)
        s = sum(1 for x in d if x.get("success"))
        st = sum(x.get("steps",0) for x in d)/max(n,1)
        tm = sum(x.get("elapsed_sec",0) for x in d)/max(n,1)
        r = s/max(n,1)*100
        summary[c] = {"n":n,"succ":s,"rate":r,"steps":st,"time":tm}
        print(f"  {CONDS[c]:<20} {r:>6.1f}% {s:>3}/{n:<3} {st:>7.1f} {tm:>6.1f}s")
    if "base_r0" in summary and "gdpo" in summary:
        d = summary["gdpo"]["rate"]-summary["base_r0"]["rate"]
        print(f"\n  GDPO 효과: {summary['base_r0']['rate']:.1f}% -> {summary['gdpo']['rate']:.1f}% ({'+' if d>=0 else ''}{d:.1f}%p)")
    if "gdpo" in summary and "gdpo_unc" in summary:
        d = summary["gdpo_unc"]["rate"]-summary["gdpo"]["rate"]
        print(f"  Unc 효과: {summary['gdpo']['rate']:.1f}% -> {summary['gdpo_unc']['rate']:.1f}% ({'+' if d>=0 else ''}{d:.1f}%p)")

    print(f"\n  {'태스크':<35} {'Base':>6} {'GDPO':>6} {'G+Unc':>6} {'변화':>8}")
    print(f"  {'─'*60}")
    tids = set()
    for data in results.values():
        for x in data: tids.add(x["task_id"])
    imp,deg,unc2 = 0,0,0
    for tid in sorted(tids):
        sc = {}
        for c,data in results.items():
            m = [x for x in data if x["task_id"]==tid]
            if m: sc[c] = m[0]
        def f(c): return " O " if c in sc and sc[c].get("success") else (" X " if c in sc else " - ")
        ch = ""
        if "base_r0" in sc and "gdpo" in sc:
            b,g = sc["base_r0"].get("success",False), sc["gdpo"].get("success",False)
            if not b and g: ch="UP"; imp+=1
            elif b and not g: ch="DOWN"; deg+=1
            else: ch="SAME"; unc2+=1
        short = tid.replace("eval1_","E:").replace("lab_","L:").replace("_train_","#")
        print(f"  {short:<35} {f('base_r0'):>6} {f('gdpo'):>6} {f('gdpo_unc'):>6} {ch:>8}")
    if "base_r0" in results:
        print(f"\n  개선:{imp} | 악화:{deg} | 유지:{unc2}")

    out = os.path.join(OUTPUT_DIR, "phase_e_comparison.json")
    comp = {"summary": summary, "tasks": {}}
    for tid in sorted(tids):
        comp["tasks"][tid] = {}
        for c,data in results.items():
            m = [x for x in data if x["task_id"]==tid]
            if m:
                x = m[0]
                comp["tasks"][tid][c] = {"success":x.get("success"),"score":x.get("score"),"steps":x.get("steps"),"time":x.get("elapsed_sec"),"answer":x.get("answer","")[:100]}
    with open(out,"w") as fp: json.dump(comp,fp,ensure_ascii=False,indent=2)
    print(f"\n  저장: {out}")

results = load()
if results: analyze(results)
else: print("결과 파일 없음")
