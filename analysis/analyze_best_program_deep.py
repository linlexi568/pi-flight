from __future__ import annotations
import os
import json
from typing import Any, Dict

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(ROOT)
PROGRAM = os.path.join(WS, "01_pi_flight", "results", "best_program.json")
OUT_DIR = os.path.join(WS, "results", "summaries")
OUT_MD = os.path.join(OUT_DIR, "best_program_summary.md")


def summarize_program(prog: Any) -> str:
    lines = []
    try:
        n_rules = len(prog) if isinstance(prog, list) else (len(prog.get("rules", [])) if isinstance(prog, dict) else None)
    except Exception:
        n_rules = None
    lines.append("# 最优程序结构化摘要\n")
    if n_rules is not None:
        lines.append(f"- 规则数: {n_rules}")
    else:
        lines.append("- 规则数: 未知（未能从 JSON 中解析出规则列表）")

    # 尝试抽取规则的关键信息
    def fmt_rule(r: Dict[str, Any], idx: int) -> str:
        # 常见字段猜测：condition / predicate / gains / P I D / action / then / else
        cond = r.get("condition") or r.get("predicate") or r.get("if")
        gains = r.get("gains") or {k: r.get(k) for k in ("P","I","D") if k in r}
        action = r.get("action") or r.get("then")
        return f"- 规则 {idx}: 条件={json.dumps(cond, ensure_ascii=False)} | 增益={gains} | 动作={json.dumps(action, ensure_ascii=False)}"

    rules = []
    if isinstance(prog, list):
        rules = prog
    elif isinstance(prog, dict):
        if isinstance(prog.get("rules"), list):
            rules = prog.get("rules")  # type: ignore
    
    if rules:
        lines.append("\n## 规则明细（最佳努力解析）")
        for i, r in enumerate(rules, 1):
            if isinstance(r, dict):
                lines.append(fmt_rule(r, i))
            else:
                lines.append(f"- 规则 {i}: {type(r).__name__}")
    else:
        lines.append("\n> 未能自动识别规则列表，以下为 JSON 原文概要（截断显示）：")

    # 附：原 JSON（缩进美化）
    lines.append("\n## 原始 JSON（美化）\n")
    try:
        lines.append("```json")
        lines.append(json.dumps(prog, indent=2, ensure_ascii=False)[:5000])  # 限制长度
        lines.append("```")
    except Exception as e:
        lines.append(f"(无法美化输出：{e})")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(PROGRAM):
        raise FileNotFoundError(f"Program not found: {PROGRAM}")
    with open(PROGRAM, "r", encoding="utf-8") as f:
        prog = json.load(f)
    report = summarize_program(prog)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[WRITE] {OUT_MD}")


if __name__ == "__main__":
    main()
