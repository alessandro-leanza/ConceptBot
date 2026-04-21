import json
from pathlib import Path

FILES = [
    "instructions/explicit_unambiguous.json",
    "instructions/explicit_ambiguous.json",
    "instructions/implicit.json",
    "instructions/risk_aware.json",
    "instructions/materials.json",
    "instructions/toxicity.json",
]


def main():
    ok = True
    for f in FILES:
        data = json.loads(Path(f).read_text())
        policy = data.get("policy")
        if not policy:
            print(f"[ERROR] {f}: missing policy")
            ok = False
            continue
        if "order_matters" not in policy or "type" not in policy:
            print(f"[ERROR] {f}: policy missing order_matters/type")
            ok = False
        for item in data.get("items", []):
            gold = item.get("gold")
            if not gold:
                print(f"[WARN] {f}:{item.get('id')} missing gold")
                continue
            has_seq = "sequences" in gold
            has_rules = "rules" in gold
            if not has_seq and not has_rules:
                print(f"[ERROR] {f}:{item.get('id')} gold has no sequences or rules")
                ok = False
            if has_seq:
                if not isinstance(gold["sequences"], list):
                    print(f"[ERROR] {f}:{item.get('id')} sequences is not list")
                    ok = False
            if has_rules:
                if not isinstance(gold["rules"], dict):
                    print(f"[ERROR] {f}:{item.get('id')} rules is not dict")
                    ok = False
    if ok:
        print("Instruction files OK")


if __name__ == "__main__":
    main()
