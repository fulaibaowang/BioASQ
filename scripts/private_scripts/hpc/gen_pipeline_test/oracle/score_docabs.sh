#!/bin/bash
# Score the long-context oracle arms on the many-gold subset with the local list
# metric (eval_list_local.py). Builds two qid manifests from the golden JSON:
#   ge25  = list questions with >=25 gold docs   (ladder-separating subset)
#   sweet = list & >=25 gold & >=8 nuggets        (nugget-rich sweet spot)
# Runs each completed generator-config across the 6 arms; direct_10 is baseline.
set -uo pipefail
cd ~/BioASQ
SPLIT="${SPLIT:-13B4}"
GOLD="bioasq_data/Task13BGoldenEnriched/${SPLIT}_golden.json"
B=/shared/workspace/biolab/yun/bioasq14_output
ORA=scripts/private_scripts/hpc/gen_pipeline_test/oracle
EVAL=scripts/private_scripts/hpc/gen_pipeline_test/eval_list_local.py
MDIR=$ORA/_manifests; mkdir -p "$MDIR"

python3 - "$GOLD" "$MDIR" <<'PY'
import json,sys,re
gold=json.load(open(sys.argv[1]))["questions"]; mdir=sys.argv[2]
def pmid(u):
    m=re.search(r"/(\d+)/?$",str(u)); return m.group(1) if m else None
ge25=[]; sweet=[]
for q in gold:
    if q.get("type")!="list": continue
    nd=len({pmid(u) for u in (q.get("documents") or []) if pmid(u)})
    nn=len(q.get("exact_answer") or []) if isinstance(q.get("exact_answer"),list) else 0
    if nd>=25:
        ge25.append((nd,q["id"]))
        if nn>=8: sweet.append((nd,q["id"]))
for name,rows in [("ge25",ge25),("sweet",sweet)]:
    with open(f"{mdir}/{name}.txt","w") as f:
        f.write(f"# {name}: {len(rows)} list questions\n")
        for nd,qid in rows: f.write(f"{nd}\t{qid}\n")
    print(f"{name}: {len(rows)} questions -> {mdir}/{name}.txt")
PY

ans() { ls "$B/$1/$2"/*_answers.jsonl 2>/dev/null | head -1; }

for CFG in oracle_docabs_${SPLIT}_llama oracle_docabs_${SPLIT}_gemma_nothink oracle_docabs_${SPLIT}_gemma_think; do
  [ -d "$B/$CFG" ] || continue
  # require all 6 arms present
  miss=0; for a in direct_10 direct_16 direct_40 claims_50 claims_160 facets_16; do
    [ -n "$(ans "$CFG" "$a")" ] || miss=1; done
  for MAN in ge25 sweet; do
    echo; echo "############ $CFG  |  subset=$MAN ############"
    [ "$miss" = 1 ] && echo "(NOTE: some arms missing — partial run)"
    python3 "$EVAL" --gold "$GOLD" --qids "$MDIR/$MAN.txt" \
      $( for a in direct_10 direct_16 direct_40 claims_50 claims_160 facets_16; do
           f=$(ans "$CFG" "$a"); [ -n "$f" ] && echo "--mode ${a}=$f"; done )
  done
done
