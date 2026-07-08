#!/bin/bash
#SBATCH -J bioasq_docabs_build
#SBATCH -p amd
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Plan B step 1: build the long-context oracle evidence (full gold abstracts).
# CPU-only single pass over the PubMed corpus shards -> new-schema evidence
# reused by the 4-arm oracle (run_oracle_arms.sh with CTX_ALL/CTX_16 overrides).
#
#   SPLIT   default 13B4
#   TOPN    default 16   (first-N gold docs per question for the *_16 file)
set -euo pipefail
cd ~/BioASQ; mkdir -p logs
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
PUBMED_HOST="/shared/workspace/biolab/pubmed"; WORKDIR="${PWD}"
SPLIT="${SPLIT:-13B4}"
TOPN="${TOPN:-16}"

srun --container-image="${CONTAINER_IMG}" --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed" \
  --container-workdir /work \
  bash -lc '
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    ORA=scripts/private_scripts/hpc/gen_pipeline_test/oracle
    GOLDEN=bioasq_data/Task13BGoldenEnriched/'"$SPLIT"'_golden.json
    python "$ORA/build_docabs_oracle.py" \
      --golden "$GOLDEN" \
      --corpus-glob "/pubmed/jsonl_2026/*.jsonl" \
      --out-all  "$ORA/oracle_docabs_all_'"$SPLIT"'.jsonl" \
      --out-topn "$ORA/oracle_docabs_'"$TOPN"'_'"$SPLIT"'.jsonl" \
      --topn '"$TOPN"'
  '
echo "Finished ${SLURM_JOB_ID} at $(date)"
