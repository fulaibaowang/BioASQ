cd ~/BioASQ


srun -p dev \
  --time=00:30:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --pty bash

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
HOME_HOST="/ceph/hpc/home/wangy"

singularity shell --nv \
  -B "${WORKDIR}:/work" \
  -B "${PUBMED_HOST}:/pubmed" \
  -B "${HOME_HOST}:/home/wangy" \
  "$CONTAINER_IMG"

python3 scripts/public/evidence/contexts_json_to_bioasq_snippets.py \
  --contexts-json /home/wangy/bioasq14_output/batch_1/evidence_snippet/BioASQ-task14bPhaseB-testset1_contexts.json \
  --corpus-path "/pubmed/jsonl_2026/*.jsonl" \
  --allow-fallback-offsets \
  --output-json /home/wangy/bioasq14_output/batch_1/evidence_snippet/BioASQ-task14bPhaseB-testset1_bioasq_snippets.json