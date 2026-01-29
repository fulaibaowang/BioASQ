```
cd BioASQ
srun -p dev --time=12:00:00 -c 4 \
  --container-image=fulaibaowang/bioasq:28.01.26 \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed:/pubmed" \
  --container-workdir /work \
  --pty bash
```


```
python /app/parse_pubmed_local.py \
    --input_dir /pubmed/baseline \
    --output_dir /pubmed/jsonl/baseline \
    --skip_existing
```


srun -p dev --time=12:00:00 -c 4 \
  --container-image=fulaibaowang/bioasq:28.01.26 \
  --container-mount-home \
  --container-mounts "${PWD}:/work,/shared/workspace/biolab/pubmed_test:/pubmed_test" \
  --container-workdir /work \
  --pty bash

python /app/parse_pubmed_local.py \
  --input_dir /pubmed_test \
  --output_dir /pubmed_test/jsonl/ \
  --skip_existing