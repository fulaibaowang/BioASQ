# data

## download and prepare pubmed 

! download 2026 baseline when it is there !

download
```
for i in $(seq -w 0001 1274); do   rsync -av --partial --progress --contimeout=60 --timeout=600     rsync://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed25n${i}.xml.gz     baseline2025/; done
```

xml to jsonl
```
docker run -it \
  -v path/to/pubmed:/pubmed/ \
  --platform=linux/amd64 fulaibaowang/bioasq:28.01.26 \
python /app/parse_pubmed_local.py \
    --input_dir /pubmed/ \
    --output_dir /pubmed/jsonl/ \
    --skip_existing
```