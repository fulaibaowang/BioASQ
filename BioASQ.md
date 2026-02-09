# BioASQ

# Website
the task: 

https://participants-area.bioasq.org/general_information/Task14b/

## Official leaderboards (past years results)
https://participants-area.bioasq.org/results/13b/phaseA/

https://bioasq.org/participate/thirteenth-challenge-winners

Teams rank top at PHASE A and publish their methods:
- UA: https://ceur-ws.org/Vol-4038/paper_22.pdf , 
 2nd place in documents, BM25 + re-ranked using a fine-tuned BERT-based cross-encoder
- NCU-IISR: https://ceur-ws.org/Vol-4038/paper_12.pdf , https://ceur-ws.org/Vol-4038/paper_19.pdf
 2nd place in snippets, BM25 + re-ranked using the BAAI/bge-rerankerv2-m3 mode

Teams rank top at PHASE A+ and publish their methods:
- UR: https://arxiv.org/abs/2508.05366 ,
 1st place in EXCAT
- NSUT: https://ceur-ws.org/Vol-4038/paper_46.pdf

Teams rank top at PHASE B and publish their methods:
- DMIS: https://ceur-ws.org/Vol-4038/paper_25.pdf


## Official evaluation measures + scripts (what you need to reproduce scores)
https://github.com/BioASQ/Evaluation-Measures

## download datesets
Official datasets download hub (training + “golden enriched”):
https://participants-area.bioasq.org/datasets/

## report
https://www.arxiv.org/pdf/2508.20554

Practical way to find “winner methods” fast:

Open the winners page to see the team names for the phase/batch you care about.  
Go to the Phase B results table and locate those teams/systems; open their System Description. 
BioASQ Participants Area.
Search that team name inside the CEUR-WS volume (Vol-4038 for 2025) to find their full system paper(s). 
ceur-ws.org

## public repo
https://github.com/SamyAteia/bioasq2024
https://ceur-ws.org/Vol-4038/
https://ceur-ws.org/Vol-4038/paper_44.pdf
https://github.com/lasigeBioTM/BioASQ13_2025

# question types

```
yun@Air BioASQ % cd bioasq_data/BioASQ-training14b
yun@Air BioASQ-training14b % FILE="trainining14b.json"
yun@Air BioASQ-training14b % total=$(jq '.questions | length' "$FILE")

yun@Air BioASQ-training14b % echo $total
5729
yun@Air BioASQ-training14b % jq -r '.questions[].type' "$FILE" | sort | uniq -c \
| awk -v total="$total" '{printf "%-8s %5d  (%5.1f%%)\n",$2,$1,100*$1/total}'
factoid   1695  ( 29.6%)
list      1130  ( 19.7%)
summary   1363  ( 23.8%)
yesno     1541  ( 26.9%)
```





