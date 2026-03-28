# HyDE gating notes

Generated from the three uploaded BioASQ files.

Policy used:
- Enable HyDE for list questions.
- Enable HyDE for summary questions.
- Enable HyDE for short factoid questions unless they look numeric, comparative, or measurement-sensitive.
- Disable HyDE for yes/no questions.

## BioASQ-task14bPhaseB-testset1
- Enabled: 55
- Disabled: 25
  - 21: list question; HyDE can help the dense retriever move from terse query wording toward document-style category/example language.
  - 19: broad summary question; HyDE can help dense retrieval by converting the query into short abstract-like biomedical language.
  - 17: yes/no question; HyDE can drift toward a stance and is better kept off by default.
  - 15: short factoid question; HyDE can help the dense branch bridge wording mismatch without being too specific.
  - 8: factoid question but likely numeric/comparative/measurement-sensitive; HyDE is more likely to hallucinate or overcommit.

## 13b_golden_50q_sample.json
- Enabled: 33
- Disabled: 17
  - 16: yes/no question; HyDE can drift toward a stance and is better kept off by default.
  - 12: broad summary question; HyDE can help dense retrieval by converting the query into short abstract-like biomedical language.
  - 12: list question; HyDE can help the dense retriever move from terse query wording toward document-style category/example language.
  - 9: short factoid question; HyDE can help the dense branch bridge wording mismatch without being too specific.
  - 1: factoid question but likely numeric/comparative/measurement-sensitive; HyDE is more likely to hallucinate or overcommit.

## training14b_3pct_sample.json
- Enabled: 151
- Disabled: 41
  - 67: short factoid question; HyDE can help the dense branch bridge wording mismatch without being too specific.
  - 45: broad summary question; HyDE can help dense retrieval by converting the query into short abstract-like biomedical language.
  - 39: list question; HyDE can help the dense retriever move from terse query wording toward document-style category/example language.
  - 36: yes/no question; HyDE can drift toward a stance and is better kept off by default.
  - 4: factoid question but likely numeric/comparative/measurement-sensitive; HyDE is more likely to hallucinate or overcommit.
  - 1: long/specific factoid question; original query is already informative enough and HyDE adds more drift risk.

