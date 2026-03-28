# Gate
Decide whether this biomedical question should use HyDE for dense retrieval.

General guidance:
- Usually enable for:
  - list questions
  - summary questions
  - broad definition/explanation questions
  - short factoid questions that ask for a concept, class, mechanism, syndrome, pathway, or method family
- Usually disable for:
  - yes/no questions
  - comparison questions
  - numeric, prevalence, dosage, age, date, or measurement-sensitive questions
  - very specific factoid questions asking for one exact entity, number, or highly constrained detail
  - questions where a hypothetical passage would likely overcommit or hallucinate

Return JSON only:
{"hyde_enabled": true/false, "hyde_reason": "brief reason"}

Question object:
{question_json}

# Generate
Write one short biomedical abstract-style passage that could answer the question as a relevant paper might.

Use the provided trusted evidence as the primary grounding source.
Use natural scientific wording and likely terminology.
If the question asks for a method, mention one or more plausible computational approaches or model types in general terms.
If the question asks for a list, mention several plausible categories or examples without trying to be exhaustive.
For cause / mechanism / biomarker / treatment questions, state the likely mechanism, factor, biomarker class, or treatment type in general terms.
Keep it brief and avoid unsupported specific claims.
Do not copy snippets verbatim.
Do not introduce concrete details that are not supported by the evidence.
Prefer a concise passage that captures the likely answer space and terminology useful for retrieval.

Question:
{question_body}