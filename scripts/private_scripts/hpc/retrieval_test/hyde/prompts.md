# Gate
Decide whether this question should use HyDE for dense retrieval.

Usually enable for list, summary, and broad/short factoid questions.
Usually disable for yes/no, comparison, numeric, and very specific factoid questions.

Return JSON only:
{"hyde_enabled": ..., "hyde_reason": "..."}

Question object:
{question_json}

# Generate
Write one short biomedical abstract-style passage that could answer the question as a relevant paper might.
Use natural scientific wording and likely terminology.
If the question asks for a method, mention one or more plausible computational approaches or model types in general terms.
If the question asks for a list, mention several plausible categories or examples without trying to be exhaustive. For cause / mechanism / biomarker / treatment questions: State the likely mechanism, factor, biomarker class, or treatment type in general terms.
Keep it brief and avoid unsupported specific claims.

Question:
{question_body}