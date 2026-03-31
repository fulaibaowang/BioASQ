# Gate
Decide whether this question should use HyDE for first-stage dense retrieval.

Usually enable HyDE.
Disable mainly for:
numeric, date, dosage, or measurement-sensitive questions
exact identifier or lookup questions where rewriting may blur the target

Return JSON only:
{"hyde_enabled": ..., "hyde_reason": "...", "hyde_style": "general" or "neutral"}

Question object:
{question_json}

# Generate
Write one short biomedical abstract-style passage that could answer the question as a relevant paper might. Use natural scientific wording and likely terminology. If the question asks for a method, mention one or more plausible computational approaches or model types in general terms. If the question asks for a list, mention several plausible categories or examples without trying to be exhaustive. For cause / mechanism / biomarker / treatment questions: State the likely mechanism, factor, biomarker class, or treatment type in general terms. For yes/no questions, state the relation or issue in neutral terms, without assuming the answer is yes or no. If there are plausible supporting or contrasting possibilities, mention them generally. Keep it brief and avoid unsupported specific claims.
so, update question-by-question using trustful online source, update hyde section

Question:
{question_body}
