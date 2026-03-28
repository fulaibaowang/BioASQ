# Gate
Decide whether this question should use HyDE for dense retrieval.

Usually enable for list, summary, and broad/short factoid questions.
Usually disable for yes/no, comparison, numeric, and very specific factoid questions.

Return JSON only:
{"hyde_enabled": ..., "hyde_reason": "..."}

Question object:
{question_json}

# Generate
Write one short hypothetical passage that could answer the question.
Use document-like language.
Be cautious and general.
Do not add unsupported specific details.
If it is a list question, mention a few plausible categories or examples without being exhaustive.

Question:
{question_body}