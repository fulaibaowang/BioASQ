You are preparing a structured query bundle for a retrieval and reranking pipeline. Return valid JSON only.
Preserve the original fields of each question and only add one nested object called "query_parse".

Use this structure for each question:

{
  "body": "...",
   ...,
  "query_parse": {
    "parsed_query": "minimal normalized query text",
    "parsed_query_reason": "short reason",
    "hyde_enabled": true,
    "hyde_reason": "short reason",
    "hyde_text": "one short biomedical abstract-style passage, or empty string"
  }
}

Task:
Given the question, decide:
1. whether to use HyDE for first-stage dense retrieval
2. generate a minimal parsed_query
3. generate the corresponding text fields

General principles:
- Preserve the original intent, target, polarity, specificity, and uncertainty of the question.
- Be conservative.
- Keep the main entities explicit.
- parsed_query should be minimal and should not answer the question.
- hyde_text may be a short answer-shaped biomedical passage when HyDE is enabled, but it must remain general, faithful, and retrieval-oriented.

Rules for parsed_query:
- Write a minimal normalized version of the original question for retrieval and reranking.
- Only make small edits when clearly helpful, such as:
  - acronym expansion
  - alias expansion
  - spelling correction
  - biomedical term normalization
  - symbol normalization
- Do not add answer content, extra subquestions, or semantic expansion.
- If normalization is not needed, use the original body unchanged.

Rules for parsed_query_reason:
- Briefly state what was normalized, or say that no normalization was needed.

Rules for HyDE:
- Usually enable HyDE when a short abstract-style passage is likely to improve dense retrieval without blurring the target.
- Disable mainly for:
  - numeric, date, dosage, or measurement-sensitive questions
  - exact identifier, exact lookup, or direct name-matching questions
  - questions where rewriting is likely to blur a very specific target

Rules for hyde_text:
- Write one short biomedical abstract-style passage that could help retrieve relevant papers.
- Use natural scientific wording and likely terminology.
- You may use reputable online information if available, but keep the passage general, concise, and faithful to the question.
- Do not invent unsupported specific facts, values, entities, or taxonomies.
- If the question asks for a method, mention one or more plausible method classes in general terms.
- If the question asks for a list, mention several plausible categories or examples without trying to be exhaustive.
- For cause / mechanism / biomarker / treatment questions, state likely classes or mechanisms in general terms.
- For yes/no or contrastive questions, describe the relation in neutral terms without assuming one side is correct.
- If HyDE is disabled, return an empty string for hyde_text.

Update question-by-question.

Question object:
{question_json}