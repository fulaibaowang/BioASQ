You are preparing a structured query bundle for a retrieval and reranking pipeline. Return valid JSON only.
Preserve the original fields of each question and only add one nested object called "query_parse".

Use this structure for each question:

{
  "body": "...",
   ...,
  "query_parse": {
    "hyde_enabled": true,
    "hyde_reason": "short reason",
    "hyde_text": "one short biomedical abstract-style passage, or empty string",
    "use_facets": false,
    "facet_reason": "short reason",
    "facets": [],
    "listwise_bullets": [],
  }
}

Task:
Given the question, decide:
1. whether to use HyDE for first-stage dense retrieval
2. whether to use facets for reranking
3. generate the corresponding text fields

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

Rules for facets:
- Use facets only if the question is clearly multi-aspect, broad, list-like, comparative, or has multiple answer slots.
- Do not use facets for simple single-focus factoid or straightforward yes/no questions unless there are clearly multiple independent aspects.
- Facet count:
    - Prefer 0 facets if the question is already single-focus.
    - Use 1-2 facets by default when facets are helpful.
    - Use 3 facets only if the question clearly has at least 3 distinct non-overlapping aspects.
    - Do not create filler facets just to reach 3.
    - If a facet is mostly a paraphrase of the original question or of another facet, omit it.
- Each facet must be short, self-contained, and directly usable as a reranking query.
- Keep the main entities from the original question explicit in every facet.
- Do not introduce unsupported new entities, categories, or answer guesses.
- If facets are not useful, set use_facets to false and return empty arrays.
- Do not rely on external information unless it is strongly implied and necessary to make a facet self-contained.

Rules for listwise_bullets:
- listwise_bullets must be short aspect labels derived from the facets.
- They are not full questions.
- If use_facets is false, return an empty array.

Update question-by-question. 

Question object:
{question_json}