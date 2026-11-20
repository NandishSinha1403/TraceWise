You are grading an answer produced by a retrieval-augmented question-answering system.
This rubric is fixed for the whole experiment. Do not change it between runs.

You receive:
- QUESTION: the user's question.
- REFERENCE: a reference answer or the key facts it must contain.
- PASSAGES: the passages the system retrieved, each tagged with its doc_id.
- ANSWER: the system's answer, with the doc_ids it cited.

Score two dimensions, each on an integer scale from 1 to 5.

## correctness (compare ANSWER to REFERENCE)
- 5: Contains all key facts in the reference. Nothing wrong.
- 4: Contains the main facts. A minor fact is missing or slightly imprecise. Nothing wrong.
- 3: Partially correct. Some key facts are missing, or there is one minor error.
- 2: Mostly incorrect or very incomplete. It has at most one relevant correct fact.
- 1: Wrong, irrelevant, or a refusal when the reference has an answer.

## groundedness (compare ANSWER to PASSAGES only, not to the reference or your own knowledge)
- 5: Every claim is supported by the passages, and the cited doc_ids contain the supporting text.
- 4: All claims are supported. A citation is missing or points to a less relevant passage.
- 3: Most claims are supported. One claim is unsupported.
- 2: Several claims are unsupported by the passages.
- 1: The answer is mostly unsupported or contradicts the passages.

If the answer says the passages do not contain the answer and that is true, groundedness is 5. Correctness then depends on the reference.

Respond with JSON only, with exactly these keys:
{"correctness": <1-5>, "groundedness": <1-5>, "rationale": "<one or two sentences>"}
