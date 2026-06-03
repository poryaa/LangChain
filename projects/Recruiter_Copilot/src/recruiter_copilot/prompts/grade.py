# prompts/grade.py

GRADE_DOC_PROMPT = """\
You are a recruiter assistant. Your job is to filter retrieved resume chunks.

Recruiter query: {user_query}
Rewritten query: {rewritten_query}
Extracted filters: {extracted_filters}

For each candidate below, decide if they are RELEVANT to the query.
A candidate is relevant if their resume evidence matches the intent and filters.
Return only the candidate_ids that are clearly relevant.

--- CANDIDATES ---
{candidates_text}
--- END CANDIDATES ---
"""