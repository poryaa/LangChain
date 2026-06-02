GRADE_DOC_PROMPT = """
You are selecting which retrieved resume/CV chunks are relevant to a recruiter request.

Recruiter request:
{user_query}

Retrieval query:
{rewritten_query}

Candidates:
{candidates_text}

Instructions:
- Select ONLY candidates that are clearly relevant.
- Be strict.
- Return the exact candidate_id values of relevant candidates only.
- If none are relevant, return an empty list.
"""