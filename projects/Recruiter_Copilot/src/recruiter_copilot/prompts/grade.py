GRADE_DOC_PROMPT = """
You are grading whether a retrieved resume/CV chunk is relevant to a recruiter request.

Recruiter request:
{user_query}

Retrieval query:
{rewritten_query}

Resume/CV chunk:
{content}

Instructions:
- Mark relevant only if the chunk contains evidence that helps answer the recruiter request.
- Be strict.
- Reject chunks that only overlap on generic words such as analytics, development, learning, management, or AI but do not actually match the intended role, skills, or constraints.
- Do not guess missing evidence.

Return:
- is_relevant: true or false
- reason: short explanation
"""