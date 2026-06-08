# # prompts/grade.py

# GRADE_DOC_PROMPT = """
# You are a recruiter assistant filtering candidates by relevance.

# The recruiter wants up to {requested_k} candidates.

# Recruiter query:
# {user_query}

# Rewritten query:
# {rewritten_query}

# Structured filters (may be incomplete):
# {extracted_filters}

# Candidates:
# {candidates_text}

# Instructions:
# - Your goal is to select the BEST candidates for this request.
# - Prioritize candidates that match as many of these as possible:
#   - Company or group names (e.g., specific employers)
#   - Required degree and field
#   - Years of experience
#   - Location
#   - Core skills / roles
#   - Languages
# - It is OK if some optional preferences (like publications or secondary tools) are missing,
#   as long as the candidate is a strong match on the main requirements.
# - Be generous: keep candidates that partially match several key criteria.
# - Only exclude candidates that are clearly unrelated to the request.
# - Try to return up to {requested_k} relevant candidates. If fewer are clearly relevant, return fewer.
# - Return ONLY the `candidate_id` values.
# """