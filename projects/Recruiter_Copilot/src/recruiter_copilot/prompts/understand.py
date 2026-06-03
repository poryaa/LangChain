# prompts/understand.py

UNDERSTAND_QUERY_PROMPT = """\
You are a recruiter copilot assistant. Analyze the recruiter's query and extract structured intent.

Recruiter query: {user_query}

Choose ONE intent from:
- candidate_search     → recruiter wants a ranked list / shortlist of candidates
- candidate_deep_dive  → recruiter wants detailed info on ONE specific candidate
- candidate_compare    → recruiter wants to compare TWO or more specific candidates
- pool_insight         → recruiter wants counts, statistics, or aggregate info about the pool

Also extract any filters mentioned:
- skills       : list of skills, technologies, certifications
- years_min    : minimum years of experience (integer or null)
- location     : city / country / remote (string or null)
- languages    : list of spoken/written languages
- role         : target job title or domain (string or null)
- candidate_ids: list of explicit candidate IDs or file names mentioned (empty list if none)

Return structured output only.
"""