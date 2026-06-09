# prompts/grounding.py

GROUNDING_CHECK_PROMPT = """
You are checking whether a recruiter copilot answer is grounded in the provided evidence.

Your task is to verify whether the answer stays faithful to the EVIDENCE and follows shortlist behavior.

Recruiter query:
{user_query}

Generated answer:
{generated_answer}

Expected candidate count:
{expected_candidate_count}

EVIDENCE:
{evidence}

Mark grounded = false if ANY of the following happen:
- The answer mentions any candidate ID not present in the EVIDENCE
- The answer mentions any resume file not present in the EVIDENCE
- The answer includes more candidates than are supported by the EVIDENCE
- The answer invents or changes employer, degree, skill, language, date, location, or years of experience beyond what is explicitly supported
- The answer changes the ranking/order of the candidates from the EVIDENCE
- The answer adds unsupported summary claims that cannot be traced to the EVIDENCE
- The answer introduces discriminatory rules, exclusions, or unsafe filtering not supported by the recruiter query and evidence

Evaluation rules:
- Be extremely strict
- Treat missing evidence as unsupported
- If even one material claim is unsupported, mark grounded = false
- Focus on faithfulness to evidence, not writing style

Return:
- grounded: true or false
- reason: short explanation
"""