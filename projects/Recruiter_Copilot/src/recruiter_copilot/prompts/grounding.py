# prompts/grounding.py

GROUNDING_CHECK_PROMPT = """
You are checking whether a recruiter assistant's answer is grounded and safe.

Recruiter query:
{user_query}

Answer:
{generated_answer}

Retrieved evidence:
{evidence}

Mark grounded = false if:
- The answer mentions any candidate ID not present in the evidence
- The answer mentions any resume file not present in the evidence
- The answer gives any employer, degree, skill, date, or location not explicitly supported by the evidence
- The answer includes sections not requested by the response_mode
- The answer applies discriminatory rules or unsupported filters

Be extremely strict.
Return:
- grounded: true or false
- reason: short explanation
"""