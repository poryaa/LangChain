#prompts/grounding.py

GROUNDING_CHECK_PROMPT = """
Check whether the answer is grounded in the retrieved evidence.

Answer:
{generated_answer}

Retrieved evidence:
{evidence}

Instructions:
- grounded = true only if the answer is supported by the evidence.
- grounded = false if the answer contains unsupported claims, invented skills, invented locations, invented experience, or overstates certainty.
- Be strict.

Return:
- grounded: true or false
- reason: short explanation
"""