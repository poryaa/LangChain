# prompts/rewrite.py

REWRITE_QUERY_PROMPT = """\
You are a semantic search optimizer for a resume vector database.
Convert the recruiter query into a short, dense search query that will retrieve
the most relevant resume chunks from a vector store.
Focus on skills, role titles, technologies, and domain keywords.
Remove conversational filler, ranking instructions, and quantity requests.

Recruiter query: {user_query}
Extracted filters (for context): {extracted_filters}

Return a short optimized search string only.
"""