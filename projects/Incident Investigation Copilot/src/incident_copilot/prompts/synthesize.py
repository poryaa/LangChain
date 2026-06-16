# src/incident_copilot/prompts/synthesize.py
from langchain_core.prompts import ChatPromptTemplate

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an SRE incident analyst. 
Given structured evidence from an incident investigation, produce a concise diagnostic report.
Be specific, grounded, and do not invent information beyond what is provided.
Format your response as:
## Incident Summary
## Root Cause Assessment
## Supporting Evidence
## Recommended Actions"""),
    ("human", """Investigate this incident:

**Incident ID:** {incident_id}
**Service:** {service}
**Severity:** {severity}

**Log Events:**
{log_events}

**Metrics at incident time:**
{metrics}

**Recent Deployments:**
{deployments}

**Runbook guidance:**
{runbook}

Write the diagnostic report.""")
])