# triage_agent.py

from agents import Agent
from resume_review_agent import build_resume_review_agent
from tenk_research_agent import build_tenk_research_agent
from aggregator_agent import build_aggregator_agent

TRIAGE_INSTRUCTIONS = """
You are the triage agent.

Routing rules (reply with *exactly* the agent name):

• If the message is only about résumés, CVs, candidates, or skills → 
  respond:  resume_review

• If the message is only about SEC filings, 10-K, MD&A, or risk factors → 
  respond:  tenk_research

• If the message touches *both* résumés AND company research in one query → 
  respond:  aggregator

• Otherwise respond:  none
""".strip()

def build_triage_agent() -> Agent:
    return Agent(
        name="triage",
        model="gpt-4o-mini",
        instructions=TRIAGE_INSTRUCTIONS,
        handoffs=[
            build_resume_review_agent(),
            build_tenk_research_agent(),
            build_aggregator_agent(),
        ],
    )
