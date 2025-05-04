# aggregator_agent.py

import json
from agents import Agent, Runner
from agents.tool import FunctionTool
from resume_review_agent import build_resume_review_agent
from tenk_research_agent import build_tenk_research_agent

class ResumeAgentTool(FunctionTool):
    def __init__(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Résumé query"}
            },
            "required": ["query"],
            "additionalProperties": False,
        }
        super().__init__(
            name="resume_agent",
            description="Fetch résumé info via resume_review agent.",
            params_json_schema=schema,
            on_invoke_tool=self.on_invoke_tool,
        )

    async def on_invoke_tool(self, _ctx, args: str) -> str:
        q = json.loads(args)["query"]
        r = await Runner.run(build_resume_review_agent(), q)
        return r.final_output

class TenkResearchTool(FunctionTool):
    def __init__(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "10-K research query"}
            },
            "required": ["query"],
            "additionalProperties": False,
        }
        super().__init__(
            name="tenk_research_agent",
            description="Fetch 10-K info via tenk_research agent.",
            params_json_schema=schema,
            on_invoke_tool=self.on_invoke_tool,
        )

    async def on_invoke_tool(self, _ctx, args: str) -> str:
        q = json.loads(args)["query"]
        r = await Runner.run(build_tenk_research_agent(), q)
        return r.final_output

AGG_INSTRUCTIONS = """
You are **Aggregator GPT**, able to combine résumé analysis and company research.

TOOLS:
 • `resume_agent` – use for résumé/CV/skill queries  
 • `tenk_research_agent` – use for SEC/10-K queries  

BEHAVIOR:
1. If a user asks only about résumés, call `resume_agent`.  
2. If only about SEC filings, call `tenk_research_agent`.  
3. If they ask about *both* in one question:
   - Call **both** tools  
   - Then return:
     ### Résumé Analysis  
     <output of resume_agent>  

     ### 10-K Research  
     <output of tenk_research_agent>  

4. Otherwise reply `none`.
""".strip()

def build_aggregator_agent() -> Agent:
    return Agent(
        name="aggregator",
        model="gpt-4o-mini",
        instructions=AGG_INSTRUCTIONS,
        tools=[ResumeAgentTool(), TenkResearchTool()],
    )
