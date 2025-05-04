# main.py

import os
from dotenv import load_dotenv

# ─── 1) Load your .env before any Agents SDK imports ──────────────────
load_dotenv(override=True)

# ─── 2) Register your OpenAI key for both LLM calls and trace export ─
from agents import set_default_openai_key
set_default_openai_key(os.getenv("OPENAI_API_KEY"), use_for_tracing=True)

# ─── Optional: enable the tracing UI by default ──────────────────────
os.environ.setdefault("OPENAI_TRACE", "1")

import asyncio
from agents import Runner, trace
from triage_agent import build_triage_agent

async def main():
    triage = build_triage_agent()
    print("Type a query (or 'exit'):")

    # ─── Conversation memory buffer ──────────────────────────────────
    run_input: list[dict[str, str]] = []

    while True:
        user_input = input("User> ")
        if user_input.lower() in ("exit", "quit"):
            break

        # 1) Record the new user turn in the memory buffer
        run_input.append({"role": "user", "content": user_input})

        # 2) Wrap this turn in a trace span for UI grouping
        with trace("ChatTurn"):
            # 3) Invoke the agent with the full message history
            result = await Runner.run(triage, run_input)

        # 4) Extract and display the assistant’s reply
        assistant_msg = result.final_output.strip()
        print("Assistant>", assistant_msg)

        # 5) Update the memory buffer with assistant’s reply
        run_input = result.to_input_list()

if __name__ == "__main__":
    asyncio.run(main())
