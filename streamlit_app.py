# streamlit_app.py
import asyncio, streamlit as st
from dotenv import load_dotenv
from agents import Runner

from triage_agent import build_triage_agent

load_dotenv()

st.set_page_config(page_title="AI Talent & 10-K Assistant", layout="wide")
st.title("🤖 Multi-Agent Demo")

# initialise router agent + chat history once per session
if "agent" not in st.session_state:
    st.session_state.agent = build_triage_agent()
if "history" not in st.session_state:
    st.session_state.history = []
# track the messages list for Runner.run
if "run_input" not in st.session_state:
    st.session_state.run_input = []

with st.form("chat", clear_on_submit=True):
    user_text = st.text_input("You:")
    send      = st.form_submit_button("Send")

if send and user_text:
    # display history
    st.session_state.history.append({"role": "user", "content": user_text})

    # build full context for this turn
    prev_msgs = st.session_state.run_input
    turn_input = prev_msgs + [{"role": "user", "content": user_text}]

    # run the agent
    with st.spinner("Thinking…"):
        res = asyncio.run(Runner.run(st.session_state.agent, turn_input))

    # save context for next turn
    st.session_state.run_input = res.to_input_list()

    # display assistant reply
    st.session_state.history.append({
        "role": "assistant",
        "content": res.final_output.strip()
    })

# render chat history
for m in st.session_state.history:
    who = "You" if m["role"] == "user" else "Assistant"
    st.markdown(f"**{who}:** {m['content']}")
