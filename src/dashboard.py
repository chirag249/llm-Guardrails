"""
Minimal SmartGuard Dashboard - Streamlit UI
Clean interface focused on Live Testing (Track A)
"""

import os
import streamlit as st
from classifier import PromptClassifier
from llm_backend import GuardedLLM

# Page config (simple, centered)
st.set_page_config(page_title="SmartGuard", page_icon="️", layout="centered")

# Initialize minimal session state
if "classifier" not in st.session_state:
    st.session_state.classifier = PromptClassifier(threshold=float(os.getenv('DEFAULT_THRESHOLD', '0.3')))
if "guarded_llm" not in st.session_state:
    st.session_state.guarded_llm = GuardedLLM(st.session_state.classifier)
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.title("️ SmartGuard")
st.write("A compact demo showing prompt safety classification (Track A). Enter a prompt and see if it is allowed.")

# Sidebar - minimal controls
st.sidebar.header("Settings")
st.sidebar.write(f"Provider: **{os.getenv('LLM_PROVIDER','gemini')}**")
st.sidebar.write(f"Model: **{os.getenv('LLM_MODEL','gemini-3-flash-preview')}**")

generate_llm = st.sidebar.checkbox("Generate LLM response", value=True)

# Cache info (if available)
classifier = st.session_state.classifier
cache_hits = getattr(classifier, 'cache_hits', 0)
cache_misses = getattr(classifier, 'cache_misses', 0)
cache_size = len(getattr(classifier, '_cache', {}))

st.sidebar.markdown("---")
st.sidebar.subheader("Cache")
st.sidebar.write(f"Hits: **{cache_hits}** | Misses: **{cache_misses}**")
st.sidebar.write(f"Size: **{cache_size}**")
if st.sidebar.button("Clear cache"):
    try:
        classifier._cache.clear()
        classifier.cache_hits = 0
        classifier.cache_misses = 0
        st.sidebar.success("Cache cleared")
    except Exception:
        st.sidebar.error("Failed to clear cache")

st.sidebar.markdown("---")
if st.sidebar.button("Clear history"):
    st.session_state.history = []
    st.experimental_rerun()

# Chat interface
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Render existing chat messages
for msg in st.session_state.chat_messages:
    role = msg.get('role', 'assistant')
    with st.chat_message(role):
        st.write(msg.get('content', ''))

# Chat input
user_input = st.chat_input("Type a message...")
if user_input:
    # Immediately render the user's message in the chat
    with st.chat_message("user"):
        st.write(user_input)

    # Append user message for persistence
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Classify
    with st.spinner("Classifying..."):
        result = st.session_state.classifier.classify(user_input)

    blocked = result['verdict'] == 'unsafe'
    if blocked:
        assistant_text = (
            f" Blocked — {result['category'].replace('_',' ').title()} (confidence: {result['confidence']:.0%})\n"
            "The prompt was blocked by SmartGuard."
        )
        # Render assistant reply immediately
        with st.chat_message("assistant"):
            st.write(assistant_text)
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
    else:
        # Allowed — call LLM if enabled
        if generate_llm:
            with st.spinner("Generating LLM response..."):
                try:
                    gen = st.session_state.guarded_llm.generate(user_input)
                    llm_resp = gen.get('response')
                    llm_err = gen.get('llm_error')
                except Exception as e:
                    llm_resp = None
                    llm_err = str(e)
            if llm_err:
                assistant_text = f"️ LLM error: {llm_err}"
            else:
                assistant_text = llm_resp or "(No response)"
        else:
            assistant_text = "(LLM generation disabled)"

        # Render assistant reply immediately
        with st.chat_message("assistant"):
            st.write(assistant_text)
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})

    # Record in history
    st.session_state.history.append({
        'prompt': user_input,
        'verdict': result['verdict'],
        'category': result['category'],
        'confidence': result['confidence']
    })
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]


# Footer
st.markdown("---")
st.caption("SmartGuard — Track A (pretrained classifier + keyword fallback).")
