import streamlit as st
# Make sure chatbot.py is in the same folder
from chatbot import Session, handle_message 

st.set_page_config(
    page_title="Numerical Methods Bot",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CSS (Stays the same as your beautiful design) ---
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
    background: radial-gradient(ellipse at 20% 0%, #1a1a2e 0%, #0a0a0f 45%, #050508 100%);
    color: #FF0000;
}
.hero { text-align: center; padding: 1rem 0; }
.hero h1 {
    background: linear-gradient(135deg, #00d4ff, #7c3aed, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; margin-bottom: 2rem; }
.chip {
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.4);
    color: #c4b5fd;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
}
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 15px;
    margin-bottom: 10px;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    # We don't call handle_message here to avoid context errors on startup
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 I'm your Numerical Methods assistant. Select a method to begin!"}
    ]

if "chat_session" not in st.session_state:
    st.session_state.chat_session = Session()

# --- UI Header ---
st.markdown('<div class="hero"><h1>∫ Numerical Methods</h1></div>', unsafe_allow_html=True)
st.markdown('''
<div class="chip-row">
    <span class="chip">Lagrange</span> <span class="chip">Newton Forward</span> 
    <span class="chip">Newton Backward</span> <span class="chip">RK4</span> <span class="chip">Heun</span>
</div>
''', unsafe_allow_html=True)

# --- Sidebar / Reset ---
with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat reset. How can I help?"}]
        st.session_state.chat_session = Session()
        st.rerun()

# --- Display Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="c:\\Users\\Admin\\Downloads\\images.jpg" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# --- User Input Logic ---
if prompt := st.chat_input("Enter method name or data..."):
    # 1. Display User Message immediately
    with st.chat_message("user", avatar="c:\\Users\\Admin\\Downloads\\images.jpg"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get Logic Response
    # We pass the current logic session and get back the updated one
    reply, updated_logic_session = handle_message(prompt, st.session_state.chat_session)
    st.session_state.chat_session = updated_logic_session

    # 3. Display Assistant Response
    with st.chat_message("assistant", avatar="c:\\Users\\Admin\\Downloads\\images.jpg"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Note: st.rerun() is NOT needed here in Streamlit 1.20+ 
    # because the script finishes and the state is saved automatically.