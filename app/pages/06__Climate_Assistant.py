import streamlit as st
import sys
import json
import datetime
from pathlib import Path

# Ensure src is in the path to import rag_engine
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Climate Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
try:
    with open("app/static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHAT_HISTORY_DIR = BASE_DIR / "data" / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

st.title("🤖 Climate AI Assistant")
st.markdown(
    """
    <div style='background-color: #1E202B; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #4CAF50; margin-bottom: 2rem;'>
        <h4 style='margin-top:0; color:#4CAF50;'>Your Document-Powered Expert</h4>
        <p style='margin-bottom:0.5rem; color:#E0E0E0;'>
            Ask me anything about Pune's climate history, agricultural impacts, or historical weather events! 
        </p>
        <p style='margin-bottom:0; font-size: 0.9em; color:#A0A0A0;'>
            ⚡ Ask me about historical trends, agricultural impacts, or specific weather events using your own custom knowledge base.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
st.divider()

# Initialize the RAG Engine (cached so it doesn't reload embeddings every refresh)
@st.cache_resource(show_spinner=False)
def get_rag_engine():
    return RAGEngine()

try:
    engine = get_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize the RAG backend. Error: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Pune Climate Assistant. How can I help you today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Sidebar: Chat History Management ---
with st.sidebar:
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Pune Climate Assistant. How can I help you today?"}]
        st.rerun()
    
    st.markdown("### 🗂️ Recent Chats")
    
    # 2. Save Current Chat (Hidden behind an expander to keep UI clean)
    with st.expander("💾 Save Current Chat"):
        if len(st.session_state.messages) > 1:
            # Generate a short, human-readable title from the first user message
            first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "General Chat")
            title = " ".join(first_user_msg.split()[:4]) + "..." # e.g. "What happened in 2005..."
            
            # Use timestamp to ensure unique files, but save title inside the JSON data
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"chat_{timestamp}.json"
            filepath = CHAT_HISTORY_DIR / filename
            
            # We save a dictionary containing both the human-readable title and the raw messages
            save_data = {"title": title, "messages": st.session_state.messages}
            
            if st.button("Save Now"):
                with open(filepath, "w") as f:
                    json.dump(save_data, f)
                st.success("Saved!")
        else:
            st.warning("Start chatting first.")

    # 3. Load Previous Chats (Displayed as clean clickable buttons like Gemini)
    saved_files = sorted(CHAT_HISTORY_DIR.glob("*.json"), reverse=True)
    if not saved_files:
        st.caption("No saved chats yet.")
    else:
        for filepath in saved_files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    
                # Handle old JSON format vs new format
                if isinstance(data, list):
                    title = filepath.stem # Fallback for older saves
                    messages = data
                else:
                    title = data.get("title", filepath.stem)
                    messages = data.get("messages", [])
                
                # Render each history item as a subtle button
                if st.button(f"💬 {title}", key=filepath.name, use_container_width=True):
                    st.session_state.messages = messages
                    st.rerun()
            except Exception:
                pass # Skip corrupted files silently

# Handle User Input
if user_question := st.chat_input("Ask about Pune's climate..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate response from RAG Engine
    with st.chat_message("assistant"):
        with st.spinner("Thinking & searching documents..."):
            try:
                response = engine.query(user_question)
                st.markdown(response)
                # Add assistant response to state
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, an error occurred while processing your request: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
