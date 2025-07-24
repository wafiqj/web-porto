import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

def home():
    st.title("Welcome 👋🏻")
    header = st.container()
    header.markdown("### *Get to Know More* Wafiq!")

    if 'pills_state' not in st.session_state:
        st.session_state.pills_state = [
            {"label": "Hello, how are you?", "clicked": False},
            {"label": "Who is Wafiq?", "clicked": False},
            {"label": "Where was he born?", "clicked": False},
            {"label": "How old is he?", "clicked": False}
        ]

    current_options_labels = [item["label"] for item in st.session_state.pills_state]
    selection = header.pills("", current_options_labels)

    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    
    # Custom CSS for the sticky header with dark mode support
    st.markdown(
        """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            background-color: var(--background-color);
            z-index: 999;
        }
        
        .fixed-header {
            border-bottom: 2px solid #cacaca;
        }
    </style>
        """,
        unsafe_allow_html=True
    )

    # --- Load Environment Variables ---
    load_dotenv()  # Load environment variables from .env file

    try:
        genai.configure(api_key=os.environ['NAMADAUN'])
        # Choose Gemini model. 'gemini-pro' is suitable for text.
        # There is also 'gemini-pro-vision' for multimodal.
        model = genai.GenerativeModel('gemini-2.5-flash')
    except KeyError:
        st.error("Google Gemini API key not found!")
        st.stop() # Stop execution if API key is missing

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    opening = st.chat_message("assistant")
    opening.write("Hi, I'm Wafiq Assistant. Is there anything you want to ask?")

    # --- Show Previous Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    def send_message(user_message):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        # Get response from Gemini model
        with st.chat_message("assistant"):
            with st.spinner("load.."):
                try:
                    # Send chat history for context
                    # Convert history format for Gemini model
                    history_for_gemini = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            history_for_gemini.append({"role": "user", "parts": [msg["content"]]})
                        elif msg["role"] == "assistant":
                            history_for_gemini.append({"role": "model", "parts": [msg["content"]]})
                    # Start chat with existing history
                    chat = model.start_chat(history=history_for_gemini[:-1]) # Send all except current user prompt
                    response = chat.send_message(user_message) # Send current user prompt
                    # Show response
                    st.markdown(response.text)
                    # Add AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

                except Exception as e:
                    st.error(f"An error occurred while processing the request: {e}")
                    st.markdown("Please try again or check your internet connection.")
    
    # --- Logic for Clicked Pills ---
    if selection:        
        # Send selected pill as user message
        for item in st.session_state.pills_state:
            if item["label"] == selection and not item["clicked"]:
                item["clicked"] = True
                send_message(selection)
                st.rerun()

    # --- User Input ---
    if prompt := st.chat_input("Type your message here..."):
        send_message(prompt)

pg = st.navigation([
    st.Page(home, title="Home", icon="🏠"),
    st.Page("about.py", title="About", icon="🧒🏻"),
    st.Page("project.py", title="Project", icon="💻"),
    st.Page("resume.py", title="Resume", icon="📄"),
    st.Page("contact.py", title="Contact", icon="📲")])

pg.run()