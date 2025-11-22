import streamlit as st
from agent import PreferenceAgent
from dotenv import load_dotenv # Added import

load_dotenv() # Added to load environment variables

st.title("AI Travel Planner")

# Initialize the agent and conversation history in Streamlit's session state
if "preference_agent" not in st.session_state:
    st.session_state.preference_agent = PreferenceAgent()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the travel planner! I\'m here to help you plan your next trip."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you plan your trip?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        response = st.session_state.preference_agent.run(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})