import streamlit as st
from sql_command_generator import generate_query

st.set_page_config(page_title="SQL Command Generator")

st.markdown("SQL Command Generator")
st.markdown("Ask your question:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Enter your question...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

        with st.spinner("Thinking..."):
            answer = generate_query(user_query)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
