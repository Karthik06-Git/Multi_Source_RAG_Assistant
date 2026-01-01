

import streamlit as st 
import requests 





####### Streamlit app

st.set_page_config(
        page_title="LLM-Powered Multi-URL RAG Chatbot",
        page_icon="🤖",
        layout="wide"
)


## Sidebar

st.sidebar.title("Documentation/Article URLs")
url1 = st.sidebar.text_input(label="URL-1", placeholder="Enter any URL")
url2 = st.sidebar.text_input(label="URL-2", placeholder="Enter any URL")
url3 = st.sidebar.text_input(label="URL-3", placeholder="Enter any URL")

urls = [url1, url2, url3]

process_urls_clicked = st.sidebar.button("Process URLs")

session_id = st.sidebar.text_input("Session ID", value="default_session")


backend_url = "https://karthik-nayanala-backend-fastapi-server.hf.space" 
                # "http://127.0.0.1:8000"


# Health Check for Backend-Server (makes sure only once)
if "backend_checked" not in st.session_state:
    try:
        health = requests.get(f"{backend_url}/", timeout=3)
        if health.status_code == 200:
            st.toast("Backend connected successfully!", icon="✅")
        else:
            st.toast("Backend running but not responding properly.", icon="⚠️")
    except:
        st.toast("Could not connect to backend. Make sure server is running.", icon="❌")

    # Mark as checked
    st.session_state.backend_checked = True




if process_urls_clicked:
    response = requests.post(
        url = "https://karthik-nayanala-backend-fastapi-server.hf.space/process_urls",      
                # "http://127.0.0.1:8000/process_urls",
        json = {"urls": urls}
    )

    if response.status_code == 200:         # if HTTP response returned by requests.post()
        response_json = response.json()
        is_processed, message = response_json["status"], response_json["message"]

        if is_processed:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)




def get_response(session_id: str, user_query: str):
    json_body = {
        "session_id": session_id,
        "user_query": user_query
    }

    response = requests.post(
        url = "https://karthik-nayanala-backend-fastapi-server.hf.space/chat_response",       
                # "http://127.0.0.1:8000/chat_response",
        json = json_body
    )

    if response.status_code == 200:         # if HTTP response returned by requests.post()
        response_json = response.json()
        return response_json["answer"]
    else:
        return "Error: Could not get response from backend-Server."
    






#### Main Interface
st.title("LLM-Powered Multi-URL RAG Chatbot")


# Chat History Display and Management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# for ensuring Old messages reappear on every rerun
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_query = st.chat_input(placeholder="Ask anything from the Documentations/Articles...")


if user_query:
    # Adding user message to UI history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    st.chat_message("user").write(user_query)

    response = get_response(session_id, user_query)

    # Adding assistant message to UI history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })
    st.chat_message("assistant").write(response)
else:
    st.warning("Please enter a query to retrieve the information from the Documentations/Articles.")





