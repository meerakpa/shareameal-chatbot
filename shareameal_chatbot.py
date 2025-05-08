import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

# set the openAPI key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# configure the streamlit pg
st.set_page_config(page_title="Share a Meal Chatbot", page_icon="🤝")

# configure the sidebar user interface
with st.sidebar:
    st.title("Share a Meal Chatbot")
    st.markdown("Ask about volunteering, donations, or how the organization works.")
    st.markdown("Built with LangChain + OpenAI GPT-4o.")

# load the webpages into doc
@st.cache_resource
def load_docs():
    urls = [
        "https://www.shareameal.net",
        "https://www.shareameal.net/copy-of-get-involved",
        "https://www.shareameal.net/group-information",
        "https://www.shareameal.net/donate",
        "https://www.shareameal.net/calendar-of-events",
        "https://www.shareameal.net/copy-of-about-1",
        "https://www.shareameal.net/our-volunteers",
        "https://www.shareameal.net/in-the-news",
        "https://www.shareameal.net/contact",
        "https://www.shareameal.net/sponsors-partnerships",
        "https://www.shareameal.net/usc"
    ]
    loader = WebBaseLoader(urls)
    return loader.load()

docs = load_docs()
docs_text = "\n\n".join(doc.page_content for doc in docs)

# set up the model and template for prompt
llm = ChatOpenAI(model="gpt-4o-mini")

prompt_template = PromptTemplate.from_template("""
    You are a Share a Meal representative. Based on the volunteer's question and the organization documents, provide an answer with reasons.

    Volunteer Question: {input}
    Organization Information: {docs}

    Your Answer:
""")

# start new chat if no prev message found
if "messages" not in st.session_state:
    st.session_state.messages = []

# display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# conversation/chat start
if user_input := st.chat_input("Ask a question about Share a Meal..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    # send the input through to get a response
    filled_prompt = prompt_template.invoke({"input": user_input, "docs": docs_text})
    response = llm.invoke(filled_prompt).content.strip()

    # add the new chat to the previous history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # display the AI response
    with st.chat_message("assistant"):
        st.markdown(response)
