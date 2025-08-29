import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="Multi-API Chatbot", layout="wide")
st.title("Multi-API Chatbot")

# api key management
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    google_api_key = "" 
    st.warning("Google API Key not found. Please add it to your secrets.")

try:
    openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    openrouter_api_key = "" 
    st.warning("OpenRouter API Key not found. Please add it to your secrets.")

@st.cache_resource
def load_chain(model_name, temp, max_tok, p, k):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Provide concise answers to the user's questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    if model_name == "Gemini":
        if not google_api_key:
            return None
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=google_api_key,
            temperature=temp,
            max_output_tokens=max_tok,
            top_p=p,
            top_k=k
        )
        return prompt | llm

    elif model_name == "Deepseek V3.1": 
        if not openrouter_api_key:
            return None
        
        llm = ChatOpenAI(
            model_name="deepseek/deepseek-chat-v3.1:free",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temp,
            max_tokens=max_tok,
            top_p=p
        )
        return prompt | llm
    return None

# sidebar management
with st.sidebar:
    st.header("Model Configuration")
    selected_model = st.selectbox(
        "Choose a model",
        ["Gemini", "Deepseek V3.1"] 
    )
    
    st.subheader("Advanced Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=4096, value=300, step=100)
    top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    
    is_openrouter = selected_model == "Deepseek V3.1" 
    top_k = st.slider(
        "Top-K (Gemini only)", 
        min_value=1, 
        max_value=100, 
        value=40, 
        step=1, 
        disabled=is_openrouter,
        help="This parameter is only supported by the Gemini model."
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    chain = load_chain(selected_model, temperature, max_tokens, top_p, top_k)

    if st.button("Summarize Chat", type="primary", use_container_width=True):
        if not chain:
             st.warning("Model not available. Cannot summarize.")
        elif "messages" not in st.session_state or len(st.session_state.messages) == 0:
            st.warning("No conversation to summarize.")
        else:
            with st.spinner("Summarizing..."):
                conversation_text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
                )
                
                summarization_prompt = f"Please provide a concise summary of the following conversation:\n\n{conversation_text}"
                
                llm = chain.last
                summary = llm.invoke(summarization_prompt)
                
                st.subheader("Chat Summary")
                st.info(summary.content)

# chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat management
if prompt_input := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    if chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))
                
                response_stream = chain.stream({
                    "chat_history": chat_history,
                    "input": prompt_input
                })
                full_response = st.write_stream(response_stream)
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Model is not available. Please check your API key settings.")
