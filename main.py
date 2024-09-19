from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

# Add these imports
from PIL import Image
import requests
from io import BytesIO
import torch

# Ensure torch uses CPU
torch.set_default_device('cpu')
# Set page configuration
st.set_page_config(page_title="Gustavo Gal AI Chat Assistant", layout="wide")

# Custom CSS for theming
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stImage{
        justify-content: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #2C2C2C;
        color: #FFFFFF;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    .user-description {
        font-size: 14px;
        line-height: 1.5;
        margin-top: 10px;
        text-align: center;
    }
    .profile-pic-container {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with user information
with st.sidebar:
    st.title("User Profile")
    
    # Profile picture
    profile_pic_url = "https://media.licdn.com/dms/image/v2/D4D03AQHReWmaW6VodA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1705443713499?e=1732147200&v=beta&t=8YUo5BphIhd-vGo5_WmR8D3Iehdyd9nW9uoi7-VID_Y"
    response = requests.get(profile_pic_url)
    img = Image.open(BytesIO(response.content))
    
    # Wrap the image in a centered container
    st.markdown('<div class="profile-pic-container">', unsafe_allow_html=True)
    st.image(img, width=200)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # User information
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>Gustavo Pedroso Gal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>gustavopgal@gmail.com</p>", unsafe_allow_html=True)
    
    # User description
    st.markdown("<p class='user-description'>Tech Lead & Data Analyst @ Ápis Company | Power BI | Azure | Excel | Data Engineering | Marketing</p>", unsafe_allow_html=True)

st.header("AI Chat Assistant")
st.subheader("Ask me anything about my professional experiences!")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)