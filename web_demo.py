import time
import requests
import streamlit as st


def get_stream(text):
    for t in text:
        time.sleep(min(0.02, 3/len(text)))
        yield t


instruction = st.sidebar.text_input(
    label="模型Instruction",
    value="你是一个AI助手，请严格按照用户要求进行对话。"
)

max_length = st.sidebar.slider(
    label="max_length",
    min_value=0,
    max_value=2048,
    value=512,
    step=1
)

top_p = st.sidebar.slider(
    label="top_p",
    min_value=0.0,
    max_value=1.0,
    value=0.95,
    step=0.01
)

top_k = st.sidebar.slider(
    label="top_k",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)

temperature = st.sidebar.slider(
    label="temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01
)

st.title("Streamlit Web Chatbot")
with st.expander("ℹ️Disclamimer"):
    st.caption(
        f"""
        可选的模型包括: 
        qwen2-7b-instruct(默认), 
        glm-4-9b-chat
        """
    )
    
option = st.selectbox(
    label="选择模型",
    options=(
        "qwen2-7b-instruct",
        "glm-4-9b-chat"
    )
)
st.write("当前模型:", option)
    

if "model_name" not in st.session_state:
    st.session_state["model_name"] = "qwen2-7b-instruct"
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "max_messages" not in st.session_state:
    # 对用户输入和返回都计数
    # 对话轮数 = max_messages / 2
    st.session_state.max_messages = 40
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
   
if len(st.session_state.messages) >= st.session_state.max_messages:
    st.info(
        "注意：已经超过了对话的最大轮数。"
    )
else:
    if prompt := st.chat_input("你好，有什么可以帮助你的吗？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            params = {
                "instruction": instruction,
                "model_id": option
            }
            try:
                response = requests.get(
                    f"http://127.0.0.1:8000/chat/{prompt}",
                    params=params
                )
                result = eval(response.text)
                response = st.write_stream(
                    get_stream(result["result"]["response"])
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                print(e)
                print(st.session_state)
                st.session_state.max_messages = len(st.session_state.messages)
                rate_limit_message = """
                    Oops! Sorry, I can't talk now. Too many people have used
                    this service recently.
                """
                st.session_state.messages.append(
                    {"role": "assistant", "content": rate_limit_message}
                )
                st.rerun()
            
                