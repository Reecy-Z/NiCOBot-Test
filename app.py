import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from langchain_experimental.tools import PythonREPLTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain import agents

from NiCOBot.agents import NiCOBot
from NiCOBot.frontend.streamlit_callback_handler import StreamlitCallbackHandlerChem
from NiCOBot.tools import *


os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
ss = st.session_state

tools = [   agents.load_tools(['wikipedia'])[0],
            PythonREPLTool(),
            ControlChemCheck(),
            Query2SMILES(),
            # Query2Reactions(),
            Query2ReactionsEmbedding(),
            # Query2CSV(),
            Query2Dataframe(),
            Query2ReactionSimilarity(),
            CheckEOrNu(),
            CheckCOBondStrength(),
            PaperAnalysis(),
            Image2SMILES(),
            TableExtractor(),
            JSON2SMILES(),
        ]

@st.cache_resource
def LLM_chain_response():

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    # memory = ConversationBufferMemory(memory_key="chat_history")
    agent = NiCOBot(
        # tools,
        model="gpt-4-turbo",
        temp=0.1,
        memory=memory,
    ).agent_executor

    return agent


tool_list = pd.Series(
   {f"✅ {t.name}":t.description for t in tools}
).reset_index()
tool_list.columns = ['Tool', 'Description']


icon = Image.open("assets/logo.png")
st.set_page_config(page_title="NiCOBot", page_icon=icon)

# Set width of sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 450px;
        max-width: 450px;
    }
    """,
    unsafe_allow_html=True,
)


# Session state
def on_api_key_change():
    api_key = ss.get("api_key") or os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key


# sidebar
with st.sidebar:
    NiCOBot_logo = Image.open("assets/GPT-NiCOBot.png")
    st.image(NiCOBot_logo)

    # Input OpenAI api key
    st.markdown("Input your OpenAI API key.")
    st.text_input(
        "OpenAI API key",
        type="password",
        key="api_key",
        on_change=on_api_key_change,
        label_visibility="collapsed",
    )

    # Display available tools
    st.markdown(f"# Available tools: {len(tools)}")
    st.dataframe(
       tool_list,
       use_container_width=True,
       hide_index=True,
       height=300

    )
    uploaded_file = st.file_uploader("Upload your file here", type=['pdf','png'])

col1, col2 = st.columns(2)

with col1:
    # 这里可以添加更多的元素或者逻辑来处理上传的文件
    if uploaded_file is not None:
        # 处理文件的逻辑
        st.write("File uploaded successfully!")

        bytes_data = uploaded_file.getvalue()

        # 定义文件要保存到的路径
        save_path = "/home/zhujingyuan/chatbot/pdf_uploads"
        file_name = uploaded_file.name
        complete_name = os.path.join(save_path, file_name)

        # 将文件内容写入服务器的文件系统
        with open(complete_name, "wb") as f:
            f.write(bytes_data)
        
        convert_pdf_to_jpg(complete_name, "pdf_output_images")
        st.success("File processed.")

print(st.session_state)
# Agent execution

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandlerChem(
            st.container(),
            max_thought_containers=4,
            collapse_completed_thoughts=False,
            output_placeholder=st.session_state,
        )
        # try:
        # TODO Modify this, not taking callbacks
        agent = LLM_chain_response()
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
        # except:
        #    st.write("Please input a valid OpenAI API key.")
