import streamlit as st
from dotenv import load_dotenv
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
import os
from PIL import Image
import subprocess
from langchain_core.prompts import ChatPromptTemplate
import torch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp
import whisper

device = torch.device('cuda')

load_dotenv()
groq_api_key=os.environ.get('GROQ_API_KEY')


model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


output_dir="audio"
def download_audio(url):
    output_file = "audio/audio"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Deleting it...")
        os.remove(output_file)


    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location':r"C:\Users\tando\PycharmProjects\pythonProject18\ffmpeg\bin\ffmpeg.exe",
        'concurrent-fragments': 4,

    }
    audio_format="mp3"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    # After the download, run any additional optimizations (like re-encoding with faster settings)
    print("11111")
    audio_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]


    if audio_files:
        audio_file = os.path.join(output_dir, audio_files[0])
        print("audio mil gya")
    else:
        print("No audio file found.")
        return

    optimize_audio_file(audio_file, output_dir, audio_format)


def optimize_audio_file(audio_file, output_dir, audio_format="mp3"):

    # optimized_file = os.path.join(output_dir, f"optimized_audio.{audio_format}")
    optimized_file = os.path.abspath(os.path.join(output_dir, f"optimized_audio.{audio_format}"))
    # audio_file = os.path.abspath(audio_file)

    # print(optimized_file)

    ffmpeg_command = [
        'ffmpeg', '-i', audio_file,
        '-c:a', 'libmp3lame', '-q:a', '2', '-preset', 'fast',
        optimized_file
    ]

    try:
        # Run ffmpeg command
        subprocess.run(ffmpeg_command, check=True)
        print(f"Optimized audio saved as {optimized_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to optimize audio: {e}")

    # Optional: delete the original file after optimizing
    if os.path.exists(audio_file):
        os.remove(audio_file)

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']


def get_text(link):
    try:
        x=YouTubeTranscriptApi.get_transcript(f"{link}", languages=('en',))
        combined_text = ' '.join(segment['text'] for segment in x)
        return combined_text
    except TranscriptsDisabled:
        # If transcript is disabled, notify the user and fall back to Whisper
        print("Video doesn't have transcription available, wait until we extract...")
        download_audio(link)
        return transcribe_audio("audio/optimized_audio.mp3")



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: "
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # model_name = "google/flan-t5-small"
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = HuggingFaceHub(repo_id="facebook/rag-token-nq", model_kwargs={"temperature":0.5, "max_length":512})
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # llm = HuggingFaceEndpoint(repo_id="google/flan-t5-small",temperature=0,max_new_tokens=150)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is not None:

        response = st.session_state.conversation.invoke({'question': user_question})
        st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    youtube_icon = Image.open(r"youtube_icon.png")

    st.set_page_config(page_title="Chat with YouTube!", page_icon=youtube_icon)

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    youtube_icon_path = "youtube_icon.png"

    # Use st.write with HTML to create a custom header
    st.write(f"""
        <h1 style="display: flex; align-items: center;">
            Chat with YouTube! 
            <img src="https://static.vecteezy.com/system/resources/previews/018/930/572/non_2x/youtube-logo-youtube-icon-transparent-free-png.png" alt="YouTube" style="width:150px;">
        </h1>
        """, unsafe_allow_html=True)
    user_url = st.text_input("Enter link:")

    if st.button("Process"):
        with st.spinner("Processing"):
                # get pdf
            urll = user_url.split("=")[1]
            raw_text = get_text(urll)

                # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            print(text_chunks)

                # create vector store
            vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                    vectorstore)


    user_question = st.text_input("Ask Questions from YouTube Videos:")
    if user_question:
        handle_userinput(user_question)



if __name__ == '__main__':
    main()