import os
import torch
import librosa
import streamlit as st
import soundfile as sf

from groq import Groq
from dotenv import dotenv_values
from audio_recorder_streamlit import audio_recorder
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Parse the Groq stream
def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# Loading fine-tuned WhisperAI
checkpoint_path = "results/checkpoint-405"
Whispermodel = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
Whisperprocessor = WhisperProcessor.from_pretrained("results")

# Setup the page config for streamlit
st.set_page_config(
    page_title="Your personal food assistant!",
    page_icon="🍝",
    layout="centered",
)

# Configuring the sidebar
with st.sidebar:
    st.header(" 🎤 Audio Recorder 🎤 ")
    recorded_audio = audio_recorder()                                        # Audio Input option to be placed at the sidebar

# Configuring the main body title & captions
st.title("Hello fellow Foodie!")
st.caption("All the best food recommendations you'll ever need in Singapore..")
st.caption("Click on the audio recorder in the left sidebar or type in your queries below to interact with me!")

# To get API keys from either os.env or st.secrets
try:
    secrets = dotenv_values(".env")                         # for local running
    GROQ_API_KEY = secrets["GROQ_API_KEY"]
except:
    secrets = st.secrets                                    # for running after deployment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

# Save the api_key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Saved initial messages and context of the chatbot
INITIAL_RESPONSE = secrets["INITIAL_RESPONSE"]
INITIAL_MSG = secrets["INITIAL_MSG"]
CHAT_CONTEXT = secrets["CHAT_CONTEXT"]

# Instantiating client using Groq
client = Groq()

# Initializing the chat history if present as streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": INITIAL_RESPONSE
         },
    ]

# Display chat history if available
for message in st.session_state.chat_history:
    with st.chat_message("role", avatar='😎'):
        st.markdown(message["content"])


# Getting user input
user_prompt = st.chat_input("Ask me")                                    # Typed Input

# If input was recorded via microphone, audio will be processed and transcribed using the fine-tuned WhisperAI model for SG locations
if recorded_audio:
    audio_file = "audio.wav"
    with open(audio_file, "wb") as f:
        f.write(recorded_audio)

    audio_data, sampling_rate = librosa.load(audio_file, sr=16000)  # sr=16000 to convert the sampling rate to 16000Hz

    # Use the model and processor to transcribe the audio:
    input_features = Whisperprocessor(
        audio_data, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    # Generate token ids using the model
    with torch.no_grad():  # Disable gradient tracking for inference
        predicted_ids = Whispermodel.generate(input_features)

    # Decode token ids to text
    transcription = Whisperprocessor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Print the transcription
    user_prompt = transcription[0]

# Processing of user input and getting response from Llama 3.1
# The user queries and assistant responses will be saved into chat history
if user_prompt:

    with st.chat_message("user", avatar="🍪"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt})               # Append user query into chat history

    # Get response from Llama 3.1
    messages = [
        {"role": "system", "content": CHAT_CONTEXT
         },
        {"role": "assistant", "content": INITIAL_MSG},
        *st.session_state.chat_history
    ]

    # Display chatbot response
    with st.chat_message("assistant", avatar='😎'):
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            stream=True  # to stream the message
        )
        response = st.write_stream(parse_groq_stream(stream))
        
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response})             # Append chatbot response into chat history