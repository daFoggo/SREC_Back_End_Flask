import numpy as np 
import pandas as pd
from getpass import getpass
import os
import subprocess
from numpy.linalg import norm
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import speech_recognition as sr
def mp4_to_wav(mp4_filename):
    
    # Define file paths
    mp3_filename = 'speech.mp3'
    wav_filename = 'speech.wav'

    # Convert mp4 to mp3
    command2mp3 = f"ffmpeg -i {mp4_filename} {mp3_filename}"
    result = subprocess.run(command2mp3, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting mp4 to mp3: {result.stderr}")
        return
    
    # Convert mp3 to wav
    command2wav = f"ffmpeg -i {mp3_filename} {wav_filename}"
    result = subprocess.run(command2wav, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting mp3 to wav: {result.stderr}")
        return

    # Check if the wav file is created successfully
    if not os.path.exists(wav_filename):
        print("wav file was not created successfully.")
        return

    # Recognize speech from the wav file
    r = sr.Recognizer()
    with sr.AudioFile(wav_filename) as source:
        audio = r.record(source, duration=120)
    
    try:
        print(r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return 



def answer_matching(file_path, core_answer):
    client = OpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
    client.api_key = os.environ["OPENAI_API_KEY"]
    if os.path.exists('./speech.wav'):
        os.remove('./speech.wav')
    if os.path.exists('./speech.mp3'):
        os.remove('./speech.mp3')

    mp4_to_wav(file_path)
    wav_path = "./speech.wav"

    audio_file = open(wav_path, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text",
        language = "en",
    )
    
    core_embedding = np.array(embeddings.embed_documents([core_answer])[0])
    answer_embedding = np.array(embeddings.embed_documents([transcription])[0])
    cosine_sim = np.dot(core_embedding, answer_embedding)/ (norm(core_embedding) * norm(answer_embedding))
    
    os.remove('./speech.mp3')
    os.remove('./speech.wav')
    return cosine_sim