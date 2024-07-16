import os
import cv2
import subprocess
import logging
import cv2 as cv
import numpy as np 
from deepface import DeepFace
from pydub import AudioSegment
from keras.models import load_model
import speech_recognition as sr


from Emotion_Fluency_Model import *

face_classifier = cv2.CascadeClassifier(r'./static/virtual_interview_static/haarcascade_frontalface_default.xml')
classifier =load_model(r'./static/virtual_interview_static/confidence_measuring_ver4.keras')

model_emotion = './static/virtual_interview_static/model_Emotions.h5'
model_fluency = './static/virtual_interview_static/model_Fluency.h5'
model = Models_Emotions_Fluency(model_emotion,model_fluency)
logging.basicConfig(level=logging.INFO)

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

def video_prediction(img):
    emotion_labels = ['confident', 'unconfident']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    results = []
    for(x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+h]
        face_img = cv2.resize(face_img, (48, 48))  # Resize ảnh về kích thước 48x48
        face_img = np.reshape(face_img, [1, face_img.shape[0], face_img.shape[1], 1])
        face_img = face_img.astype('float32')
        face_img /= 255 

        prediction = classifier.predict(face_img)
        label = emotion_labels[int(round(prediction[0][0]))]
        face_roi = rgb_frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['emotion']

        results.append({
            'confidence': label,
            'emotion': emotion,
        })
    return results

def wav_prediction(file_path):
    mp4_to_wav(file_path)
    wav_path = "./speech.wav"
    pred_emotion, pred_fluency, average_fluency = model.predict(wav_path, cutdur=2)
    max_fluency = 2
    min_fluency = 0 
    normalized_fluency = (average_fluency - min_fluency) / (max_fluency - min_fluency) * 100
    pred_emotion, pred_fluency, normalized_fluency
    os.remove('./speech.mp3')
    os.remove('./speech.wav')
    return {
        'emotion_prediction': pred_emotion,
        'fluency_prediction': pred_fluency,
        'pronunciation_score': normalized_fluency
    }
    
def prediction(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    result = {}
    video_pred = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break   
        predictions = video_prediction(frame)
        timestamp = frame_count / fps
        for pred in predictions:
            video_pred.append({
                's': timestamp,
                'confidence': pred['confidence'],
                'emotion': pred['emotion']
            })
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    voice_pred = wav_prediction(video_path)
    result.update({
        'voice_prediction': voice_pred,
        'video_prediction': video_pred
    })
    return result