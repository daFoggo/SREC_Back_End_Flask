import os
import logging
from pydub import AudioSegment
import shutil
import librosa
import numpy as np
from keras.models import load_model

class Models_Emotions_Fluency:

    def __init__(self, model_emotion, model_fluency):
        logging.info("Models_Emotions_Fluency")
        self.model_emotion = load_model(model_emotion)
        self.model_fluency = load_model(model_fluency)
        logging.info("Model emotion and fluency loaded!")

    # YouType to audio path and duration
    def audio_extraction(self, file_path):
        audio = file_path
        duration = librosa.get_duration(filename=file_path)
        return audio, duration

    # Audio to sub Audios
    def sub_audios_creation(self, audio_path, dur_cut, actual_dur):
        try:
            shutil.rmtree('sub_audios')
        except:
            x = 0
        os.mkdir('sub_audios')
        count = 0
        max = dur_cut * 1000
        while (count * dur_cut < actual_dur):  # 5*counter <
            newAudio = AudioSegment.from_wav(audio_path)
            newAudio = newAudio[count * max:(count + 1) * max]

            newAudio.export(f'sub_audios/new{count}.wav', format="wav")
            count += 1

        return f'sub_audios/'

    # Extracting Features of Emotions from an audio path
    def feature_extraction_Emotions(self, file_name):
        X, sample_rate = librosa.load(file_name, sr=None)  # Can also load file using librosa
        if X.ndim > 1:
            X = X[:, 0]
        X = X.T
        ## stFourier Transform
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=18).T, axis=0)  # Returns N_mel coefs
        rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)  # RMS Energy for each Frame (Stanford's). Returns 1 value
        spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)  # Spectral Flux (Stanford's). Returns 1 Value
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)  # Returns 1 value
        bw = np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate).T, axis=0)  # Returns 1 value
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)  # Returns 1 value
        ##Return computed audio features
        return mfccs, rmse, spectral_flux, spectral_centroid, zcr, bw

    # Extracting Features of Fluency from an audio path
    def feature_extraction_Fluency(self, file_name):
        X, sample_rate = librosa.load(file_name, sr=None)  # Can also load file using librosa
        if X.ndim > 1:
            X = X[:, 0]
        X = X.T
        ## stFourier Transform
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)  # Returns N_mel coefs
        rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)  # RMS Energy for each Frame (Stanford's). Returns 1 value
        spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)  # Spectral Flux (Stanford's). Returns 1 Value
        # print("spectral_flux: ",spectral_flux)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)  # Returns 1 value
        ##Return computed audio features
        return mfccs, rmse, spectral_flux, zcr

    # returning the Emotions Features
    def return_Input_Emotions(self, mfccs, rmse, spectral_flux, spectral_centroid, zcr, bw):
        extracted_features = np.hstack([mfccs, rmse, spectral_flux, spectral_centroid, zcr, bw])
        # print(extracted_features.shape)
        features = np.expand_dims(extracted_features, axis=1)
        # print(x.shape)
        features = np.expand_dims(features, axis=0)
        # print(x.shape)
        return features

    # returning the Fluency Features
    def return_Input_Fluency(self, mfccs, rmse, spectral_flux, zcr):
        extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
        # print(extracted_features.shape)
        features = np.expand_dims(extracted_features, axis=1)
        # print(x.shape)
        features = np.expand_dims(features, axis=0)
        # print(x.shape)
        return features

    # Predicting the Emotions of each sketch
    def Predict_sketch_Emotions(self, sketch_path):
        mfccs, rmse, spectral_flux, spectral_centroid, zcr, bw = self.feature_extraction_Emotions(f'sub_audios/{sketch_path}')
        input_features = self.return_Input_Emotions(mfccs, rmse, spectral_flux, spectral_centroid, zcr, bw)
        pred = self.model_emotion.predict(input_features)
        return np.argmax(pred)

    # Predicting the Fluency of each sketch
    def Predict_sketch_Fluency(self, sketch_path):
        mfccs, rmse, spectral_flux, zcr = self.feature_extraction_Fluency(f'sub_audios/{sketch_path}')
        input_features = self.return_Input_Fluency(mfccs, rmse, spectral_flux, zcr)
        pred = self.model_fluency.predict(input_features)
        return np.argmax(pred)

    # Predicting the Emotions and the Fluency
    def predict(self, file_path, cutdur):

        sketches_predictions_emotions = []
        sketches_predictions_fluency = []
        audio, duration = self.audio_extraction(file_path)

        print("duration = ", duration)

        sketches_path = self.sub_audios_creation(audio, cutdur, duration)

        for sketch_path in os.listdir(sketches_path):
            sketches_predictions_emotions.append(self.Predict_sketch_Emotions(sketch_path))
            sketches_predictions_fluency.append(self.Predict_sketch_Fluency(sketch_path))

        # getting the percentage of each emotion in the full Audio
        matching_count_emotions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for i in sketches_predictions_emotions:
            matching_count_emotions[i] = matching_count_emotions[i] + 1
        perc_list_emotions = [matching_count_emotions[i] / len(sketches_predictions_emotions) * 100 for i in range(7)]

        self.pred_emotion = {'sleepy': round(perc_list_emotions[0]),
                             'neutral': round(perc_list_emotions[1]),
                             'happy': round(perc_list_emotions[2]),
                             'angry': round(perc_list_emotions[3]),
                             'disgusted': round(perc_list_emotions[4]),
                             'fear': round(perc_list_emotions[5]),
                             'surprise': round(perc_list_emotions[6])}

        matching_count_fluency = {0: 0, 1: 0, 2: 0}
        for i in sketches_predictions_fluency:
            matching_count_fluency[i] = matching_count_fluency[i] + 1
        perc_list_fluency = [matching_count_fluency[i] / len(sketches_predictions_fluency) * 100 for i in range(3)]

        self.pred_fluency = {'Low': round(perc_list_fluency[0]),
                             'Intermediate': round(perc_list_fluency[1]),
                             'High': round(perc_list_fluency[2])}

        # print(f'sleepy {perc_list[0]}% ------- neutral {perc_list[1]}% -------- happy {perc_list[2]}% ----- angry {perc_list[3]}% --- disgusted {perc_list[4]}%---- fear {perc_list[5]}% ---- surprise {perc_list[6]}%')
        self.average_fluency = sum(sketches_predictions_fluency) / len(sketches_predictions_fluency)

        return self.pred_emotion, self.pred_fluency, self.average_fluency


def main():
    file_path = "../Voice-analysis-machine-learning/test.wav"  # replace with the actual path to your local audio file
    cutdur = 5
    model = Models_Emotions_Fluency('model_Emotions.h5', 'model_FLuency.h5')
    # model.audio_extraction('https://youtu.be/T8Zj1oLGaQE')
    pred_emotion, pred_fluency, average_fluency = model.predict(file_path, cutdur)
    logging.info(f"The emotions are as follows:\n {pred_emotion} \nThe Fluency are as follows:\n {pred_fluency} \nThe average fluency is: {average_fluency}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()