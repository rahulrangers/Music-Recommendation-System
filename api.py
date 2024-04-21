from flask import Flask,jsonify
from flask import request
import pandas as pd
import numpy as np
import joblib
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import tempfile
import warnings
from lightgbm import LGBMClassifier
from flask_cors import CORS
warnings.filterwarnings("ignore")

df = pd.read_csv('features.csv')
X = df.drop(columns=['label','Unnamed: 0'])
le = LabelEncoder()
y = le.fit_transform(df['label'])
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Extracting the required features from the audio files
def extract_features(audio, sample_rate):
    features = []
    # Chroma Features
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_stft_mean = np.mean(chroma_stft).astype(np.float32)
    chroma_stft_var = np.var(chroma_stft).astype(np.float32)
    features.extend([chroma_stft_mean])
    features.extend([chroma_stft_var])

    # RMS
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms).astype(np.float32)
    rms_var = np.var(rms).astype(np.float32)
    features.extend([rms_mean])
    features.extend([rms_var])

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_mean = np.mean(spectral_centroid).astype(np.float32)
    spectral_centroid_var = np.var(spectral_centroid).astype(np.float32)
    features.extend([spectral_centroid_mean])
    features.extend([spectral_centroid_var])

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth).astype(np.float32)
    spectral_bandwidth_var = np.var(spectral_bandwidth).astype(np.float32)
    features.extend([spectral_bandwidth_mean])
    features.extend([spectral_bandwidth_var])

    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rolloff_mean = np.mean(rolloff).astype(np.float32)
    rolloff_var = np.var(rolloff).astype(np.float32)
    features.extend([rolloff_mean])
    features.extend([rolloff_var])

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate).astype(np.float32)
    zero_crossing_rate_var = np.var(zero_crossing_rate).astype(np.float32)
    features.extend([zero_crossing_rate_mean])
    features.extend([zero_crossing_rate_var])

    # Harmony and Perceptr
    harmony, perceptr = librosa.effects.hpss(audio)
    harmony_mean = np.mean(harmony).astype(np.float32)
    harmony_var = np.var(harmony).astype(np.float32)
    perceptr_mean = np.mean(perceptr).astype(np.float32)
    perceptr_var = np.var(perceptr).astype(np.float32)
    features.extend([harmony_mean])
    features.extend([harmony_var])
    features.extend([perceptr_mean])
    features.extend([perceptr_var])

    # Tempo
    tempo  = librosa.beat.beat_track(y=audio, sr = sample_rate)[0].astype(np.float32)
    features.extend([tempo])

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1).astype(np.float32)
    mfcc_vars = np.var(mfccs, axis=1).astype(np.float32)

    features.extend(mfcc_means)
    features.extend(mfcc_vars)

    return features

model = LGBMClassifier(max_depth=15, learning_rate=0.23, path_smooth=20, max_bin=90, verbose=-1, force_col_wise=True)
model.fit(X_train,y_train)
joblib.dump(model, 'classifier.pkl')
joblib.dump(le, 'encoder.pkl')
model_file = "classifier.pkl"
label_encoder = "encoder.pkl"
def Genre_Classifier(path):
    audio_data , sample_rate = librosa.load(path)
    audio_data = librosa.effects.trim(audio_data)[0]

    features = []
    divided = sample_rate * 3
    for k in np.arange(0,len(audio_data)-divided, divided):
        lower = int(k)
        upper = int(k +divided)

        feature_row = extract_features(audio_data[lower:upper], sample_rate)
        features.append(np.array(feature_row))

    features = pd.DataFrame(features)

    classifier = joblib.load(model_file)
    result = classifier.predict(features)

    decode = joblib.load(label_encoder)
    return decode.inverse_transform([pd.DataFrame(result).mode()[0][0]])[0]


app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])

def upload_file():
   
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Create a temporary file in the same directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        # Call the Genre_Classifier function with the temporary file path
        genre = Genre_Classifier(temp_path)

        # Delete the temporary file
        os.remove(temp_path)
        os.rmdir(temp_dir)

        return jsonify({'genre': genre}), 200

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)