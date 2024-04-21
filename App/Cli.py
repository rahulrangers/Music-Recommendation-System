import argparse
import joblib
import numpy as np
import pandas as pd
import librosa
from recommender import MusicRecommender

def recommend_music(song_name, n=10, model_path="recommender_model.pkl"):
    try:
        print("Loading model from {}...".format(model_path))
        recommender = MusicRecommender.load_model(filepath=model_path)
        
        print(f"Type of recommender after loading: {type(recommender)}")
        
        if recommender is None:
            print("Error: Failed to load the model.")
            return
        recommendations = recommender.get_top_recommendations(song_name, n)
        
        if recommendations is None:
            print("Error: Failed to get recommendations.")
            return
        
        for recommendation in recommendations:
            print(recommendation)
    except Exception as e:
        print(f"Error: {e}, {__file__}")



def classify_audio(file_path, model_path="classifier.pkl", encoder_path="encoder.pkl"):
    try:
        # Load classifier and encoder
        classifier = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        
        # Load audio data
        audio_data, sample_rate = librosa.load(file_path)
        audio_data = librosa.effects.trim(audio_data)[0]
        
        # Extract features
        features = []
        divided = sample_rate * 3
        for k in np.arange(0, len(audio_data) - divided, divided):
            lower = int(k)
            upper = int(k + divided)

            feature_row = extract_features(audio_data[lower:upper], sample_rate)
            features.append(np.array(feature_row))
        
        features = np.array(features)
        
        # Predict genre
        label = classifier.predict(features)
        
        # Inverse transform to get original genre name
        predicted_genre = encoder.inverse_transform([pd.DataFrame(label).mode()[0][0]])[0]
        
        print(f"Predicted genre for {file_path}: {predicted_genre}")
        
    except Exception as e:
        print(f"Error: {e}")

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
    tempo = librosa.beat.beat_track(y=audio, sr = sample_rate)[0].astype(np.float32)
    features.extend([tempo])

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1).astype(np.float32)
    mfcc_vars = np.var(mfccs, axis=1).astype(np.float32)

    features.extend(mfcc_means)
    features.extend(mfcc_vars)

    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Recommendation and Classification CLI")
    parser.add_argument("--recommend", type=str, help="Song name for recommendations")
    parser.add_argument("--classify", type=str, help="Path to audio file for classification")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations")
    
    args = parser.parse_args()

    if args.recommend:
        recommend_music(args.recommend, args.n)
    elif args.classify:
        classify_audio(args.classify)
    else:
        print("Please provide either --recommend or --classify argument.")
