import os
import joblib
import librosa
import numpy as np
from flask import Flask, request

# Define constants
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 10
TARGET_PITCH_SEMITONES = 4
N_MFCC = 13
WAVELET_LEVEL = 3
models = {}
WORD_MAP = {
    'bagel': 'g-d',
    'garlic': 'g-d',
    'grapefruit': 'g-d',
    'green_beans': 'g-d',
    'green_onion': 'g-d',
    'groceries': 'g-d',
    'guava': 'g-d',
    'gum': 'g-d',
    'mango': 'g-d',
    'yogurt': 'g-d',
    'air_conditioner': 'Gliding',
    'armchair': 'Gliding',
    'carpet': 'Gliding',
    'computer': 'Gliding',
    'corner': 'Gliding',
    'curtains': 'Gliding',
    'drawer': 'Gliding',
    'hanger': 'Gliding',
    'remote': 'Gliding',
    'wardrobe': 'Gliding',
    'alarm': 'lateralization',
    'charger': 'lateralization',
    'cutlery': 'lateralization',
    'door': 'lateralization',
    'garage': 'lateralization',
    'kitchenware': 'lateralization',
    'mirror_frame': 'lateralization',
    'refrigerator': 'lateralization',
    'roof': 'lateralization',
    'trash': 'lateralization',
    'chimpanzee': 'Lisping',
    'gazelle': 'Lisping',
    'hippopotamus': 'Lisping',
    'horse': 'Lisping',
    'lizard': 'Lisping',
    'otters': 'Lisping',
    'sloths': 'Lisping',
    'squirrel': 'Lisping',
    'swans': 'Lisping',
    'zebra': 'Lisping',
}


app = Flask(__name__)

def initialize():
    """
    Initialize the server by loading all the models in a dictionary
    """
    global models

    for type_ in os.listdir('models'):
        type_name = type_.split('_')[0]
        models[type_name] = joblib.load(f'models/{type_}')

# Preprocessing functions
def preprocess_audio(audio_wave, target_duration=0.8, sample_rate=16000):
    # Load the audio file
    # audio, sr = librosa.load(file_path, sr=sample_rate)
    audio, sr = audio_wave, sample_rate
    # Trim silence
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=20) 
    # Calculate the target length in samples
    target_length = int(target_duration * sample_rate) 
    # Pad or trim to the target length
    if len(trimmed_audio) > target_length:
        # Trim to target length
        processed_audio = trimmed_audio[:target_length]
    else:
        # Pad with zeros to match target length
         padding = target_length - len(trimmed_audio)
         processed_audio = np.pad(trimmed_audio, (0, padding), mode='constant')
    
    return processed_audio, sr

def pitch_normalization(processed_audio, sr, target_pitch=4):
    # Shift the pitch
    pitch_normalized_signal = librosa.effects.pitch_shift(processed_audio, sr=SAMPLE_RATE, n_steps=target_pitch)
    return pitch_normalized_signal

def extract_mfcc_features(audio_signal, sample_rate, frame_duration=0.025, hop_duration=0.01, n_mfcc=10):
    # Calculate frame size and hop length in samples
    frame_size = int(frame_duration * sample_rate)
    hop_length = int(hop_duration * sample_rate)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(
        y=audio_signal, 
        sr=sample_rate, 
        n_mfcc=n_mfcc, 
        n_fft=frame_size, 
        hop_length=hop_length
    )
    
    return mfccs  # Return raw MFCC features

def process_and_extract_features(audio):
    try:
        # Preprocess the audio
        processed_audio, sr = preprocess_audio(audio)
        if processed_audio is None or len(processed_audio) == 0:
            raise ValueError("Preprocessing returned empty audio!")

        # Normalize the pitch
        normalized_audio = pitch_normalization(processed_audio, sr)
        if normalized_audio is None or len(normalized_audio) == 0:
            raise ValueError("Pitch normalization returned empty audio!")

        # Extract MFCC features
        features = extract_mfcc_features(normalized_audio, sr)
        if features is None or len(features) == 0:
            raise ValueError("MFCC feature extraction failed!")

        return np.hstack(features)
    except Exception as e:
        print(f"Error in process_and_extract_features for {audio}: {e}")
        return None

@app.route('/')
def hello_world():
    return {'Message': "Hello! Model server is running"}

@app.route('/predict', methods=['POST'])
def predict():
    word = request.json['word'].lower()
    audio_file = np.array(request.json['audio'])

    if word not in WORD_MAP:
        return {'Status':'Failed', 'Error': f'Word not found in the model map: {word}'}, 400

    model = models[WORD_MAP[word]]

    features = process_and_extract_features(audio_file)

    prediction = model.predict(features.reshape(1, -1))

    return {'Status':'Success', 'Prediction': prediction[0]}



if __name__ == '__main__':
    initialize()
    app.run(port=5000, debug=True)