import sys
import torch
import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from cnn import HeartCNN

def apply_bandpass(data, sr, lowcut=20.0, highcut=400.0, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def fix_length(data, sr, max_sec=5):
    max_len = sr * max_sec
    if len(data) > max_len:
        return data[:max_len]
    return np.pad(data, (0, max_len - len(data)), mode='constant')

def extract_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=800)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-9)

def predict_audio(file_path, model_path='heart_cnn_weights.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y, sr = librosa.load(file_path, sr=2000)
    y_filtered = apply_bandpass(y, sr)
    y_fixed = fix_length(y_filtered, sr)
    mel_spec = extract_mel_spectrogram(y_fixed, sr)
    
    mel_spec = np.expand_dims(mel_spec, axis=0)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    tensor_spec = torch.tensor(mel_spec, dtype=torch.float32).to(device)
    
    model = HeartCNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(tensor_spec)
        prob = torch.sigmoid(output).item()
        prediction = round(prob)
        
    return prediction, prob

if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_wav>")
    sys.exit(1)
    
audio_file = sys.argv[1]
pred, probability = predict_audio(audio_file)

if pred == 1:
    print(f"Abnormal (Probability of pathology: {probability:.4f})")
else:
    print(f"Normal (Probability of pathology: {probability:.4f})")