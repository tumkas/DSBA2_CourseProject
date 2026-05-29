import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt

def load_data(base_dir):
    all_data = []
    folders = [f for f in os.listdir(base_dir) if f.startswith('training-')]
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        ref_file = os.path.join(folder_path, 'REFERENCE.csv')
        if os.path.exists(ref_file):
            df = pd.read_csv(ref_file, header=None, names=['filename', 'label'])
            df['filepath'] = df['filename'].apply(lambda x: os.path.join(folder_path, f"{x}.wav"))
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

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

def process_all_files(base_dir, output_dir, max_sec=5):
    df = load_data(base_dir)
    if df.empty:
        return pd.DataFrame()
        
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []
    
    for _, row in df.iterrows():
        try:
            y, sr = librosa.load(row['filepath'], sr=2000)
            y_filtered = apply_bandpass(y, sr)
            y_fixed = fix_length(y_filtered, sr, max_sec=max_sec)
            mel_spec = extract_mel_spectrogram(y_fixed, sr)
            
            save_path = os.path.join(output_dir, f"{row['filename']}.npy")
            np.save(save_path, mel_spec)
            
            processed_data.append({
                'filename': row['filename'],
                'label': row['label'],
                'npy_path': save_path
            })
        except Exception:
            pass
            
    return pd.DataFrame(processed_data)

base_directory = './dataset' 
output_directory = './processed_spectrograms'
final_df = process_all_files(base_directory, output_directory)

if not final_df.empty:
    final_df.to_csv('processed_data.csv', index=False)