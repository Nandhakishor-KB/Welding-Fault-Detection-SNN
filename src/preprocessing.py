import librosa
import numpy as np
import os

# Configuration based on your report
SAMPLE_RATE = 16000      # 16kHz sample rate [cite: 791]
TARGET_DURATION = 5      # 5 seconds audio clips [cite: 792]
N_MFCC = 13              # 13 MFCC features [cite: 793]

def process_audio(file_path):
    """
    Loads an audio file, pads/trims it to 5 seconds, and extracts MFCC features.
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Calculate target length in samples (16000 * 5 = 80000 samples)
        target_length = int(SAMPLE_RATE * TARGET_DURATION)
        
        # Pad or Trim audio to ensure consistent length [cite: 823-827]
        if len(audio) < target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            # Trim if too long
            audio = audio[:target_length]
            
        # Extract MFCC features [cite: 831]
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # Normalize the features [cite: 833]
        mfcc = librosa.util.normalize(mfcc)
        
        return mfcc
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    print("Preprocessing module loaded.")
