"""
Prediction Script for Colab-Trained Model
Use this to predict with the model trained in Google Colab
"""

import sys
import pickle
import numpy as np
import librosa
from pathlib import Path
import xgboost as xgb

def extract_features(signal, sr=96000):
    """Extract 102 features from audio signal"""
    features = {}
    
    # MFCC features (80)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    for i in range(40):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])
    
    # Spectral features (8)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_std'] = np.std(zcr)
    
    # Power spectrum features (6)
    fft = np.fft.fft(signal)
    power = np.abs(fft) ** 2
    freqs = np.fft.fftfreq(len(signal))
    freqs_rad = 2 * np.pi * freqs
    
    positive_idx = freqs_rad >= 0
    freqs_rad = freqs_rad[positive_idx]
    power = power[positive_idx]
    
    target_mask = (freqs_rad >= 0.15) & (freqs_rad <= 0.25)
    features['peak_power_at_0.2rad'] = np.max(power[target_mask]) if np.any(target_mask) else 0
    features['mean_power_at_0.2rad'] = np.mean(power[target_mask]) if np.any(target_mask) else 0
    features['total_power'] = np.sum(power)
    features['low_freq_power'] = np.sum(power[freqs_rad < 0.1])
    features['mid_freq_power'] = np.sum(power[(freqs_rad >= 0.1) & (freqs_rad < 0.5)])
    features['high_freq_power'] = np.sum(power[freqs_rad >= 0.5])
    
    # Statistical features (8)
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['var'] = np.var(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    
    mean = np.mean(signal)
    std = np.std(signal)
    features['skewness'] = np.mean(((signal - mean) / std) ** 3) if std > 0 else 0
    features['kurtosis'] = np.mean(((signal - mean) / std) ** 4) if std > 0 else 0
    
    return np.array(list(features.values()))

def predict_audio(audio_path, model_path='poultry_health_model.pkl'):
    """
    Predict health status from audio file using Colab-trained model.
    
    Parameters:
    -----------
    audio_path : str
        Path to audio file
    model_path : str
        Path to downloaded model from Google Drive
    """
    
    print("="*70)
    print("POULTRY HEALTH PREDICTION")
    print("="*70)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        print("\nPlease download the model from Google Drive:")
        print("  Location: /content/drive/MyDrive/Poultry_Models/poultry_health_model.pkl")
        print(f"  Save it as: {model_path}")
        return None
    
    # Load model
    print(f"\nLoading model: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    class_names = model_data['class_names']
    model_type = model_data['model_type']
    test_accuracy = model_data['test_accuracy']
    
    print(f"Model Type: {model_type}")
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"\n❌ Audio file not found: {audio_path}")
        return None
    
    print(f"\nProcessing: {audio_path}")
    
    try:
        # Step 1: Load audio
        print("\n[1/4] Loading audio...")
        audio, sr = librosa.load(audio_path, sr=96000, mono=True, duration=10)
        duration = len(audio) / sr
        print(f"  ✓ Duration: {duration:.2f}s | Sample rate: {sr} Hz")
        
        # Step 2: Apply Hamming window
        print("\n[2/4] Applying Hamming window...")
        N = len(audio)
        n = np.arange(N)
        hamming_window = 0.54 + 0.46 * np.cos((2 * np.pi / N) * n)
        windowed = audio * hamming_window
        print("  ✓ Hamming window applied")
        
        # Step 3: Apply fast filtering
        print("\n[3/4] Applying filter...")
        alpha = 0.1
        filtered = np.zeros_like(windowed)
        filtered[0] = windowed[0]
        for i in range(1, len(windowed)):
            filtered[i] = alpha * windowed[i] + (1 - alpha) * filtered[i-1]
        print("  ✓ Signal filtered")
        
        # Step 4: Extract features
        print("\n[4/4] Extracting features and predicting...")
        features = extract_features(filtered, sr)
        print(f"  ✓ Extracted {len(features)} features")
        
        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        dmatrix = xgb.DMatrix(features_scaled)
        probabilities = model.predict(dmatrix)[0]
        prediction = probabilities.argmax()
        
        predicted_class = class_names[prediction]
        confidence = probabilities[prediction]
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        print(f"\n🎯 Predicted Class: {predicted_class}")
        print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print("\n📈 Class Probabilities:")
        for i, (name, prob) in enumerate(zip(class_names, probabilities)):
            bar_length = int(prob * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            marker = ' ← PREDICTED' if i == prediction else ''
            print(f"  {name:12s}: {prob:.4f} ({prob*100:5.2f}%) {bar}{marker}")
        
        # Interpretation
        print("\n💡 Interpretation:")
        if predicted_class == 'Healthy':
            print("  ✓ Normal, healthy vocalization detected")
            print("  No signs of respiratory distress")
        elif predicted_class == 'Unhealthy':
            print("  ⚠ WARNING: Signs of respiratory disease detected")
            print("  Indicators: Coughing, snoring, or rales")
            print("  → Recommendation: Veterinary examination advised")
        else:
            print("  Background noise detected")
            print("  No clear vocalization pattern")
        
        print("\n" + "="*70)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': dict(zip(class_names, probabilities))
        }
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("="*70)
        print("POULTRY HEALTH PREDICTION - Usage Instructions")
        print("="*70)
        print("\nUsage: python predict_from_colab_model.py <audio_file.wav> [model_path]")
        print("\nExamples:")
        print("  python predict_from_colab_model.py Healthy/1.wav")
        print("  python predict_from_colab_model.py Unhealthy/1.wav")
        print("  python predict_from_colab_model.py Noise/1.wav poultry_health_model.pkl")
        print("\nSetup:")
        print("  1. Download model from Google Drive:")
        print("     /content/drive/MyDrive/Poultry_Models/poultry_health_model.pkl")
        print("  2. Save it in the same directory as this script")
        print("  3. Run prediction on your audio files")
        print("\n" + "="*70)
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'poultry_health_model.pkl'
    
    result = predict_audio(audio_path, model_path)
    sys.exit(0 if result else 1)
