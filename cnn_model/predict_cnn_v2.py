"""
Poultry Health CNN V2 — Prediction Script
==========================================
Matches the training notebook: cnn-training-kaggle.ipynb

Usage:
    python predict_cnn_v2.py <audio_file_or_folder>

Requires:
    - results/poultry_cnn_model_package_v2.pkl   (scaler + metadata)
    - results/poultry_cnn_model_v2.keras          (single best model)
    - results/poultry_cnn_ensemble_1.keras         (ensemble model 1)
    - results/poultry_cnn_ensemble_2.keras         (ensemble model 2)
    - results/poultry_cnn_ensemble_3.keras         (ensemble model 3)
"""

import os
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
import pickle
import warnings
import numpy as np
import librosa
import tensorflow as tf

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# ============================================================
# Paths — relative to this script's directory
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODEL_PACKAGE_PATH = os.path.join(RESULTS_DIR, "poultry_cnn_model_package_v2.pkl")
SINGLE_MODEL_PATH = os.path.join(RESULTS_DIR, "poultry_cnn_model_v2.keras")
ENSEMBLE_PATHS = [
    os.path.join(RESULTS_DIR, "poultry_cnn_ensemble_1.keras"),
    os.path.join(RESULTS_DIR, "poultry_cnn_ensemble_2.keras"),
    os.path.join(RESULTS_DIR, "poultry_cnn_ensemble_3.keras"),
]

SR = 96000  # Sample rate used in training
CLASS_NAMES = ["Healthy", "Noise", "Unhealthy"]


# ============================================================
# Feature Extraction — MUST match training notebook exactly
# ============================================================
def extract_features_v2(signal, sr=96000):
    """Extract 243 features with safe value ranges (matches training)."""
    features = {}
    if len(signal) < 2048:
        signal = np.pad(signal, (0, 2048 - len(signal)), mode="constant")

    # MFCC (80)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    for i in range(40):
        features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i}_std"] = np.std(mfcc[i])

    # Delta MFCC (80)
    delta_mfcc = librosa.feature.delta(mfcc)
    for i in range(40):
        features[f"delta_mfcc_{i}_mean"] = np.mean(delta_mfcc[i])
        features[f"delta_mfcc_{i}_std"] = np.std(delta_mfcc[i])

    # Chroma (24)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=2048, hop_length=512)
    for i in range(12):
        features[f"chroma_{i}_mean"] = np.mean(chroma[i])
        features[f"chroma_{i}_std"] = np.std(chroma[i])

    # Mel Spectrogram Stats (7)
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features["mel_spec_mean"] = np.mean(mel_spec_db)
    features["mel_spec_std"] = np.std(mel_spec_db)
    features["mel_spec_max"] = np.max(mel_spec_db)
    features["mel_spec_min"] = np.min(mel_spec_db)
    n_bands = mel_spec.shape[0]
    features["mel_low_energy"] = np.mean(mel_spec_db[: n_bands // 3, :])
    features["mel_mid_energy"] = np.mean(
        mel_spec_db[n_bands // 3 : 2 * n_bands // 3, :]
    )
    features["mel_high_energy"] = np.mean(mel_spec_db[2 * n_bands // 3 :, :])

    # Spectral Contrast (14)
    try:
        contrast = librosa.feature.spectral_contrast(
            y=signal, sr=sr, n_fft=2048, hop_length=512
        )
        for i in range(min(7, contrast.shape[0])):
            features[f"contrast_{i}_mean"] = np.mean(contrast[i])
            features[f"contrast_{i}_std"] = np.std(contrast[i])
    except Exception:
        for i in range(7):
            features[f"contrast_{i}_mean"] = 0.0
            features[f"contrast_{i}_std"] = 0.0

    # Tonnetz (12)
    try:
        harmonic = librosa.effects.harmonic(signal)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        for i in range(6):
            features[f"tonnetz_{i}_mean"] = np.mean(tonnetz[i])
            features[f"tonnetz_{i}_std"] = np.std(tonnetz[i])
    except Exception:
        for i in range(6):
            features[f"tonnetz_{i}_mean"] = 0.0
            features[f"tonnetz_{i}_std"] = 0.0

    # Spectral features (8)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_std"] = np.std(spectral_centroid)
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_rolloff_std"] = np.std(spectral_rolloff)
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)
    features["zero_crossing_rate_mean"] = np.mean(zcr)
    features["zero_crossing_rate_std"] = np.std(zcr)

    # Spectral Flatness (2)
    spec_flat = librosa.feature.spectral_flatness(y=signal)[0]
    features["spectral_flatness_mean"] = np.mean(spec_flat)
    features["spectral_flatness_std"] = np.std(spec_flat)

    # Spectral Entropy (1)
    power = np.abs(np.fft.fft(signal)) ** 2
    power = power[: len(power) // 2]
    power_sum = np.sum(power)
    if power_sum > 0:
        power_norm = power / power_sum
        features["spectral_entropy"] = -np.sum(
            power_norm * np.log2(power_norm + 1e-12)
        )
    else:
        features["spectral_entropy"] = 0.0

    # Autocorrelation (3)
    autocorr = librosa.autocorrelate(signal, max_size=min(5000, len(signal) // 2))
    if len(autocorr) > 1:
        autocorr_norm = autocorr / (autocorr[0] + 1e-10)
        end = min(100, len(autocorr_norm))
        features["autocorr_max"] = np.max(autocorr_norm[1:end])
        features["autocorr_mean"] = np.mean(autocorr_norm[1:end])
        features["autocorr_std"] = np.std(autocorr_norm[1:end])
    else:
        features["autocorr_max"] = 0.0
        features["autocorr_mean"] = 0.0
        features["autocorr_std"] = 0.0

    # Statistical features (8)
    features["sig_mean"] = np.mean(signal)
    features["sig_std"] = np.std(signal)
    features["sig_var"] = np.var(signal)
    features["sig_max"] = np.max(signal)
    features["sig_min"] = np.min(signal)
    features["sig_rms"] = np.sqrt(np.mean(signal**2))
    sig_std_safe = np.std(signal) + 1e-10
    features["sig_skewness"] = np.mean(
        ((signal - np.mean(signal)) / sig_std_safe) ** 3
    )
    features["sig_kurtosis"] = np.mean(
        ((signal - np.mean(signal)) / sig_std_safe) ** 4
    )

    # RMS Energy Stats (4)
    rms_frames = librosa.feature.rms(y=signal, frame_length=2048, hop_length=512)[0]
    features["rms_mean"] = np.mean(rms_frames)
    features["rms_std"] = np.std(rms_frames)
    features["rms_max"] = np.max(rms_frames)
    features["rms_min"] = np.min(rms_frames)

    # Convert and CLIP extreme values (CRITICAL — matches training)
    feat_array = np.array(list(features.values()), dtype=np.float32)
    feat_array = np.clip(feat_array, -1e6, 1e6)
    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=1e6, neginf=-1e6)
    return feat_array


# ============================================================
# Audio Preprocessing — MUST match training notebook exactly
# ============================================================
def preprocess_signal(audio, sr=96000):
    """Apply Hamming window + gentle low-pass filter (matches training)."""
    # Pad short audio to 2 seconds
    min_length = sr * 2
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)), mode="constant")

    N = len(audio)
    n = np.arange(N)
    hamming_window = 0.54 + 0.46 * np.cos((2 * np.pi / N) * n)
    windowed = audio * hamming_window

    # Gentle low-pass filter (alpha=0.3)
    alpha = 0.3
    filtered = np.zeros_like(windowed)
    filtered[0] = windowed[0]
    for i in range(1, len(windowed)):
        filtered[i] = alpha * windowed[i] + (1 - alpha) * filtered[i - 1]

    return filtered


# ============================================================
# Load model package (scaler + metadata)
# ============================================================
def load_model_package():
    """Load the pickled model package containing scaler and config."""
    if not os.path.exists(MODEL_PACKAGE_PATH):
        print(f"Error: Model package not found: {MODEL_PACKAGE_PATH}")
        print("   Make sure the 'results' folder with training outputs is present.")
        sys.exit(1)

    with open(MODEL_PACKAGE_PATH, "rb") as f:
        pkg = pickle.load(f)

    print(f"Model package loaded successfully")
    print(f"  Type:           {pkg.get('model_type', 'unknown')}")
    print(f"  Features:       {pkg.get('num_features', 'unknown')}")
    print(f"  Feature version:{pkg.get('feature_version', 'unknown')}")
    print(f"  Single acc:     {pkg.get('test_accuracy_single', 0) * 100:.2f}%")
    print(f"  Ensemble acc:   {pkg.get('test_accuracy_ensemble', 0) * 100:.2f}%")
    return pkg


# ============================================================
# Load trained models
# ============================================================
def load_models(use_ensemble=True):
    """Load trained .keras model(s)."""
    models = []

    if use_ensemble:
        for path in ENSEMBLE_PATHS:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path, compile=False)
                models.append(model)
                print(f"  Loaded: {os.path.basename(path)}")
            else:
                print(f"  Warning: Not found: {path}")

    if not models:
        # Fall back to single model
        if os.path.exists(SINGLE_MODEL_PATH):
            model = tf.keras.models.load_model(SINGLE_MODEL_PATH, compile=False)
            models.append(model)
            print(f"  Loaded single model: {os.path.basename(SINGLE_MODEL_PATH)}")
        else:
            print(f"Error: No model files found in {RESULTS_DIR}")
            sys.exit(1)

    print(f"  Total models loaded: {len(models)}")
    return models


# ============================================================
# Predict single audio file
# ============================================================
def predict_audio(audio_path, models, scaler, num_features):
    """
    Predict the health class of a single audio file.

    Returns:
        class_name (str), confidence (float), all_probabilities (dict)
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=SR, mono=True, duration=10)

        # Preprocess (Hamming window + filter) — matches training
        filtered = preprocess_signal(audio, sr)

        # Extract features — matches training
        features = extract_features_v2(filtered, sr)

        if len(features) != num_features:
            print(f"  Warning: Feature count mismatch: got {len(features)}, expected {num_features}")
            return None, None, None

        # Clip raw features — matches training
        features = np.clip(features, -1e6, 1e6)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale — matches training
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_scaled = np.clip(features_scaled, -10, 10).astype(np.float32)
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape for Conv1D: (1, num_features, 1)
        features_input = features_scaled.reshape(1, num_features, 1)

        # Predict with all models (ensemble averaging)
        all_preds = []
        for model in models:
            pred = model.predict(features_input, verbose=0)
            all_preds.append(pred)

        # Average predictions across ensemble
        avg_pred = np.mean(all_preds, axis=0)[0]  # shape: (3,)

        # Get class
        predicted_idx = np.argmax(avg_pred)
        class_name = CLASS_NAMES[predicted_idx]
        confidence = float(avg_pred[predicted_idx])

        # All probabilities
        probs = {CLASS_NAMES[i]: float(avg_pred[i]) for i in range(len(CLASS_NAMES))}

        return class_name, confidence, probs

    except Exception as e:
        print(f"  Error processing {audio_path}: {e}")
        return None, None, None


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_cnn_v2.py <audio_file_or_folder>")
        print("")
        print("Examples:")
        print("  python predict_cnn_v2.py sample.wav")
        print("  python predict_cnn_v2.py /path/to/audio/folder")
        sys.exit(1)

    target = sys.argv[1]
    use_ensemble = "--single" not in sys.argv  # default: use ensemble

    print("=" * 70)
    print("Poultry Health CNN V2 - Prediction System")
    print("=" * 70)

    # Load model package
    pkg = load_model_package()
    scaler = pkg["scaler"]
    num_features = pkg["num_features"]

    # Load models
    print("\nLoading models...")
    models = load_models(use_ensemble=use_ensemble)

    # Determine input type
    if os.path.isfile(target):
        audio_files = [target]
    elif os.path.isdir(target):
        audio_files = sorted(
            [
                os.path.join(target, f)
                for f in os.listdir(target)
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
            ]
        )
    else:
        print(f"Error: Path not found: {target}")
        sys.exit(1)

    if not audio_files:
        print(f"Error: No audio files found in: {target}")
        sys.exit(1)

    # Run predictions
    print(f"\n{'=' * 70}")
    print(f"Predicting {len(audio_files)} file(s)...")
    print(f"{'=' * 70}\n")

    results = {"Healthy": 0, "Noise": 0, "Unhealthy": 0, "Error": 0}

    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        class_name, confidence, probs = predict_audio(
            audio_path, models, scaler, num_features
        )

        if class_name is None:
            results["Error"] += 1
            continue

        results[class_name] += 1

        # Display result
        print(f"File: {filename}")
        print(f"  Prediction:  {class_name} ({confidence * 100:.1f}%)")
        print(f"  Healthy:     {probs['Healthy'] * 100:.1f}%")
        print(f"  Noise:       {probs['Noise'] * 100:.1f}%")
        print(f"  Unhealthy:   {probs['Unhealthy'] * 100:.1f}%")
        print()

    # Summary
    total = len(audio_files)
    print(f"{'=' * 70}")
    print(f"SUMMARY ({total} files)")
    print(f"{'=' * 70}")
    print(f"  Healthy:     {results['Healthy']:4d} ({results['Healthy'] / total * 100:.1f}%)")
    print(f"  Noise:       {results['Noise']:4d} ({results['Noise'] / total * 100:.1f}%)")
    print(f"  Unhealthy:   {results['Unhealthy']:4d} ({results['Unhealthy'] / total * 100:.1f}%)")
    if results["Error"] > 0:
        print(f"  Errors:      {results['Error']:4d}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
