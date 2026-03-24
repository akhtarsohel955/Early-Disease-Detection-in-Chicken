"""
Configuration file for the Poultry Health Classification System
"""

# Data Configuration
DATA_CONFIG = {
    'data_dir': './',
    'healthy_dir': 'Healthy',
    'unhealthy_dir': 'Unhealthy',
    'noise_dir': 'Noise',
    'target_sr': 96000,  # 96 kHz as per paper
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'frame_duration': 20,  # seconds (10-30 as per paper)
    'use_adaptive_kalman': True,
    'process_variance': 1e-5,  # Kalman filter Q matrix
    'measurement_variance': 1e-2,  # Kalman filter R matrix
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
}

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    },
    'mlp': {
        'hidden_layer_sizes': (256, 128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'cache_features': True,
}

# Output Configuration
OUTPUT_CONFIG = {
    'output_dir': 'outputs',
    'visualization_dir': 'outputs/visualizations',
    'model_dir': 'outputs/models',
    'cache_dir': 'outputs/cache',
}

# Class Labels
CLASS_LABELS = {
    0: 'Healthy',
    1: 'Noise',
    2: 'Unhealthy'
}
