"""
Feature Extraction Module
Extracts relevant features from preprocessed audio signals
"""

import numpy as np
import librosa
from scipy import signal as scipy_signal


class FeatureExtractor:
    """
    Extract audio features for classification.
    """
    
    def __init__(self, sr=96000, n_mfcc=40, n_fft=2048, hop_length=512):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        sr : int
            Sample rate
        n_mfcc : int
            Number of MFCC coefficients
        n_fft : int
            FFT window size
        hop_length : int
            Hop length for STFT
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc(self, signal):
        """
        Extract MFCC features.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input audio signal
            
        Returns:
        --------
        np.ndarray
            MFCC features (n_mfcc x time_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_spectral_features(self, signal):
        """
        Extract spectral features.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input audio signal
            
        Returns:
        --------
        dict
            Dictionary of spectral features
        """
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=signal, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=signal, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=signal, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            signal, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        return features
    
    def extract_power_spectrum_features(self, signal):
        """
        Extract features from power spectrum.
        As per paper, unhealthy signals show spike around 0.2 radians/sample.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input audio signal
            
        Returns:
        --------
        dict
            Power spectrum features
        """
        # Compute FFT
        fft = np.fft.fft(signal)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(signal))
        
        # Only positive frequencies
        positive_idx = freqs >= 0
        freqs = freqs[positive_idx]
        power = power[positive_idx]
        
        # Normalize frequencies to radians/sample
        freqs_rad = 2 * np.pi * freqs
        
        # Find peak around 0.2 radians/sample (indicator of unhealthy)
        target_freq = 0.2
        tolerance = 0.05
        mask = (freqs_rad >= target_freq - tolerance) & (freqs_rad <= target_freq + tolerance)
        
        features = {
            'peak_power_at_0.2rad': np.max(power[mask]) if np.any(mask) else 0,
            'mean_power_at_0.2rad': np.mean(power[mask]) if np.any(mask) else 0,
            'total_power': np.sum(power),
            'low_freq_power': np.sum(power[freqs_rad < 0.1]),
            'mid_freq_power': np.sum(power[(freqs_rad >= 0.1) & (freqs_rad < 0.5)]),
            'high_freq_power': np.sum(power[freqs_rad >= 0.5]),
        }
        
        return features
    
    def extract_statistical_features(self, signal):
        """
        Extract statistical features from signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input audio signal
            
        Returns:
        --------
        dict
            Statistical features
        """
        features = {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'var': np.var(signal),
            'max': np.max(signal),
            'min': np.min(signal),
            'rms': np.sqrt(np.mean(signal**2)),
            'skewness': self._skewness(signal),
            'kurtosis': self._kurtosis(signal),
        }
        
        return features
    
    def _skewness(self, signal):
        """Calculate skewness."""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std) ** 3) if std > 0 else 0
    
    def _kurtosis(self, signal):
        """Calculate kurtosis."""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std) ** 4) if std > 0 else 0
    
    def extract_all_features(self, signal):
        """
        Extract all features from signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input audio signal
            
        Returns:
        --------
        np.ndarray
            Feature vector
        """
        # Check signal validity
        if len(signal) == 0:
            raise ValueError("Empty signal")
        
        if np.all(signal == 0):
            raise ValueError("Signal contains only zeros")
        
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or Inf values")
        
        feature_dict = {}
        
        try:
            # MFCC features
            mfcc = self.extract_mfcc(signal)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            for i in range(len(mfcc_mean)):
                feature_dict[f'mfcc_{i}_mean'] = mfcc_mean[i]
                feature_dict[f'mfcc_{i}_std'] = mfcc_std[i]
        except Exception as e:
            raise ValueError(f"MFCC extraction failed: {e}")
        
        try:
            # Spectral features
            spectral_features = self.extract_spectral_features(signal)
            for key, values in spectral_features.items():
                feature_dict[f'{key}_mean'] = np.mean(values)
                feature_dict[f'{key}_std'] = np.std(values)
        except Exception as e:
            raise ValueError(f"Spectral feature extraction failed: {e}")
        
        try:
            # Power spectrum features
            power_features = self.extract_power_spectrum_features(signal)
            feature_dict.update(power_features)
        except Exception as e:
            raise ValueError(f"Power spectrum extraction failed: {e}")
        
        try:
            # Statistical features
            stat_features = self.extract_statistical_features(signal)
            feature_dict.update(stat_features)
        except Exception as e:
            raise ValueError(f"Statistical feature extraction failed: {e}")
        
        # Convert to array
        feature_vector = np.array(list(feature_dict.values()))
        
        # Final validation
        if not np.isfinite(feature_vector).all():
            raise ValueError("Feature vector contains NaN or Inf values")
        
        return feature_vector, list(feature_dict.keys())
