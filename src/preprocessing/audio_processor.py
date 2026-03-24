"""
Audio Signal Preprocessing Pipeline
Implements framing, windowing, and filtering as per the research paper
"""

import numpy as np
import librosa
import soundfile as sf
from .filters import HammingWindow, KalmanFilter, AdaptiveKalmanFilter


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline for poultry vocalization analysis.
    """
    
    def __init__(self, 
                 target_sr=96000,
                 frame_duration=20,
                 use_adaptive_kalman=True,
                 process_variance=1e-5,
                 measurement_variance=1e-2):
        """
        Initialize the audio preprocessor.
        
        Parameters:
        -----------
        target_sr : int
            Target sampling rate (96 kHz as per paper)
        frame_duration : int
            Frame duration in seconds (10-30 seconds as per paper)
        use_adaptive_kalman : bool
            Whether to use adaptive Kalman filter
        process_variance : float
            Kalman filter process noise covariance
        measurement_variance : float
            Kalman filter measurement noise covariance
        """
        self.target_sr = target_sr
        self.frame_duration = frame_duration
        self.use_adaptive_kalman = use_adaptive_kalman
        
        # Initialize filters
        if use_adaptive_kalman:
            self.kalman_filter = AdaptiveKalmanFilter(
                process_variance=process_variance,
                measurement_variance=measurement_variance
            )
        else:
            self.kalman_filter = KalmanFilter(
                process_variance=process_variance,
                measurement_variance=measurement_variance
            )
        
        self.hamming_window = HammingWindow()
    
    def load_audio(self, file_path):
        """
        Load audio file and resample if necessary.
        
        Parameters:
        -----------
        file_path : str
            Path to audio file
            
        Returns:
        --------
        tuple
            (audio_signal, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        return audio, sr
    
    def frame_signal(self, signal, sr):
        """
        Split signal into stationary frames.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input audio signal
        sr : int
            Sample rate
            
        Returns:
        --------
        list of np.ndarray
            List of signal frames
        """
        frame_length = int(self.frame_duration * sr)
        
        # Calculate number of frames
        num_frames = len(signal) // frame_length
        
        frames = []
        for i in range(num_frames):
            start_idx = i * frame_length
            end_idx = start_idx + frame_length
            frames.append(signal[start_idx:end_idx])
        
        # Handle remaining samples
        if len(signal) % frame_length != 0:
            frames.append(signal[num_frames * frame_length:])
        
        return frames
    
    def apply_hamming_window(self, frame):
        """
        Apply Hamming window to a frame.
        
        Parameters:
        -----------
        frame : np.ndarray
            Signal frame
            
        Returns:
        --------
        np.ndarray
            Windowed frame
        """
        return self.hamming_window.apply(frame)
    
    def apply_kalman_filter(self, signal):
        """
        Apply Kalman filter to denoise signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        # Reset filter state for each new signal
        self.kalman_filter.reset()
        return self.kalman_filter.filter(signal)
    
    def preprocess(self, file_path, apply_framing=True):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        file_path : str
            Path to audio file
        apply_framing : bool
            Whether to split into frames
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'raw': raw signal
            - 'windowed': windowed signal(s)
            - 'filtered': Kalman filtered signal(s)
            - 'sr': sample rate
            - 'frames': list of processed frames (if apply_framing=True)
        """
        # Load audio
        signal, sr = self.load_audio(file_path)
        
        result = {
            'raw': signal,
            'sr': sr
        }
        
        if apply_framing:
            # Frame the signal
            frames = self.frame_signal(signal, sr)
            
            processed_frames = []
            for frame in frames:
                # Apply Hamming window
                windowed = self.apply_hamming_window(frame)
                
                # Apply Kalman filter
                filtered = self.apply_kalman_filter(windowed)
                
                processed_frames.append({
                    'windowed': windowed,
                    'filtered': filtered
                })
            
            result['frames'] = processed_frames
            
            # Concatenate all filtered frames
            all_filtered = np.concatenate([f['filtered'] for f in processed_frames])
            result['filtered'] = all_filtered
            
        else:
            # Process entire signal
            windowed = self.apply_hamming_window(signal)
            filtered = self.apply_kalman_filter(windowed)
            
            result['windowed'] = windowed
            result['filtered'] = filtered
        
        return result
    
    def compute_power_spectrum(self, signal):
        """
        Compute power spectrum of the signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        tuple
            (frequencies, power_spectrum)
        """
        # Compute FFT
        fft = np.fft.fft(signal)
        
        # Compute power spectrum
        power = np.abs(fft) ** 2
        
        # Frequency bins (normalized)
        freqs = np.fft.fftfreq(len(signal))
        
        # Return only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        return positive_freqs, positive_power
