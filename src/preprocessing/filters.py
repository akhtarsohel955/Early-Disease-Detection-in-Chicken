"""
Custom Signal Processing Filters
Implements Hamming Window and Kalman Filter as per Data in Brief 50 (2023) 109528
"""

import numpy as np


class HammingWindow:
    """
    Implements the Hamming window function to reduce spectral leakage.
    Formula: h(n) = 0.54 + 0.46 * cos((2π/N) * n) where 0 ≤ n ≤ N
    """
    
    @staticmethod
    def apply(signal, frame_length=None):
        """
        Apply Hamming window to the signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        frame_length : int, optional
            Length of the window. If None, uses signal length
            
        Returns:
        --------
        np.ndarray
            Windowed signal
        """
        N = frame_length if frame_length is not None else len(signal)
        n = np.arange(N)
        
        # Exact formula from the paper
        hamming_window = 0.54 + 0.46 * np.cos((2 * np.pi / N) * n)
        
        if len(signal) != N:
            # If signal is longer, apply window to each frame
            return signal[:N] * hamming_window
        
        return signal * hamming_window
    
    @staticmethod
    def create_window(N):
        """
        Create a Hamming window of length N.
        
        Parameters:
        -----------
        N : int
            Window length
            
        Returns:
        --------
        np.ndarray
            Hamming window coefficients
        """
        n = np.arange(N)
        return 0.54 + 0.46 * np.cos((2 * np.pi / N) * n)


class KalmanFilter:
    """
    Implements recursive Kalman filter for audio signal denoising.
    
    State Equations:
    - State Update: x_k = F * x_{k-1} + G * w_{k-1}
    - Measurement: y_k = H * x_k + v_k
    
    Algorithm Steps:
    1. Time Update: x̂_{k|k-1} = F * x̂_{k-1|k-1}
    2. Error Covariance: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
    3. Kalman Gain: K_k = P_{k|k-1} * H^T * [H * P_{k|k-1} * H^T + R]^{-1}
    4. Measurement Update: x̂_{k|k} = x̂_{k|k-1} + K_k * (y_k - H * x̂_{k|k-1})
    5. Error Covariance Update: P_{k|k} = (I - K_k * H) * P_{k|k-1}
    """
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        """
        Initialize Kalman Filter with tunable parameters.
        
        Parameters:
        -----------
        process_variance : float
            Process noise covariance (Q) - how much we trust the model
        measurement_variance : float
            Measurement noise covariance (R) - how much we trust the measurements
        """
        # State transition matrix (1D signal, simple model)
        self.F = np.array([[1.0]])
        
        # Measurement matrix (we observe the state directly)
        self.H = np.array([[1.0]])
        
        # Process noise covariance
        self.Q = np.array([[process_variance]])
        
        # Measurement noise covariance
        self.R = np.array([[measurement_variance]])
        
        # Initial state estimate
        self.x_hat = None
        
        # Initial error covariance
        self.P = np.array([[1.0]])
        
    def filter(self, signal):
        """
        Apply Kalman filter to the entire signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input noisy signal
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        filtered_signal = np.zeros_like(signal)
        
        # Initialize state with first measurement
        self.x_hat = np.array([[signal[0]]])
        
        for k in range(len(signal)):
            # Current measurement
            y_k = np.array([[signal[k]]])
            
            # Time Update (Prediction)
            x_hat_minus = self.F @ self.x_hat
            P_minus = self.F @ self.P @ self.F.T + self.Q
            
            # Measurement Update (Correction)
            # Kalman Gain
            S = self.H @ P_minus @ self.H.T + self.R
            K = P_minus @ self.H.T @ np.linalg.inv(S)
            
            # Update estimate with measurement
            innovation = y_k - self.H @ x_hat_minus
            self.x_hat = x_hat_minus + K @ innovation
            
            # Update error covariance
            I = np.eye(self.F.shape[0])
            self.P = (I - K @ self.H) @ P_minus
            
            # Store filtered value
            filtered_signal[k] = self.x_hat[0, 0]
        
        return filtered_signal
    
    def reset(self):
        """Reset the filter state."""
        self.x_hat = None
        self.P = np.array([[1.0]])


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Enhanced Kalman filter with adaptive noise estimation for better audio denoising.
    """
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2, 
                 adaptation_rate=0.01):
        super().__init__(process_variance, measurement_variance)
        self.adaptation_rate = adaptation_rate
        self.innovation_history = []
        
    def filter(self, signal):
        """
        Apply adaptive Kalman filter with online noise estimation.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input noisy signal
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        filtered_signal = np.zeros_like(signal)
        self.x_hat = np.array([[signal[0]]])
        
        for k in range(len(signal)):
            y_k = np.array([[signal[k]]])
            
            # Prediction
            x_hat_minus = self.F @ self.x_hat
            P_minus = self.F @ self.P @ self.F.T + self.Q
            
            # Kalman Gain
            S = self.H @ P_minus @ self.H.T + self.R
            K = P_minus @ self.H.T @ np.linalg.inv(S)
            
            # Innovation
            innovation = y_k - self.H @ x_hat_minus
            self.innovation_history.append(innovation[0, 0])
            
            # Adaptive R estimation (based on innovation variance)
            if len(self.innovation_history) > 50:
                recent_innovations = self.innovation_history[-50:]
                innovation_var = np.var(recent_innovations)
                self.R = (1 - self.adaptation_rate) * self.R + \
                         self.adaptation_rate * np.array([[innovation_var]])
            
            # Update
            self.x_hat = x_hat_minus + K @ innovation
            I = np.eye(self.F.shape[0])
            self.P = (I - K @ self.H) @ P_minus
            
            filtered_signal[k] = self.x_hat[0, 0]
        
        return filtered_signal
