"""
Data Loading and Management Utilities
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle


class DatasetLoader:
    """
    Load and manage the poultry vocalization dataset.
    """
    
    def __init__(self, data_dir='./'):
        """
        Initialize dataset loader.
        
        Parameters:
        -----------
        data_dir : str
            Root directory containing Healthy, Unhealthy, and Noise folders
        """
        self.data_dir = Path(data_dir)
        self.healthy_dir = self.data_dir / 'Healthy'
        self.unhealthy_dir = self.data_dir / 'Unhealthy'
        self.noise_dir = self.data_dir / 'Noise'
        
        # Label mapping
        self.label_map = {
            'Healthy': 0,
            'Noise': 1,
            'Unhealthy': 2
        }
        
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def get_file_paths(self):
        """
        Get all audio file paths with labels.
        
        Returns:
        --------
        tuple
            (file_paths, labels, label_names)
        """
        file_paths = []
        labels = []
        label_names = []
        
        # Healthy files
        if self.healthy_dir.exists():
            healthy_files = sorted(self.healthy_dir.glob('*.wav'))
            file_paths.extend(healthy_files)
            labels.extend([self.label_map['Healthy']] * len(healthy_files))
            label_names.extend(['Healthy'] * len(healthy_files))
            print(f"Found {len(healthy_files)} Healthy files")
        
        # Noise files
        if self.noise_dir.exists():
            noise_files = sorted(self.noise_dir.glob('*.wav'))
            file_paths.extend(noise_files)
            labels.extend([self.label_map['Noise']] * len(noise_files))
            label_names.extend(['Noise'] * len(noise_files))
            print(f"Found {len(noise_files)} Noise files")
        
        # Unhealthy files
        if self.unhealthy_dir.exists():
            unhealthy_files = sorted(self.unhealthy_dir.glob('*.wav'))
            file_paths.extend(unhealthy_files)
            labels.extend([self.label_map['Unhealthy']] * len(unhealthy_files))
            label_names.extend(['Unhealthy'] * len(unhealthy_files))
            print(f"Found {len(unhealthy_files)} Unhealthy files")
        
        return file_paths, np.array(labels), label_names
    
    def split_dataset(self, file_paths, labels, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets.
        
        Parameters:
        -----------
        file_paths : list
            List of file paths
        labels : np.ndarray
            Array of labels
        test_size : float
            Proportion of test set
        val_size : float
            Proportion of validation set (from training set)
        random_state : int
            Random seed
            
        Returns:
        --------
        dict
            Dictionary with train, val, test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            file_paths, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\nDataset Split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        return {
            'train': {'paths': X_train, 'labels': y_train},
            'val': {'paths': X_val, 'labels': y_val},
            'test': {'paths': X_test, 'labels': y_test}
        }
    
    def get_class_distribution(self, labels):
        """
        Get class distribution.
        
        Parameters:
        -----------
        labels : np.ndarray
            Array of labels
            
        Returns:
        --------
        dict
            Class distribution
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {}
        
        for label, count in zip(unique, counts):
            class_name = self.reverse_label_map[label]
            distribution[class_name] = count
        
        return distribution
    
    def save_split(self, split_data, filepath):
        """
        Save dataset split to file.
        
        Parameters:
        -----------
        split_data : dict
            Split data dictionary
        filepath : str
            Path to save file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"Dataset split saved to {filepath}")
    
    def load_split(self, filepath):
        """
        Load dataset split from file.
        
        Parameters:
        -----------
        filepath : str
            Path to saved file
            
        Returns:
        --------
        dict
            Split data dictionary
        """
        with open(filepath, 'rb') as f:
            split_data = pickle.load(f)
        print(f"Dataset split loaded from {filepath}")
        return split_data
