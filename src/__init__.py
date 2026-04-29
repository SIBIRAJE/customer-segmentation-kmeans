"""
Customer Segmentation using K-Means Clustering
Author: Your Name
Date: 2024
"""

from .data_preprocessing import DataPreprocessor
from .train_model import KMeansTrainer
from .visualize_results import Visualizer

__all__ = ['DataPreprocessor', 'KMeansTrainer', 'Visualizer']

__version__ = '1.0.0'