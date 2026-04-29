"""
Data Preprocessing Module for Customer Segmentation
Handles data loading, cleaning, and feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for customer segmentation

    Attributes:
        filepath (str): Path to the CSV data file
        data (DataFrame): Raw customer data
        scaler (StandardScaler): Scaler object for feature normalization
    """

    def __init__(self, filepath):
        """
        Initialize the DataPreprocessor

        Args:
            filepath (str): Path to the customer data CSV file
        """
        self.filepath = filepath
        self.data = None
        self.X = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.features = []

    def load_data(self):
        """
        Load customer data from CSV file

        Returns:
            DataFrame: Loaded customer data
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"✅ Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.filepath}")
            print("Please run generate_data.py first to create sample data")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise

    def explore_data(self):
        """
        Perform basic exploratory data analysis

        Returns:
            DataFrame: Descriptive statistics of the data
        """
        print("\n" + "=" * 60)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        print("\n📋 First 5 rows:")
        print(self.data.head())

        print("\n📋 Data Types:")
        print(self.data.dtypes)

        print("\n📋 Missing Values:")
        print(self.data.isnull().sum())

        print("\n📋 Descriptive Statistics:")
        stats = self.data.describe()
        print(stats)

        print("\n📋 Data Info:")
        self.data.info()

        return stats

    def clean_data(self):
        """
        Clean the data by handling missing values and outliers

        Returns:
            DataFrame: Cleaned customer data
        """
        print("\n" + "=" * 60)
        print("🧹 DATA CLEANING")
        print("=" * 60)

        initial_shape = self.data.shape

        # Remove duplicates
        self.data = self.data.drop_duplicates()
        duplicates_removed = initial_shape[0] - self.data.shape[0]
        print(f"✅ Removed {duplicates_removed} duplicate rows")

        # Handle missing values
        if self.data.isnull().sum().sum() > 0:
            print(f"⚠️ Found missing values: {self.data.isnull().sum().sum()}")
            for col in self.data.columns:
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                    print(f"   - Filled missing in {col} with median")
                else:
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                    print(f"   - Filled missing in {col} with mode")

        # Remove outliers using IQR method for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        outliers_removed = 0

        for col in numerical_cols:
            if col != 'CustomerID':  # Don't remove outliers from ID
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outliers_removed += len(outliers)
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)

        print(f"✅ Capped {outliers_removed} outlier values")
        print(f"✅ Final dataset shape: {self.data.shape}")

        return self.data

    def select_features(self, features=['Annual_Income_k$', 'Spending_Score']):
        """
        Select features for clustering

        Args:
            features (list): List of feature names to use for clustering

        Returns:
            DataFrame: Selected features
        """
        self.features = features
        self.X = self.data[features]
        print(f"\n✅ Selected features for clustering: {features}")
        return self.X

    def scale_features(self):
        """
        Standardize features using StandardScaler

        Returns:
            ndarray: Scaled feature matrix
        """
        print("\n" + "=" * 60)
        print("📏 FEATURE SCALING")
        print("=" * 60)

        self.X_scaled = self.scaler.fit_transform(self.X)
        print(f"✅ Features scaled successfully")
        print(f"   - Mean of scaled data: {self.X_scaled.mean():.2f}")
        print(f"   - Std of scaled data: {self.X_scaled.std():.2f}")

        return self.X_scaled

    def get_preprocessed_data(self):
        """
        Run complete preprocessing pipeline

        Returns:
            tuple: (scaled_features, original_data_with_cleaning)
        """
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.select_features()
        self.scale_features()

        return self.X_scaled, self.data


# For testing the module independently
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor('../data/customer_data.csv')
    X_scaled, cleaned_data = preprocessor.get_preprocessed_data()
    print("\n✅ Preprocessing test completed successfully!")