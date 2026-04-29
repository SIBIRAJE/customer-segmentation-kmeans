"""
K-Means Clustering Model for Customer Segmentation
Handles model training, optimal K selection, and cluster analysis
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
import warnings

warnings.filterwarnings('ignore')


class KMeansTrainer:
    """
    K-Means clustering trainer for customer segmentation

    Attributes:
        X_scaled (ndarray): Scaled feature matrix
        original_data (DataFrame): Original data with customer info
        kmeans (KMeans): Trained KMeans model
        labels (ndarray): Cluster labels for each customer
    """

    def __init__(self, X_scaled, original_data):
        """
        Initialize the KMeansTrainer

        Args:
            X_scaled (ndarray): Scaled feature matrix
            original_data (DataFrame): Original customer data
        """
        self.X_scaled = X_scaled
        self.original_data = original_data.copy()
        self.kmeans = None
        self.labels = None
        self.inertias = []
        self.silhouette_scores = []
        self.calinski_scores = []

    def find_optimal_clusters(self, max_clusters=10):
        """
        Find optimal number of clusters using multiple metrics

        Args:
            max_clusters (int): Maximum number of clusters to test

        Returns:
            int: Optimal number of clusters
        """
        print("\n" + "=" * 60)
        print("🎯 FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 60)

        for k in range(2, max_clusters + 1):
            print(f"Testing k={k}...", end=" ")

            # Train KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)

            # Calculate metrics
            self.inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(self.X_scaled, kmeans.labels_)
            self.silhouette_scores.append(sil_score)

            calinski_score = calinski_harabasz_score(self.X_scaled, kmeans.labels_)
            self.calinski_scores.append(calinski_score)

            print(f"Inertia={kmeans.inertia_:.0f}, Silhouette={sil_score:.3f}")

        # Find elbow point
        kl = KneeLocator(range(2, max_clusters + 1), self.inertias,
                         curve='convex', direction='decreasing')
        optimal_k = kl.elbow if kl.elbow else 4

        # Alternative: use silhouette score to find best k
        best_silhouette_k = np.argmax(self.silhouette_scores) + 2

        print("\n" + "=" * 60)
        print("📊 OPTIMAL CLUSTER ANALYSIS")
        print("=" * 60)
        print(f"🎯 Elbow method suggests: {optimal_k} clusters")
        print(f"🎯 Silhouette method suggests: {best_silhouette_k} clusters")
        print(f"✅ Using: {optimal_k} clusters for final model")

        return optimal_k

    def plot_elbow_method(self, save_path='outputs/elbow_method.png'):
        """
        Plot elbow method graph and other metrics

        Args:
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        k_values = range(2, len(self.inertias) + 2)

        # Plot 1: Elbow Method
        axes[0, 0].plot(k_values, self.inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0, 0].set_ylabel('Inertia (WCSS)', fontsize=12)
        axes[0, 0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Mark the elbow
        kl = KneeLocator(k_values, self.inertias, curve='convex', direction='decreasing')
        if kl.elbow:
            axes[0, 0].axvline(x=kl.elbow, color='red', linestyle='--',
                               label=f'Elbow at k={kl.elbow}')
            axes[0, 0].legend()

        # Plot 2: Silhouette Scores
        axes[0, 1].plot(k_values, self.silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
        axes[0, 1].set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        best_k = np.argmax(self.silhouette_scores) + 2
        axes[0, 1].axvline(x=best_k, color='green', linestyle='--',
                           label=f'Best k={best_k} (Score={self.silhouette_scores[best_k - 2]:.3f})')
        axes[0, 1].legend()

        # Plot 3: Calinski-Harabasz Score
        axes[1, 0].plot(k_values, self.calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1, 0].set_ylabel('Calinski-Harabasz Score', fontsize=12)
        axes[1, 0].set_title('Calinski-Harabasz Index', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Combined Analysis
        axes[1, 1].plot(k_values, self.inertias / max(self.inertias), 'b-',
                        label='Normalized Inertia', linewidth=2)
        axes[1, 1].plot(k_values, self.silhouette_scores, 'r-',
                        label='Silhouette Score', linewidth=2)
        axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1, 1].set_ylabel('Normalized Score', fontsize=12)
        axes[1, 1].set_title('Combined Metrics Analysis', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ Elbow method plot saved to {save_path}")

    def train_model(self, n_clusters=4):
        """
        Train K-Means model with specified number of clusters

        Args:
            n_clusters (int): Number of clusters to create

        Returns:
            tuple: (trained_kmeans_model, cluster_labels)
        """
        print("\n" + "=" * 60)
        print(f"🤖 TRAINING K-MEANS MODEL with {n_clusters} clusters")
        print("=" * 60)

        # Train model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X_scaled)

        # Add cluster labels to original data
        self.original_data['Cluster'] = self.labels

        # Calculate model performance metrics
        sil_score = silhouette_score(self.X_scaled, self.labels)
        db_score = davies_bouldin_score(self.X_scaled, self.labels)
        ch_score = calinski_harabasz_score(self.X_scaled, self.labels)

        print(f"\n📊 MODEL PERFORMANCE METRICS:")
        print(f"   • Silhouette Score: {sil_score:.4f}")
        print(f"     (Range: -1 to 1, higher is better)")
        print(f"   • Davies-Bouldin Score: {db_score:.4f}")
        print(f"     (Lower is better, 0 is perfect)")
        print(f"   • Calinski-Harabasz Score: {ch_score:.2f}")
        print(f"     (Higher is better)")

        # Interpret silhouette score
        if sil_score > 0.5:
            print(f"   ✅ Good clustering structure detected")
        elif sil_score > 0.3:
            print(f"   📊 Moderate clustering structure")
        else:
            print(f"   ⚠️ Weak clustering structure - consider different features")

        return self.kmeans, self.labels

    def analyze_clusters(self, save_path='outputs/cluster_profiles.csv'):
        """
        Analyze and profile each cluster

        Args:
            save_path (str): Path to save cluster profiles CSV

        Returns:
            DataFrame: Cluster profiles with statistics
        """
        print("\n" + "=" * 60)
        print("📈 CLUSTER ANALYSIS & PROFILING")
        print("=" * 60)

        # Calculate cluster profiles
        cluster_profiles = self.original_data.groupby('Cluster').agg({
            'Annual_Income_k$': ['mean', 'std', 'min', 'max', 'median'],
            'Spending_Score': ['mean', 'std', 'min', 'max', 'median'],
            'CustomerID': 'count'
        }).round(2)

        # Flatten column names
        cluster_profiles.columns = [
            'Income_Mean', 'Income_Std', 'Income_Min', 'Income_Max', 'Income_Median',
            'Spending_Mean', 'Spending_Std', 'Spending_Min', 'Spending_Max', 'Spending_Median',
            'Customer_Count'
        ]

        # Calculate percentages
        cluster_profiles['Customer_Percentage'] = (
                cluster_profiles['Customer_Count'] / cluster_profiles['Customer_Count'].sum() * 100
        ).round(2)

        # Add cluster labels based on characteristics
        cluster_labels = []
        recommendations = []

        for idx in cluster_profiles.index:
            income = cluster_profiles.loc[idx, 'Income_Mean']
            spending = cluster_profiles.loc[idx, 'Spending_Mean']

            if income > 70 and spending > 70:
                label = "💎 VIP Premium Customers"
                rec = "Exclusive loyalty program, premium offers, early access"
            elif income > 70 and spending <= 70:
                label = "💰 High Income Conservative"
                rec = "Targeted promotions, showcase value, premium but value-focused"
            elif income <= 70 and spending > 70:
                label = "🎯 Value Seekers"
                rec = "Discount campaigns, bundle offers, installment plans"
            elif income <= 40 and spending <= 40:
                label = "📌 Budget Conscious"
                rec = "Essential products, heavy discounts, cashback offers"
            else:
                label = "🌟 Standard Customers"
                rec = "Engagement campaigns, personalized recommendations, cross-selling"

            cluster_labels.append(label)
            recommendations.append(rec)

        cluster_profiles['Cluster_Label'] = cluster_labels
        cluster_profiles['Business_Strategy'] = recommendations

        # Save to CSV
        cluster_profiles.to_csv(save_path)

        # Print cluster profiles
        print("\n📊 CLUSTER PROFILES SUMMARY:")
        print("-" * 80)

        for idx in cluster_profiles.index:
            print(f"\n🔹 Cluster {idx}: {cluster_profiles.loc[idx, 'Cluster_Label']}")
            print(f"   📍 Customers: {cluster_profiles.loc[idx, 'Customer_Count']} "
                  f"({cluster_profiles.loc[idx, 'Customer_Percentage']:.1f}%)")
            print(f"   💰 Average Income: ${cluster_profiles.loc[idx, 'Income_Mean']:.1f}k")
            print(f"   🛍️ Average Spending Score: {cluster_profiles.loc[idx, 'Spending_Mean']:.1f}")
            print(f"   💡 Strategy: {cluster_profiles.loc[idx, 'Business_Strategy']}")

        print(f"\n✅ Cluster profiles saved to {save_path}")

        return cluster_profiles

    def get_cluster_summary(self):
        """
        Get a quick summary of clusters

        Returns:
            dict: Summary statistics for each cluster
        """
        summary = {}

        for cluster in sorted(self.original_data['Cluster'].unique()):
            cluster_data = self.original_data[self.original_data['Cluster'] == cluster]

            summary[cluster] = {
                'size': len(cluster_data),
                'avg_income': cluster_data['Annual_Income_k$'].mean(),
                'avg_spending': cluster_data['Spending_Score'].mean(),
                'income_range': (cluster_data['Annual_Income_k$'].min(),
                                 cluster_data['Annual_Income_k$'].max()),
                'spending_range': (cluster_data['Spending_Score'].min(),
                                   cluster_data['Spending_Score'].max())
            }

        return summary


# For testing the module independently
if __name__ == "__main__":
    # This will run if the file is executed directly
    from data_preprocessing import DataPreprocessor

    # Test with sample data
    preprocessor = DataPreprocessor('../data/customer_data.csv')
    X_scaled, data = preprocessor.get_preprocessed_data()

    trainer = KMeansTrainer(X_scaled, data)
    optimal_k = trainer.find_optimal_clusters(max_clusters=8)
    trainer.plot_elbow_method()
    trainer.train_model(n_clusters=optimal_k)
    trainer.analyze_clusters()

    print("\n✅ Model training test completed successfully!")