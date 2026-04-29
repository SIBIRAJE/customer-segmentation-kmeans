"""
Visualization Module for Customer Segmentation Results
Creates various plots for cluster visualization and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Visualization class for cluster analysis results

    Attributes:
        data (DataFrame): Original data with cluster labels
        labels (ndarray): Cluster labels
        kmeans (KMeans): Trained KMeans model
    """

    def __init__(self, data, labels, kmeans_model):
        """
        Initialize the Visualizer

        Args:
            data (DataFrame): Data with cluster labels
            labels (ndarray): Cluster labels
            kmeans_model: Trained KMeans model
        """
        self.data = data
        self.labels = labels
        self.kmeans = kmeans_model
        self.n_clusters = len(np.unique(labels))

        # Create color palette
        self.colors = sns.color_palette("husl", n_colors=self.n_clusters)

    def plot_clusters_2d(self, save_path='outputs/clusters.png'):
        """
        Create 2D scatter plot of clusters

        Args:
            save_path (str): Path to save the plot
        """
        print("\n📊 Creating 2D cluster visualization...")

        fig, ax = plt.subplots(figsize=(14, 9))

        # Scatter plot for each cluster
        for cluster in range(self.n_clusters):
            mask = self.labels == cluster
            ax.scatter(
                self.data[mask]['Annual_Income_k$'],
                self.data[mask]['Spending_Score'],
                c=[self.colors[cluster]],
                label=f'Cluster {cluster}',
                alpha=0.6,
                s=120,
                edgecolors='black',
                linewidth=1.5
            )

        # Plot centroids
        centroids = self.kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c='red', marker='X', s=400,
            label='Centroids',
            edgecolors='black',
            linewidth=2.5,
            zorder=5
        )

        # Add cluster labels for centroids
        for i, centroid in enumerate(centroids):
            ax.annotate(f'C{i}',
                        (centroid[0], centroid[1]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=12,
                        fontweight='bold',
                        color='darkred')

        # Customize plot
        ax.set_xlabel('Annual Income (k$)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Spending Score (1-100)', fontsize=13, fontweight='bold')
        ax.set_title('Customer Segmentation using K-Means Clustering',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistical annotations
        textstr = f'Clusters: {self.n_clusters}\nCustomers: {len(self.data)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Cluster visualization saved to {save_path}")

    def plot_pca_clusters(self, save_path='outputs/pca_clusters.png'):
        """
        Plot clusters in reduced PCA space

        Args:
            save_path (str): Path to save the plot
        """
        print("\n📊 Creating PCA visualization...")

        # Apply PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.data[['Annual_Income_k$', 'Spending_Score']])

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1],
            c=self.labels, cmap='viridis',
            alpha=0.7, s=100, edgecolors='black', linewidth=1
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=11)

        # Add explained variance
        explained_variance = pca.explained_variance_ratio_
        ax.set_xlabel(f'First Principal Component ({explained_variance[0] * 100:.1f}% variance)',
                      fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Second Principal Component ({explained_variance[1] * 100:.1f}% variance)',
                      fontsize=12, fontweight='bold')
        ax.set_title('Clusters Visualized in PCA-Reduced Space',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ PCA visualization saved to {save_path}")
        print(f"   Total variance explained: {explained_variance.sum() * 100:.1f}%")

    def plot_cluster_centroids(self, save_path='outputs/centroids_analysis.png'):
        """
        Plot cluster centroids comparison

        Args:
            save_path (str): Path to save the plot
        """
        print("\n📊 Creating centroids analysis visualization...")

        # Get centroids
        centroids = self.kmeans.cluster_centers_

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot for income
        x = range(self.n_clusters)
        ax1.bar(x, centroids[:, 0], color=self.colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Annual Income (k$)', fontsize=12, fontweight='bold')
        ax1.set_title('Cluster Centers - Annual Income', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(centroids[:, 0]):
            ax1.text(i, v + 1, f'${v:.0f}k', ha='center', fontweight='bold')

        # Bar plot for spending
        ax2.bar(x, centroids[:, 1], color=self.colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Spending Score', fontsize=12, fontweight='bold')
        ax2.set_title('Cluster Centers - Spending Score', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(centroids[:, 1]):
            ax2.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Centroids analysis saved to {save_path}")

    def plot_distribution(self, save_path='outputs/cluster_distribution.png'):
        """
        Plot distribution of customers across clusters

        Args:
            save_path (str): Path to save the plot
        """
        print("\n📊 Creating cluster distribution visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Count plot
        cluster_counts = self.data['Cluster'].value_counts().sort_index()

        bars = ax1.bar(cluster_counts.index, cluster_counts.values,
                       color=self.colors, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, count in zip(bars, cluster_counts.values):
            percentage = (count / len(self.data)) * 100
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f'{count}\n({percentage:.1f}%)',
                     ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax1.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
        ax1.set_title('Customer Distribution Across Clusters', fontsize=13, fontweight='bold')
        ax1.set_xticks(cluster_counts.index)
        ax1.grid(True, alpha=0.3, axis='y')

        # Pie chart
        wedges, texts, autotexts = ax2.pie(cluster_counts.values,
                                           labels=[f'Cluster {i}' for i in cluster_counts.index],
                                           colors=self.colors,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           explode=[0.02] * self.n_clusters)

        # Style the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)

        ax2.set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Distribution plot saved to {save_path}")

    def plot_radar_chart(self, save_path='outputs/radar_chart.png'):
        """
        Create radar chart comparing cluster characteristics

        Args:
            save_path (str): Path to save the plot
        """
        print("\n📊 Creating radar chart visualization...")

        # Calculate cluster means
        cluster_means = self.data.groupby('Cluster')[['Annual_Income_k$', 'Spending_Score']].mean()

        # Normalize data (0-1 scale)
        normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

        # Prepare radar chart
        categories = ['Annual Income', 'Spending Score']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot each cluster
        for cluster in normalized.index:
            values = normalized.loc[cluster].values.tolist()
            values += values[:1]  # Close the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}', markersize=8)
            ax.fill(angles, values, alpha=0.15)

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['25%', '50%', '75%'], fontsize=10)
        ax.set_title('Cluster Characteristics Comparison (Radar Chart)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Radar chart saved to {save_path}")

    def plot_correlation_heatmap(self, save_path='outputs/correlation_heatmap.png'):
        """
        Plot correlation heatmap for the dataset

        Args:
            save_path (str): Path to save the plot
        """
        print("\n📊 Creating correlation heatmap...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate correlations
        corr_matrix = self.data[['Annual_Income_k$', 'Spending_Score']].corr()

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                    cmap='coolw')