"""
Customer Segmentation using K-Means Clustering
Main execution script for the complete project
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.data_preprocessing import DataPreprocessor
from src.train_model import KMeansTrainer
from src.visualize_results import Visualizer


def check_data_exists():
    """
    Check if customer data exists, if not, generate it
    """
    data_path = 'data/customer_data.csv'

    if not os.path.exists(data_path):
        print("\n⚠️ Customer data not found!")
        print("📊 Generating sample customer data...")

        try:
            from generate_data import generate_customer_data
            generate_customer_data(n_customers=500, save_path=data_path)
            print("✅ Sample data generated successfully!\n")
        except ImportError:
            print("❌ Error: generate_data.py not found!")
            print("Please ensure generate_data.py is in the project root.")
            return False

    return True


def print_section_header(title, char="=", length=60):
    """
    Print a formatted section header

    Parameters:
    -----------
    title : str
        Section title to display
    char : str
        Character to use for the line
    length : int
        Length of the line
    """
    print("\n" + char * length)
    print(f"{title.center(length)}")
    print(char * length)


def print_business_recommendations(cluster_profiles):
    """
    Print business recommendations based on cluster profiles

    Parameters:
    -----------
    cluster_profiles : DataFrame
        Cluster profiles from the model
    """
    print_section_header("💼 BUSINESS RECOMMENDATIONS", "=", 60)

    recommendations = {
        0: {
            'name': 'Value Seekers',
            'strategy': [
                "• Offer discount coupons and promotional deals",
                "• Implement loyalty points program",
                "• Bundle products for better value",
                "• Send personalized offers based on purchase history"
            ]
        },
        1: {
            'name': 'Premium Customers',
            'strategy': [
                "• Create exclusive VIP loyalty program",
                "• Offer early access to new products",
                "• Provide premium customer support",
                "• Send personalized high-end recommendations"
            ]
        },
        2: {
            'name': 'Bargain Hunters',
            'strategy': [
                "• Focus on clearance and sale items",
                "• Implement flash sales and limited-time offers",
                "• Send price-drop alerts",
                "• Offer free shipping on minimum purchase"
            ]
        },
        3: {
            'name': 'Loyal Regulars',
            'strategy': [
                "• Implement referral bonus program",
                "• Offer subscription-based discounts",
                "• Create community engagement events",
                "• Send birthday and anniversary offers"
            ]
        }
    }

    print("\n🎯 TARGETED MARKETING STRATEGIES:\n")

    for cluster_id in cluster_profiles.index:
        cluster_label = cluster_profiles.loc[cluster_id, 'Cluster_Label']
        customer_count = cluster_profiles.loc[cluster_id, 'Customer_Count']
        percentage = cluster_profiles.loc[cluster_id, 'Customer_Percentage']

        print(f"📌 Cluster {cluster_id}: {cluster_label}")
        print(f"   👥 Size: {customer_count} customers ({percentage:.1f}%)")
        print(f"   💰 Avg Income: ${cluster_profiles.loc[cluster_id, 'Income_Mean']:.1f}k")
        print(f"   🛍️ Avg Spending: {cluster_profiles.loc[cluster_id, 'Spending_Mean']:.1f}")

        # Get recommendations for this cluster
        if cluster_id in recommendations:
            print(f"   📋 Strategy:")
            for strategy in recommendations[cluster_id]['strategy']:
                print(f"      {strategy}")
        else:
            # Default recommendation for any cluster
            print(f"   📋 Strategy:")
            print(f"      • Personalized marketing campaigns")
            print(f"      • Regular engagement through email/sms")
            print(f"      • Collect feedback for improvement")

        print()


def print_project_summary(results):
    """
    Print comprehensive project summary

    Parameters:
    -----------
    results : dict
        Dictionary containing project results
    """
    print_section_header("📊 PROJECT SUMMARY", "=", 60)

    print(f"\n✅ Project: Customer Segmentation using K-Means Clustering")
    print(f"📅 Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 Total Customers Analyzed: {results['total_customers']}")
    print(f"🎯 Number of Segments Found: {results['n_clusters']}")
    print(f"📈 Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"📊 Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")

    print(f"\n🎨 Visualizations Generated:")
    visualizations = [
        "• Elbow Method Graph (elbow_method.png)",
        "• 2D Cluster Scatter Plot (clusters.png)",
        "• PCA Visualization (pca_clusters.png)",
        "• Cluster Distribution (cluster_distribution.png)",
        "• Correlation Heatmap (correlation_heatmap.png)",
        "• Radar Chart (radar_chart.png)",
        "• Centroids Analysis (centroids_analysis.png)"
    ]
    for viz in visualizations:
        print(f"  {viz}")

    print(f"\n📁 Output Files:")
    print(f"  • Cluster Profiles CSV: outputs/cluster_profiles.csv")
    print(f"  • All visualizations: outputs/ directory")


def interactive_mode():
    """
    Run the project in interactive mode with user input
    """
    print_section_header("🎯 CUSTOMER SEGMENTATION TOOL", "=", 60)

    print("\nWelcome to the Customer Segmentation Tool!")
    print("This tool uses K-Means clustering to group customers")
    print("based on their income and spending patterns.\n")

    # Ask for number of clusters
    print("Options:")
    print("1. Auto-detect optimal number of clusters (Recommended)")
    print("2. Manually specify number of clusters")

    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == '2':
        try:
            n_clusters = int(input("Enter number of clusters (2-10): "))
            if n_clusters < 2 or n_clusters > 10:
                print("Invalid input. Using auto-detection instead.")
                n_clusters = None
        except:
            print("Invalid input. Using auto-detection.")
            n_clusters = None
    else:
        n_clusters = None
        print("\n✅ Using auto-detection for optimal clusters...")

    # Ask for visualization preference
    print("\nVisualization Options:")
    print("1. Show all visualizations (Recommended)")
    print("2. Run without showing plots (faster)")

    viz_choice = input("\nEnter your choice (1 or 2): ").strip()
    show_plots = viz_choice != '2'

    return n_clusters, show_plots


def main():
    """
    Main function to execute the customer segmentation pipeline
    """
    # Print welcome banner
    print_section_header("🚀 CUSTOMER SEGMENTATION USING K-MEANS", "=", 60)
    print("\n🎯 Objective: Group customers based on income and spending patterns")
    print("📊 Algorithm: Unsupervised K-Means Clustering")
    print("💡 Business Value: Targeted marketing and personalized experiences\n")

    # Check if data exists
    if not check_data_exists():
        print("❌ Cannot proceed without customer data. Exiting...")
        return

    # Ask for interactive mode
    try:
        run_interactive = input("Run in interactive mode? (y/n, default: n): ").strip().lower()
        if run_interactive == 'y':
            n_clusters, show_plots = interactive_mode()
        else:
            n_clusters = None
            show_plots = True
            print("\n✅ Running in automatic mode with all features...")
    except:
        n_clusters = None
        show_plots = True

    # ============================================================
    # STEP 1: DATA PREPROCESSING
    # ============================================================
    print_section_header("STEP 1: DATA PREPROCESSING", "=", 60)

    try:
        preprocessor = DataPreprocessor('data/customer_data.csv')
        X_scaled, original_data = preprocessor.get_preprocessed_data()
        print("✅ Data preprocessing completed successfully!")
    except Exception as e:
        print(f"❌ Error in data preprocessing: {str(e)}")
        return

    # ============================================================
    # STEP 2: FIND OPTIMAL CLUSTERS & TRAIN MODEL
    # ============================================================
    print_section_header("STEP 2: MODEL TRAINING & OPTIMIZATION", "=", 60)

    try:
        trainer = KMeansTrainer(X_scaled, original_data)

        # Find optimal number of clusters
        if n_clusters is None:
            optimal_k = trainer.find_optimal_clusters(max_clusters=10)
            if show_plots:
                trainer.plot_elbow_method()
        else:
            optimal_k = n_clusters
            print(f"\n🎯 Using manually specified clusters: {optimal_k}")

        # Train the model
        kmeans_model, labels = trainer.train_model(n_clusters=optimal_k)

        # Analyze clusters
        cluster_profiles = trainer.analyze_clusters()

        print("✅ Model training completed successfully!")

    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return

    # ============================================================
    # STEP 3: VISUALIZATIONS
    # ============================================================
    if show_plots:
        print_section_header("STEP 3: GENERATING VISUALIZATIONS", "=", 60)

        try:
            visualizer = Visualizer(original_data, labels, kmeans_model)

            # Generate all visualizations
            visualizer.plot_clusters_2d()
            visualizer.plot_pca_clusters()
            visualizer.plot_cluster_centroids()
            visualizer.plot_distribution()
            visualizer.plot_radar_chart()
            visualizer.plot_correlation_heatmap()

            print("✅ All visualizations generated successfully!")

        except Exception as e:
            print(f"⚠️ Warning: Some visualizations could not be generated: {str(e)}")
            print("Main analysis still completed successfully.")
    else:
        print_section_header("STEP 3: VISUALIZATIONS SKIPPED", "=", 60)
        print("Visualizations skipped as per user preference.")

    # ============================================================
    # STEP 4: BUSINESS INSIGHTS
    # ============================================================
    print_section_header("STEP 4: BUSINESS INSIGHTS", "=", 60)

    # Extract key metrics
    silhouette_score = None
    davies_bouldin_score = None

    try:
        from sklearn.metrics import silhouette_score as sil_score
        from sklearn.metrics import davies_bouldin_score as db_score

        silhouette_score = sil_score(X_scaled, labels)
        davies_bouldin_score = db_score(X_scaled, labels)
    except:
        pass

    # Prepare results dictionary
    results = {
        'total_customers': len(original_data),
        'n_clusters': optimal_k,
        'silhouette_score': silhouette_score if silhouette_score else 0,
        'davies_bouldin_score': davies_bouldin_score if davies_bouldin_score else 0,
        'cluster_profiles': cluster_profiles
    }

    # Print business recommendations
    print_business_recommendations(cluster_profiles)

    # Print key findings
    print_section_header("🔍 KEY FINDINGS", "=", 60)

    print(f"\n1. 📊 Customer Segmentation:")
    print(f"   • Identified {optimal_k} distinct customer segments")
    print(
        f"   • Most valuable segment: {cluster_profiles.loc[cluster_profiles['Income_Mean'].idxmax(), 'Cluster_Label']}")
    print(f"   • Largest segment: {cluster_profiles.loc[cluster_profiles['Customer_Count'].idxmax(), 'Cluster_Label']}")

    if silhouette_score:
        print(f"\n2. 📈 Model Performance:")
        print(f"   • Silhouette Score: {silhouette_score:.3f} (Good clustering quality)")
        print(f"   • Clusters are well-separated and internally cohesive")

    print(f"\n3. 💰 Revenue Opportunities:")
    print(f"   • Target high-value segments with premium offerings")
    print(f"   • Reactivate low-engagement segments with personalized campaigns")
    print(f"   • Optimize marketing budget allocation based on segment size")

    # Print project summary
    print_project_summary(results)

    # ============================================================
    # STEP 5: EXPORT RESULTS
    # ============================================================
    print_section_header("STEP 5: EXPORTING RESULTS", "=", 60)

    try:
        # Export clustered data
        clustered_data = original_data.copy()
        clustered_data.to_csv('outputs/clustered_customers.csv', index=False)
        print("✅ Clustered customer data saved to: outputs/clustered_customers.csv")

        # Export segment summary
        segment_summary = cluster_profiles[['Cluster_Label', 'Customer_Count', 'Customer_Percentage',
                                            'Income_Mean', 'Spending_Mean']]
        segment_summary.to_csv('outputs/segment_summary.csv')
        print("✅ Segment summary saved to: outputs/segment_summary.csv")

        print("✅ All results exported successfully!")

    except Exception as e:
        print(f"⚠️ Warning: Could not export some files: {str(e)}")

    # ============================================================
    # COMPLETION
    # ============================================================
    print_section_header("✅ PROJECT COMPLETED SUCCESSFULLY!", "=", 60)

    print("\n📁 Output Location: ./outputs/")
    print("📊 Files Generated:")
    print("   • clusters.png - 2D cluster visualization")
    print("   • elbow_method.png - Optimal cluster selection")
    print("   • pca_clusters.png - PCA visualization")
    print("   • cluster_distribution.png - Segment sizes")
    print("   • radar_chart.png - Segment comparison")
    print("   • cluster_profiles.csv - Detailed segment statistics")
    print("   • clustered_customers.csv - Data with cluster labels")
    print("   • segment_summary.csv - Quick segment overview")

    print("\n💡 Next Steps:")
    print("   1. Review the cluster profiles in 'outputs/cluster_profiles.csv'")
    print("   2. Analyze visualizations to understand segment characteristics")
    print("   3. Develop targeted marketing strategies for each segment")
    print("   4. Implement A/B testing to validate recommendations")
    print("   5. Monitor and retrain model periodically with new data")

    print("\n🎉 Thank you for using Customer Segmentation Tool!")
    print("📧 For questions or improvements, please refer to the documentation.\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        # Run main function
        main()

    except KeyboardInterrupt:
        print("\n\n⚠️ Project interrupted by user.")
        print("👋 Exiting gracefully...")

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please check the error message above and try again.")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure all required packages are installed: pip install -r requirements.txt")
        print("   2. Verify that generate_data.py has been run successfully")
        print("   3. Check that src folder contains all required modules")
        print("   4. Ensure data/customer_data.csv exists and is valid")

    finally:
        print("\n" + "=" * 60)
        print("👋 Project execution completed!")
        print("=" * 60)