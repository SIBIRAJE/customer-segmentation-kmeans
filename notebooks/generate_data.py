"""
Customer Data Generator for Segmentation Project
Generates realistic customer data with income and spending patterns
"""

import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)


def generate_customer_data(n_customers=500, save_path='data/customer_data.csv'):
    """
    Generate synthetic customer data for segmentation analysis

    Parameters:
    -----------
    n_customers : int
        Number of customers to generate (default: 500)
    save_path : str
        Path where to save the CSV file (default: 'data/customer_data.csv')

    Returns:
    --------
    pandas.DataFrame
        Generated customer data
    """

    print("=" * 60)
    print("📊 CUSTOMER DATA GENERATOR")
    print("=" * 60)
    print(f"Generating {n_customers} customer records...")

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Generate basic customer information
    customer_ids = np.arange(1, n_customers + 1)
    age = np.random.randint(18, 70, n_customers)

    # Generate income with different segments (creating natural clusters)
    # Segment 1: Low income (30% of customers)
    n_low = int(n_customers * 0.30)
    # Segment 2: Medium income (45% of customers)
    n_medium = int(n_customers * 0.45)
    # Segment 3: High income (25% of customers)
    n_high = n_customers - n_low - n_medium

    income_low = np.random.normal(25000, 5000, n_low).clip(15000, 40000)
    income_medium = np.random.normal(55000, 8000, n_medium).clip(40000, 75000)
    income_high = np.random.normal(90000, 15000, n_high).clip(75000, 150000)

    annual_income = np.concatenate([income_low, income_medium, income_high])
    np.random.shuffle(annual_income)  # Shuffle to mix segments

    # Generate spending scores based on income with some variation
    # Creating different spending behaviors within same income levels

    spending_scores = []

    for income in annual_income:
        if income < 35000:  # Low income group
            # Some low income customers spend more (impulse buyers)
            if np.random.random() < 0.3:
                spending = np.random.normal(70, 10)  # High spenders
            else:
                spending = np.random.normal(35, 12)  # Conservative spenders

        elif income < 70000:  # Medium income group
            # Medium income has mixed behavior
            if np.random.random() < 0.4:
                spending = np.random.normal(75, 10)  # Big spenders
            elif np.random.random() < 0.6:
                spending = np.random.normal(55, 12)  # Average spenders
            else:
                spending = np.random.normal(35, 10)  # Frugal

        else:  # High income group
            # High income customers
            if np.random.random() < 0.7:
                spending = np.random.normal(85, 10)  # Premium spenders
            else:
                spending = np.random.normal(55, 12)  # Conservative high income

        spending_scores.append(spending)

    spending_score = np.array(spending_scores).clip(0, 100).round(2)
    annual_income_k = (annual_income / 1000).round(2)

    # Create DataFrame
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': age,
        'Annual_Income_k$': annual_income_k,
        'Spending_Score': spending_score
    })

    # Add some realistic patterns
    # Younger customers tend to spend more relative to income
    for idx in range(len(data)):
        if data.loc[idx, 'Age'] < 30:
            data.loc[idx, 'Spending_Score'] = min(100, data.loc[idx, 'Spending_Score'] + 5)
        elif data.loc[idx, 'Age'] > 55:
            data.loc[idx, 'Spending_Score'] = max(0, data.loc[idx, 'Spending_Score'] - 5)

    # Add some noise to make it realistic
    data['Spending_Score'] = data['Spending_Score'] + np.random.normal(0, 3, n_customers)
    data['Spending_Score'] = data['Spending_Score'].clip(0, 100).round(2)

    # Save to CSV
    data.to_csv(save_path, index=False)

    # Print statistics
    print("\n✅ Data Generated Successfully!")
    print(f"📍 Saved to: {save_path}")
    print(f"📊 Dataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\n" + "=" * 60)
    print("📈 DATA STATISTICS")
    print("=" * 60)
    print("\n💰 Income Statistics:")
    print(f"   Mean: ${data['Annual_Income_k$'].mean():.2f}k")
    print(f"   Median: ${data['Annual_Income_k$'].median():.2f}k")
    print(f"   Min: ${data['Annual_Income_k$'].min():.2f}k")
    print(f"   Max: ${data['Annual_Income_k$'].max():.2f}k")

    print("\n🛍️ Spending Score Statistics:")
    print(f"   Mean: {data['Spending_Score'].mean():.2f}")
    print(f"   Median: {data['Spending_Score'].median():.2f}")
    print(f"   Min: {data['Spending_Score'].min():.2f}")
    print(f"   Max: {data['Spending_Score'].max():.2f}")

    print("\n👥 Age Distribution:")
    print(f"   Mean Age: {data['Age'].mean():.1f} years")
    print(f"   Age Range: {data['Age'].min()} - {data['Age'].max()} years")

    print("\n📋 First 5 Records:")
    print(data.head())

    print("\n📋 Last 5 Records:")
    print(data.tail())

    return data


def generate_enhanced_data(n_customers=1000, save_path='data/customer_data_enhanced.csv'):
    """
    Generate enhanced customer data with additional features

    Parameters:
    -----------
    n_customers : int
        Number of customers to generate
    save_path : str
        Path to save the CSV file

    Returns:
    --------
    pandas.DataFrame
        Enhanced customer data
    """

    print("\n" + "=" * 60)
    print("🎯 GENERATING ENHANCED CUSTOMER DATA")
    print("=" * 60)

    # Create data directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Generate basic data first
    base_data = generate_customer_data(n_customers, save_path='temp_data.csv')

    # Add additional features for enhanced analysis
    base_data['Gender'] = np.random.choice(['Male', 'Female'], n_customers, p=[0.48, 0.52])
    base_data['Income_Category'] = pd.cut(base_data['Annual_Income_k$'],
                                          bins=[0, 30, 60, 90, 200],
                                          labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

    base_data['Spending_Category'] = pd.cut(base_data['Spending_Score'],
                                            bins=[0, 33, 66, 100],
                                            labels=['Low Spender', 'Medium Spender', 'High Spender'])

    # Add loyalty score (how loyal customer is)
    base_data['Loyalty_Score'] = np.random.uniform(0, 100, n_customers).round(2)

    # Add purchase frequency (times per month)
    base_data['Purchase_Frequency'] = np.random.poisson(5, n_customers).clip(1, 20)

    # Add customer tenure (months)
    base_data['Tenure_Months'] = np.random.randint(1, 60, n_customers)

    # Save enhanced data
    base_data.to_csv(save_path, index=False)

    print(f"\n✅ Enhanced data saved to: {save_path}")
    print(
        f"📊 New features added: Gender, Income_Category, Spending_Category, Loyalty_Score, Purchase_Frequency, Tenure_Months")

    # Clean up temp file
    if os.path.exists('temp_data.csv'):
        os.remove('temp_data.csv')

    return base_data


def load_and_verify_data(filepath='data/customer_data.csv'):
    """
    Load and verify the generated data

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    print("\n" + "=" * 60)
    print("🔍 VERIFYING DATA")
    print("=" * 60)

    try:
        data = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully from {filepath}")
        print(f"📊 Shape: {data.shape}")
        print(f"📋 Columns: {list(data.columns)}")
        print(f"🔢 Missing values: {data.isnull().sum().sum()}")

        # Check for data quality
        if data['Annual_Income_k$'].isnull().any():
            print("⚠️ Warning: Missing values found in income column")

        if (data['Spending_Score'] < 0).any() or (data['Spending_Score'] > 100).any():
            print("⚠️ Warning: Spending score outside 0-100 range")

        return data

    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        print("Please generate data first using generate_customer_data()")
        return None


def create_sample_data_for_testing():
    """
    Create a small sample dataset for testing
    """
    print("\n" + "=" * 60)
    print("🧪 CREATING TEST DATA")
    print("=" * 60)

    sample_data = pd.DataFrame({
        'CustomerID': range(1, 51),
        'Age': np.random.randint(18, 70, 50),
        'Annual_Income_k$': np.random.uniform(20, 150, 50).round(2),
        'Spending_Score': np.random.uniform(20, 95, 50).round(2)
    })

    # Create test directory
    os.makedirs('data', exist_ok=True)
    sample_data.to_csv('data/test_data.csv', index=False)
    print("✅ Test data created at: data/test_data.csv")
    print(f"📊 Test dataset shape: {sample_data.shape}")

    return sample_data


# Main execution
if __name__ == "__main__":
    print("🚀 CUSTOMER DATA GENERATION TOOL")
    print("=" * 60)

    # Generate main dataset
    data = generate_customer_data(n_customers=500, save_path='data/customer_data.csv')

    # Generate enhanced dataset (optional)
    print("\n" + "=" * 60)
    choice = input("Do you want to generate enhanced data with more features? (y/n): ")
    if choice.lower() == 'y':
        enhanced_data = generate_enhanced_data(n_customers=500, save_path='data/customer_data_enhanced.csv')

    # Create test data
    print("\n" + "=" * 60)
    choice = input("Do you want to create a small test dataset? (y/n): ")
    if choice.lower() == 'y':
        test_data = create_sample_data_for_testing()

    # Verify data
    print("\n" + "=" * 60)
    verify_data = load_and_verify_data('data/customer_data.csv')

    print("\n" + "=" * 60)
    print("✅ DATA GENERATION COMPLETE!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print("   - data/customer_data.csv (Main dataset)")
    if choice.lower() == 'y':
        print("   - data/customer_data_enhanced.csv (Enhanced dataset)")
        print("   - data/test_data.csv (Test dataset)")
    print("\n🎯 Next step: Run main.py to perform customer segmentation!")