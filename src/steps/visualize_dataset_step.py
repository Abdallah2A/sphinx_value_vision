import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap
from zenml import step
import matplotlib.ticker as ticker


@step
def visualize_housing_data(file_path, output_folder):
    """
    Visualizes a housing price dataset with histograms, box plots, correlation matrix, and scatter plots.
    Assumes the dataset is already decoded.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - output_folder (str): Directory where visualization images will be saved.
    """
    if os.path.exists(output_folder):
        return
    # Set Seaborn style for consistent and appealing visuals
    sns.set_theme(style='whitegrid')

    # Load the dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded. First few rows:")
    print(df.head())

    # Identify categorical and numerical features
    # Categorical: columns with dtype 'object'
    categorical_features = [col for col in df.columns if df[col].dtype == 'object']
    # Numerical: features of type 'int64' or 'float64'
    numerical_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # Create output folder if it doesn’t exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Histograms: Distribution of numerical features
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.savefig(os.path.join(output_folder, f'histogram_{feature.replace("/", ",")}.png'))
        plt.close()
        print(f"Histogram for {feature} saved.")

    # 2. Correlation Matrix Heatmap
    if numerical_features:
        plt.figure(figsize=(15, 10))
        corr = df[numerical_features].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.savefig(os.path.join(output_folder, 'correlation_matrix.png'))
        plt.close()
        print("Correlation matrix saved.")
    else:
        print("No numerical features found for correlation matrix.")

    # 3. Scatter Plots: Numerical features vs Price
    if 'price' in numerical_features:
        for feature in numerical_features:
            if feature != 'price':
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=df[feature], y=df['price'])
                plt.title(f'{feature.capitalize()} vs Price')
                plt.xlabel(feature.capitalize())
                plt.ylabel('price')
                # Format y-axis for large numbers
                ax = plt.gca()
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}M' if x >= 1e6 else f'{x / 1e3:.1f}K'))
                plt.savefig(os.path.join(output_folder, f'scatter_{feature.replace("/", ",")}_vs_price.png'))
                plt.close()
                print(f"Scatter plot for {feature} vs price saved.")
    else:
        print("price not found in numerical features. Skipping scatter plots vs price.")

    # 4. Box Plots: Numerical features for outlier detection
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.savefig(os.path.join(output_folder, f'boxplot_{feature.replace("/", ",")}.png'))
        plt.close()
        print(f"Box plot for {feature} saved.")

    # 5. Box Plots: Price by Categorical Features
    if 'price' in df.columns and categorical_features:
        for feature in categorical_features:
            if df[feature].nunique() <= 20:  # Limit to avoid cluttered plots
                plt.figure(figsize=(10, 6))
                ax = sns.boxplot(x=df[feature], y='price', data=df)
                plt.title(f'price Distribution by {feature.capitalize()}')
                plt.xlabel(feature.capitalize())
                plt.ylabel('price')
                # Wrap x-axis labels to a maximum width of 15 characters
                labels = [textwrap.fill(label.get_text(), width=15) for label in ax.get_xticklabels()]
                ax.set_xticklabels(labels, rotation=45, ha='right')  # Rotate and align right
                # Format y-axis for large numbers
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}M' if x >= 1e6 else f'{x / 1e3:.1f}K'))
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f'boxplot_price_by_{feature.replace("/", ",")}.png'),
                            bbox_inches='tight')
                plt.close()
                print(f"Box plot of price by {feature} saved.")
            else:
                print(f"Skipped box plot for {feature} due to too many unique values ({df[feature].nunique()}).")
    else:
        print("price or categorical features not found. Skipping categorical box plots.")

    print(f"\nAll visualizations have been saved in '{output_folder}'.")


# if __name__ == "__main__":
#     visualize_housing_data('../../data/dataset.csv', '../../visual_matrices')
