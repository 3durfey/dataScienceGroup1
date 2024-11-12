import matplotlib.pyplot as plt
import seaborn as sns

def sqft_plot(df):
    # Check if the 'price' column exists
    if 'price' in df:
        # Convert the 'price' column to numeric (if necessary) and drop NaN values
        #df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Set the style for seaborn
        sns.set_theme(style="whitegrid")


        # 1. Histogram of Square Feet
        plt.figure(figsize=(10, 6))
        sns.histplot(df['square_feet'], bins=30, kde=True, color='blue')
        plt.title('Distribution of Square Feet in Apartments')
        plt.xlabel('Square Feet')
        plt.ylabel('Frequency')
        plt.axvline(df['square_feet'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean Size')
        plt.axvline(df['square_feet'].median(), color='green', linestyle='dashed', linewidth=1, label='Median Size')
        plt.legend()
        plt.show()

        # 2. Box Plot of Rental Prices
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['price'], color='orange')
        plt.title('Box Plot of Rental Prices')
        plt.xlabel('Rental Price ($)')
        plt.show()
        
        