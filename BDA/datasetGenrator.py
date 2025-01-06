import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define regions and their corresponding states
regions = {
    'North': ['Punjab', 'Haryana', 'Himachal Pradesh', 'Uttarakhand', 'Delhi'],
    'South': ['Tamil Nadu', 'Kerala', 'Karnataka', 'Andhra Pradesh', 'Telangana'],
    'East': ['West Bengal', 'Odisha', 'Bihar', 'Jharkhand', 'Assam'],
    'West': ['Maharashtra', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'Goa']
}

# Create a mapping of state to region for quick lookup
state_to_region = {state: region for region, states in regions.items() for state in states}

# Define product categories and their corresponding weightages by state
product_categories = ['Electronics', 'Apparel', 'Groceries', 'Furniture', 'Footwear']
weightages = {
    'Punjab': {'Electronics': 2.0, 'Apparel': 1.8, 'Groceries': 1.0, 'Furniture': 1.0, 'Footwear': 1.5},
    'Haryana': {'Electronics': 2.0, 'Apparel': 1.5, 'Groceries': 0.9, 'Furniture': 1.1, 'Footwear': 1.3},
    'Himachal Pradesh': {'Electronics': 1.2, 'Apparel': 1.4, 'Groceries': 1.2, 'Furniture': 1.3, 'Footwear': 1.0},
    'Uttarakhand': {'Electronics': 1.5, 'Apparel': 1.1, 'Groceries': 1.0, 'Furniture': 1.0, 'Footwear': 1.0},
    'Delhi': {'Electronics': 3.5, 'Apparel': 2.5, 'Groceries': 1.5, 'Furniture': 1.3, 'Footwear': 1.8},
    'Tamil Nadu': {'Electronics': 2.5, 'Apparel': 2.5, 'Groceries': 1.5, 'Furniture': 1.0, 'Footwear': 1.3},
    'Kerala': {'Electronics': 2.0, 'Apparel': 2.0, 'Groceries': 1.6, 'Furniture': 1.1, 'Footwear': 1.5},
    'Karnataka': {'Electronics': 2.5, 'Apparel': 2.3, 'Groceries': 1.8, 'Furniture': 1.5, 'Footwear': 1.7},
    'Andhra Pradesh': {'Electronics': 1.8, 'Apparel': 2.0, 'Groceries': 1.6, 'Furniture': 1.0, 'Footwear': 1.2},
    'Telangana': {'Electronics': 2.2, 'Apparel': 1.9, 'Groceries': 1.5, 'Furniture': 1.1, 'Footwear': 1.4},
    'West Bengal': {'Electronics': 1.5, 'Apparel': 2.1, 'Groceries': 1.9, 'Furniture': 1.0, 'Footwear': 1.3},
    'Odisha': {'Electronics': 1.5, 'Apparel': 1.8, 'Groceries': 1.5, 'Furniture': 0.9, 'Footwear': 1.2},
    'Bihar': {'Electronics': 1.3, 'Apparel': 1.5, 'Groceries': 1.6, 'Furniture': 0.8, 'Footwear': 1.1},
    'Jharkhand': {'Electronics': 1.0, 'Apparel': 1.2, 'Groceries': 1.5, 'Furniture': 0.8, 'Footwear': 1.0},
    'Assam': {'Electronics': 1.2, 'Apparel': 1.3, 'Groceries': 1.7, 'Furniture': 0.7, 'Footwear': 1.0},
    'Maharashtra': {'Electronics': 3.0, 'Apparel': 2.5, 'Groceries': 1.8, 'Furniture': 1.7, 'Footwear': 2.0},
    'Gujarat': {'Electronics': 2.5, 'Apparel': 2.0, 'Groceries': 1.5, 'Furniture': 1.5, 'Footwear': 1.6},
    'Rajasthan': {'Electronics': 2.0, 'Apparel': 1.5, 'Groceries': 1.2, 'Furniture': 1.2, 'Footwear': 1.5},
    'Madhya Pradesh': {'Electronics': 1.8, 'Apparel': 1.3, 'Groceries': 1.5, 'Furniture': 1.1, 'Footwear': 1.4},
    'Goa': {'Electronics': 1.5, 'Apparel': 2.0, 'Groceries': 1.2, 'Furniture': 1.0, 'Footwear': 1.4},
}

# Define the months and their weightages for sales
month_weightages = {
    'January': 1.1,
    'February': 0.9,
    'March': 1.0,
    'April': 1.2,
    'May': 1.3,
    'June': 1.0,
    'July': 1.0,
    'August': 1.1,
    'September': 1.2,
    'October': 1.5,  # Festive season
    'November': 1.7,  # Festive season
    'December': 1.6,  # Festive season
}

# Define the number of entries
num_entries = 500000  # Increased dataset size for better training

def generate_sales_data(num_entries):
    sales_data = []
    print("Available states:", list(weightages.keys()))  # Debug line
    for _ in range(num_entries):
        state = random.choice(list(weightages.keys()))
        region = state_to_region[state]
        product_category = random.choice(product_categories)
        month = random.choice(list(month_weightages.keys()))

        # Base sales calculation with seasonal adjustment
        base_sales = max(0, int(np.random.normal(loc=30000 + product_categories.index(product_category) * 2000, scale=5000)))
        sales = int(base_sales * weightages[state][product_category] * month_weightages[month])

        """# Adding a percentage noise to the sales figure
        noise_percentage = np.random.uniform(-0.05, 0.10)  # Noise for variation
        sales = int(sales * (1 + noise_percentage))
        sales = max(0, min(sales, 200000))  # Cap sales to prevent unreasonably high values"""

        sales_data.append({
            'State': state,
            'Product_Category': product_category,
            'Month': month,
            'Region': region,
            'Sales (₹)': sales
        })
    return sales_data

# Create a DataFrame from the generated data
sales_data = generate_sales_data(num_entries)
df = pd.DataFrame(sales_data)

# Check for outliers in 'Sales (₹)' and remove them
q1, q3 = df['Sales (₹)'].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['Sales (₹)'] >= lower_bound) & (df['Sales (₹)'] <= upper_bound)]

# Save the DataFrame to a CSV file
df.to_csv('indian_sales.csv', index=False)

# Display the first few rows of the DataFrame
print(df.head())
