"""
ARTI308 - Machine Learning
Lab Assignment: Data Analysis and Visualization
Dataset: Amazon Sales
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Ensure the file 'amazon_sales_dataset.csv' is in the same folder as this script
df = pd.read_csv('amazon_sales_dataset.csv')

# Preprocessing: Convert order_date to datetime and extract Month-Year for trends
df['order_date'] = pd.to_datetime(df['order_date'])
df['Month_Year'] = df['order_date'].dt.to_period('M').astype(str)

# 2. Aggregation: Calculate total Revenue and Quantity Sold by Product Category
category_totals = df.groupby('product_category')[['total_revenue', 'quantity_sold']].sum()

print("-" * 40)
print("TOTAL REVENUE BY CATEGORY")
print("-" * 40)
print(category_totals)
print("-" * 40)

# --- 3. VISUALIZATION (No Pop-ups) ---
sns.set_theme(style="whitegrid")

# Plot 1: Monthly Revenue Trend for top categories
# We'll filter for a cleaner view of the trend over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df.sort_values('order_date'), x='Month_Year', y='total_revenue', 
             hue='product_category', estimator='sum', errorbar=None, marker='o')

plt.title('Monthly Revenue Trend by Product Category', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Total Revenue ($)')
plt.xlabel('Month')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('amazon_revenue_trend.png')
plt.close()

# Plot 2: Relationship between Price and Total Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='price', y='total_revenue', hue='product_category', alpha=0.6)

plt.title('Relationship: Price vs Total Revenue', fontsize=14)
plt.ylabel('Total Revenue ($)')
plt.xlabel('Unit Price ($)')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('amazon_price_vs_revenue.png')
plt.close()

print("\nSuccess! Results have been saved to 'amazon_revenue_trend.png' and 'amazon_price_vs_revenue.png'.")