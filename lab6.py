import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the dataset
df = pd.read_csv('amazon_sales_dataset.csv')

# 2. Basic Exploration
print("--- Data Info ---")
print(df.info())

# 3. Data Preprocessing
# Select features (X) and target (y)
# We exclude 'total_revenue' from X to use it as the prediction target
target = 'total_revenue'
features = ['price', 'discount_percent', 'quantity_sold', 'rating', 'review_count', 
            'product_category', 'customer_region', 'payment_method']

X = df[features]
y = df[target]

# Define categorical and numerical columns for transformation
categorical_cols = ['product_category', 'customer_region', 'payment_method']
numeric_cols = ['price', 'discount_percent', 'quantity_sold', 'rating', 'review_count']

# Create a preprocessing pipeline (Encoding categorical data)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# 4. Prepare for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a full model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Train the model
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)

print("\n--- Model Performance ---")
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))