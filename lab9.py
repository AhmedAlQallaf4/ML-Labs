import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Load Data
df = pd.read_csv('amazon_sales_dataset.csv')

# 2. Preprocessing: Encoding categorical variables
le = LabelEncoder()
df['category_enc'] = le.fit_transform(df['product_category'])
df['region_enc'] = le.fit_transform(df['customer_region'])

# 3. Define Features and Target
features = ['price', 'discount_percent', 'quantity_sold', 'rating', 'review_count', 
            'total_revenue', 'category_enc', 'region_enc']
X = df[features]
y = df['payment_method']

# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dt_pred = dtree.predict(X_test)

print("--- Decision Tree Classification Report ---")
print(classification_report(y_test, dt_pred))

# 6. Train Random Forest
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rf_pred = rfc.predict(X_test)

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_pred))