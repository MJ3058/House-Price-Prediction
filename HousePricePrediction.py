import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

# Loading data
file_path = 'C:/Users/mirja/Onedrive/Documents/train.csv'
data = pd.read_csv(file_path)

# Drop ID column
if 'Id' in data.columns:
    data.drop(columns=['Id'], inplace=True)

# Show columns with missing values only
missing_values = data.isnull().sum().sort_values(ascending=False)
print("Missing values before processing:")
print(missing_values[missing_values > 0])

# Fill missing values for remaining columns
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype in ['int64', 'float64']: # Numerical columns
            data[col] = data[col].fillna(data[col].median()) # Fill with median
        else: # Categorical columns
            data[col] = data[col].fillna(data[col].mode()[0]) # Fill with most frequent value

# Select numerical and categorical columns
num_cols= data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

# Applying Ordinal Encoding
encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
data[cat_cols] = encoder.fit_transform(data[cat_cols])

# Verify final feature count
print(f"\nFinal feature count after preprocessing: {data.shape[1]-1}") # Exclude SalePrice

# Splitting features and target variable
X = data.drop(columns=['SalePrice']) # Features
y = data['SalePrice'] # Target variable

# Debug Feature Count before Train-Test Split
print(f"Number of features BEFORE train-test-split: {X.shape[1]}")
print(f"Feature Names: {X.columns.tolist()}")

# Perform Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Select numerical columns again after split
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols]) #Fit on train
X_test[num_cols] = scaler.transform(X_test[num_cols]) # Transform test

# Final Feature Count after encoding and scaling
print(f"\nFinal Training Data: {X_train.shape}, Testing Data: {X_test.shape}")
print(f"Number of features AFTER train-test-split: {X_train.shape[1]}")
print(f"Final Feature Names: {X_train.columns.tolist()}")

# Initialising model
model = LinearRegression()

# Training model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Initialise Ridge Regression (+ penalty)
ridge_model = Ridge(alpha=200)
ridge_model.fit(X_train, y_train)

# Predict with Ridge
y_ridge_pred = ridge_model.predict(X_test)

# Evaluate Ridge model
mse_ridge = mean_squared_error(y_test, y_ridge_pred)
r2_ridge = r2_score(y_test, y_ridge_pred)

print(f"\nRidge Regression - MSE: {mse_ridge:.2f}")
print(f"Ridge Regression - R2S: {r2_ridge:.4f}")

# Apply log transformation to SalePrice
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Plot distribution of house prices
plt.figure(figsize = (8, 5))
sns.histplot(y_train_log, bins = 50, kde = True)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.show()

# Train Ridge model on log-transformed target
ridge_model_log = Ridge(alpha = 200)
ridge_model_log.fit(X_train, y_train_log)

y_pred_log = ridge_model_log.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test_log)

mse_log = mean_squared_error(y_test_original, y_pred)
r2_log = r2_score(y_test_original, y_pred)
print(f"\nAfter Log Transformation - MSE: {mse_log:.2f} ")
print(f"After Log Transformation - R2S: {r2_log:.4f}")

# Compute correlation matrix and plot the heatmap
corr_matrix = data.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(corr_matrix, cmap = "coolwarm", annot = False, linewidths = 0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Get correlation values for SalePrice
saleprice_corr = corr_matrix['SalePrice'].sort_values(ascending=False)
print("\nTop 10 Features most correlated with SalePrice:")
print(saleprice_corr.head(11))

# Scatter plot of OverallQual vs SalePrice
plt.figure(figsize = (8, 5))
sns.scatterplot(x = X_train['OverallQual'], y = y_train, alpha= 0.5)
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.title("Overall Quality vs. Sale Price")
plt.show()

# Scatter plot of GrLivArea vs SalePrice
plt.figure(figsize = (8, 5))
sns.scatterplot(x = X_train['GrLivArea'], y = y_train, alpha= 0.5)
plt.xlabel("Above Ground Living Area (sq ft")
plt.ylabel("Sale Price")
plt.title("GrLivArea vs. Sale Price")
plt.show()

# Box plot of YearBuilt vs SalePrice
plt.figure(figsize=(12, 5))
sns.boxplot(x=X_train['YearBuilt'], y=y_train)
plt.xticks(rotation=90)
plt.xlabel("Year Built")
plt.ylabel("Sale Price")
plt.title("Year Built vs Sale Price")
plt.show()

# Predict on test data
y_pred = ridge_model.predict(X_test)

# Create a scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Prices")
plt.show()

# Train the final Ridge Regression Model
final_model = Ridge(alpha = 200)
final_model.fit(X_train, y_train_log)

# Save trained model
joblib.dump(final_model, "HousePricesModel.pkl")

print("\nFinal Model saved as 'HousePricesModel.pkl'")

# Load trained model
loaded_model = joblib.load('C:/Users/mirja/PycharmProjects/PythonProject1/HousePricesModel.pkl')

# Predict on test data
y_pred_log = loaded_model.predict(X_test)

# Convert predictions back to normal SalePrice
y_pred_final = np.expm1(y_pred_log)


print("\nPredicted House Prices:", y_pred_final)
