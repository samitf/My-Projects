import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Import joblib for saving the model

# Load your dataset
data = pd.read_csv("indian_sales.csv")

# Check the schema of the dataset
print(data.info())

# Rename the 'Sales (₹)' column to 'Sales' for easier reference
data.rename(columns={"Sales (₹)": "Sales"}, inplace=True)

# Handling missing values (filling with 0)
data.fillna(0, inplace=True)

# Apply log transformation to Sales (adding 1 to avoid log(0))
data["log_Sales"] = np.log(data["Sales"] + 1)

# Define categorical columns
categorical_cols = ["State", "Product_Category", "Month"]

# One-hot encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(data[categorical_cols])

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# Combine the encoded features with the original dataset
data = pd.concat([data, encoded_df], axis=1)

# Define feature columns and target variable
feature_columns = encoded_df.columns.tolist()
X = data[feature_columns]
y = data["log_Sales"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree Regression Model
dt = DecisionTreeRegressor()

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 8]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
predictions = best_model.predict(X_test_scaled)

# Show predictions
predicted_df = pd.DataFrame({
    "Features": X_test_scaled.tolist(),
    "Sales": y_test,
    "log_Sales": predictions
})
print(predicted_df)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Print metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R²:", r2)

# Save the best model, encoder, and scaler as .pkl files
joblib.dump(best_model, 'model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model, encoder, and scaler saved as model.pkl, encoder.pkl, and scaler.pkl")
