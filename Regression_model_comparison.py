""""" BAN 5003 - Regression model comparison """
import os
import pandas as pd
import psycopg2
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

DSN = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "BAN5003"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

QUERY = """
SELECT
    oi.order_id,
    oi.order_item_id,
    oi.freight_value,
    p.product_weight_g / 1000.0 AS product_weight_kg,
    p.product_length_cm,
    p.product_width_cm,
    p.product_height_cm,
    p.product_category_name,
    s.seller_state,
    c.customer_state,
    CONCAT(s.seller_state, '-', c.customer_state) AS shipping_route
FROM olist.order_items oi
JOIN olist.products p ON oi.product_id = p.product_id
JOIN olist.orders o ON oi.order_id = o.order_id
JOIN olist.customers c ON o.customer_id = c.customer_id
JOIN olist.sellers s ON oi.seller_id = s.seller_id
"""
products_df = pd.read_sql(QUERY, psycopg2.connect(**DSN))

# Display basic information about the dataset and check for missing values
print("\nTotal rows:", len(products_df))
print("Total columns:", len(products_df.columns))
print("\nColumn names:")
print(products_df.columns.tolist())
print("\nData types:")
print(products_df.dtypes)
print("\nMissing values:")
print(products_df.isnull().sum())

# Check for duplicates in the order_items
duplicates = products_df.duplicated(subset=["order_id", "order_item_id"]).sum()
print("\nDuplicate order_items:", duplicates)

#Checking for invalid product dimensions and freight values
invalid_dimensions = {
    "Weight <= 0": (products_df["product_weight_kg"] <= 0).sum(),
    "Length <= 0": (products_df["product_length_cm"] <= 0).sum(),
    "Width <= 0": (products_df["product_width_cm"] <= 0).sum(),
    "Height <= 0": (products_df["product_height_cm"] <= 0).sum(),
    "Freight < 0": (products_df["freight_value"] < 0).sum(),
    "Freight == 0": (products_df["freight_value"] == 0).sum(),
}
print("\nInvalid product dimensions and freight values:")
print(pd.Series(invalid_dimensions))

invalid_measurements = (
      (products_df["product_weight_kg"] <= 0)
    | (products_df["product_length_cm"] <= 0)
    | (products_df["product_width_cm"] <= 0)
    | (products_df["product_height_cm"] <= 0)
    )
print("\nRows with invalid measurements:", invalid_measurements.sum())

# Cleaning the dataset by removing rows with invalid measurements

clean_df = products_df.loc[~invalid_measurements].copy()
print("\nRows before cleaning:", len(products_df))
print("Rows after cleaning:", len(clean_df))
print("Rows removed:", len(products_df) - len(clean_df))

#Independent and dependent variables
numeric_features = [
    "product_weight_kg", 
    "product_length_cm", 
    "product_width_cm", 
    "product_height_cm"
]
categorical_features = [
    "product_category_name",
    "shipping_route"
]
X = clean_df[numeric_features + categorical_features]
Y = clean_df["freight_value"]
print("\n Independent variables (X):")
print(X.columns.tolist())
print("\n Dependent variable (Y):")
print(Y.name)
print("\n Number of observations:", len(clean_df))

#Splitting the dataset into training, validation, and test sets using GroupShuffleSplit to ensure that all order_items from the same order_id are in the same set
split1=GroupShuffleSplit(
    n_splits=1, 
    train_size=0.60, 
    random_state=42
)
train_idx, temp_idx = next(split1.split(X, Y, groups=clean_df["order_id"]))
X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
X_temp, Y_temp = X.iloc[temp_idx], Y.iloc[temp_idx]

split2=GroupShuffleSplit(
    n_splits=1, 
    test_size=0.50, 
    random_state=42
)
val_idx, test_idx = next(split2.split(X_temp, Y_temp, groups=clean_df.iloc[temp_idx]["order_id"]))
X_val, Y_val = X_temp.iloc[val_idx], Y_temp.iloc[val_idx]
X_test, Y_test = X_temp.iloc[test_idx], Y_temp.iloc[test_idx]
print("\nTraining set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))    
print("\nTotal rows in all sets:", len(X_train) + len(X_val) + len(X_test))

#Preprocessing pipeline for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
# Model A : Regression model
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor), 
        ("model", LinearRegression())
    ]
)
linear_model.fit(X_train, Y_train)
print("\nLinear regression model fitted successfully.")

#Validating the model on the validation set
Y_val_pred = linear_model.predict(X_val)
mae = mean_absolute_error(Y_val, Y_val_pred)
rmse = root_mean_squared_error(Y_val, Y_val_pred)
r2 = r2_score(Y_val, Y_val_pred)
print("\nLinear Regression validation set performance:")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  R-squared: {r2:.3f}")

# Model B : Random Forest model
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor), 
        ("model", RandomForestRegressor(random_state=42))
    ]
)
rf_model.fit(X_train, Y_train)
print("\nRandom Forest model fitted successfully.")

#Validating the Random Forest model on the validation set
Y_val_pred_rf = rf_model.predict(X_val)
mae_rf = mean_absolute_error(Y_val, Y_val_pred_rf)
rmse_rf = root_mean_squared_error(Y_val, Y_val_pred_rf)
r2_rf = r2_score(Y_val, Y_val_pred_rf)
print("\nRandom Forest model validation set performance:")
print(f"  Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
print(f"  R-squared: {r2_rf:.3f}")  

# 5-fold cross-validation for both models
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)
groups_train = clean_df.iloc[train_idx]["order_id"]
linear_cv_scores = -cross_val_score(linear_model, X_train, Y_train, cv=cv, groups=groups_train, scoring='neg_mean_absolute_error')
rf_cv_scores = -cross_val_score(rf_model, X_train, Y_train, cv=cv, groups=groups_train, scoring='neg_mean_absolute_error')
print(f"\n5-fold Cross-Validation Results:")
print(f"Linear Regression MAE: {linear_cv_scores.mean():.2f} (+/- {linear_cv_scores.std():.2f})")
print(f"Random Forest MAE: {rf_cv_scores.mean():.2f} (+/- {rf_cv_scores.std():.2f})")

#Test-set evaluation - Linear Regression
Y_test_pred = linear_model.predict(X_test)
mae_test = mean_absolute_error(Y_test, Y_test_pred)
rmse_test = root_mean_squared_error(Y_test, Y_test_pred)
r2_test = r2_score(Y_test, Y_test_pred)
print("\nLinear Regression model test set performance:")
print(f"  Mean Absolute Error (MAE): {mae_test:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test:.2f}")
print(f"  R-squared: {r2_test:.3f}")   

#Test-set evaluation - Random Forest
Y_test_pred_rf = rf_model.predict(X_test)
mae_test_rf = mean_absolute_error(Y_test, Y_test_pred_rf)
rmse_test_rf = root_mean_squared_error(Y_test, Y_test_pred_rf)
r2_test_rf = r2_score(Y_test, Y_test_pred_rf)
print("\nRandom Forest model test set performance:")
print(f"  Mean Absolute Error (MAE): {mae_test_rf:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test_rf:.2f}")
print(f"  R-squared: {r2_test_rf:.3f}")

