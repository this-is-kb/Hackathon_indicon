import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Configuration
DATA_PATH = 'Datasheet_kaggle.csv'  # Replace with your dataset path
TARGET_COLUMN = 'lengthofstay'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 5

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv(file_path)

    # Drop irrelevant columns
    data = data.drop(columns=['eid', 'discharged'])

    # Convert date to datetime and extract features
    data['vdate'] = pd.to_datetime(data['vdate'], errors='coerce')
    data['admit_month'] = data['vdate'].dt.month
    data['admit_day'] = data['vdate'].dt.day
    data['admit_weekday'] = data['vdate'].dt.weekday
    data = data.drop(columns=['vdate'])

    # Encode categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        data[col] = label_encoders[col].fit_transform(data[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    return data_imputed, label_encoders

def split_and_normalize_data(data, target_column):
    """Split data into training and testing sets and normalize features."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

def evaluate_models(models, X_train, y_train, X_test, y_test, cv_splits):
    """Train and evaluate models using cross-validation and test set."""
    results = {}
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        # Cross-validation R² scores
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_mean_r2 = cv_r2_scores.mean()
        cv_std_r2 = cv_r2_scores.std()

        # Train and predict on the test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate on test set
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store results
        results[name] = {
            "Cross-Validation R² Scores": cv_r2_scores.tolist(),
            "Mean R² (CV)": cv_mean_r2,
            "Std R² (CV)": cv_std_r2,
            "MAE (Test)": mae,
            "MSE (Test)": mse,
            "R² (Test)": r2
        }
    return results

def print_results(results):
    """Print model evaluation results."""
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")

def main():
    # Load and preprocess the dataset
    data, _ = load_and_preprocess_data(DATA_PATH)

    # Split and normalize data
    X_train, X_test, y_train, y_test = split_and_normalize_data(data, TARGET_COLUMN)

    # Define models
    models = {
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='rmse')
    }

    # Train and evaluate models
    results = evaluate_models(models, X_train, y_train, X_test, y_test, CV_SPLITS)

    # Print results
    print_results(results)

if __name__ == "__main__":
    main()

