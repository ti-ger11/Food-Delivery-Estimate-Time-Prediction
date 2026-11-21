"""
Om (Member 2) - Traditional ML models: Linear Regression, Random Forest, XGBoost.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


def train_linear(X_train, y_train):
    """Baseline model - no tuning needed."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression trained")
    return model


def train_random_forest(X_train, y_train):
    """Random Forest tuned with RandomizedSearchCV (3-fold)."""
    params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        params, n_iter=8, cv=3, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=1,
    )
    search.fit(X_train, y_train)
    print(f"Random Forest best params: {search.best_params_}")
    return search.best_estimator_


def train_xgboost(X_train, y_train):
    """XGBoost tuned with RandomizedSearchCV (3-fold)."""
    params = {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        params, n_iter=10, cv=3, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=1,
    )
    search.fit(X_train, y_train)
    print(f"XGBoost best params: {search.best_params_}")
    return search.best_estimator_