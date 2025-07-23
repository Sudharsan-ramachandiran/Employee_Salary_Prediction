from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def train_xgb(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1,
        reg_lambda=2,
        random_state=42,
        verbosity=1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"XGBoost MSE: {mse:.2f}")
    return model, predictions
