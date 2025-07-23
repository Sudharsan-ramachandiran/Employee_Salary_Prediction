from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_rf(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Random Forest MSE: {mse:.2f}")
    return model, predictions

# Demo block to run this file directly
if __name__ == "__main__":
    import numpy as np
    # Generate dummy data for demonstration
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100) * 100
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20) * 100
    train_rf(X_train, y_train, X_test, y_test)


