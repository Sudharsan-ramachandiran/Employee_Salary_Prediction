from sklearn.metrics import mean_squared_error

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def train_nn(X_train, y_train, X_test, y_test):
    if not TENSORFLOW_AVAILABLE:
        return None, None

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Neural Network MSE: {mse:.2f}")
    return model, predictions

 
