import matplotlib.pyplot as plt

def plot_prediction(actual, predicted):
    plt.figure(figsize=(10, 7))
    plt.scatter(actual, predicted, alpha=0.5, color="steelblue")
    plt.title("Actual vs Predicted Salary")
    plt.xlabel("Actual Salary (USD)")
    plt.ylabel("Predicted Salary (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
