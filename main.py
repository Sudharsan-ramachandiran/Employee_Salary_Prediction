from src.trainer import train_model
from src.visualizer import plot_prediction

def main():
    zip_file = "data/archive.zip"
    extract_dir = "data/archive"

    y_test_actual, y_pred_actual = train_model(zip_file, extract_dir)

    plot_prediction(y_test_actual, y_pred_actual)

if __name__ == "__main__":
    main()
