import pandas as pd
from pathlib import Path


# Get the folder where this file is located
BASE_DIR = Path(__file__).resolve().parent

# Dataset location
DATA_PATH = BASE_DIR / "sorted_data.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df