import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path='data/ice_cream_sales.csv'):
    df = pd.read_csv(path)
    X = df[['temperatura']]
    y = df['vendas']
    return train_test_split(X, y, test_size=0.2, random_state=42)
