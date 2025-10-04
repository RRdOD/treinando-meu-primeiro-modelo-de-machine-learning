import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.data_preparation import load_data

def train():
    mlflow.set_experiment("IceCreamSales")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data()
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "modelo_vendas")

        print("Modelo treinado com MSE:", mse)

if __name__ == "__main__":
    train()
