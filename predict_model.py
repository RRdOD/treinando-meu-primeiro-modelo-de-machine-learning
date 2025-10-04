import mlflow.sklearn

def predict(temperatura):
    model = mlflow.sklearn.load_model("runs:/<ID_DA_EXECUÇÃO>/modelo_vendas")
    return model.predict([[temperatura]])
