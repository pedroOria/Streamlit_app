from flask import Flask, request
import pandas as pd
from knn_model import IrisModel

app = Flask("__main__")

data = pd.read_csv("https://raw.githubusercontent.com/luisfernandosolis/streamlit-mldashboard/main/Iris.csv")


@app.route("/train", methods = ["GET"])
def train():
    k_value = request.args.get("k_value")
    model_train = IrisModel(dataset=data)  
    acc_result = model_train.train(int(k_value))
    return str(acc_result)


@app.route("/predict", methods = ["GET"])
def predict():
    SepalLengthCm = request.args.get("SepalLengthCm")
    SepalWidthCm = request.args.get("SepalWidthCm")
    PetalLengthCm = request.args.get("PetalLengthCm")
    PetalWidthCm = request.args.get("PetalWidthCm")
    k_value = request.args.get("k_value")
    model_pred= IrisModel(data)
    acc_result = model_pred.train(int(k_value))
    predict_result = model_pred.predict(float(SepalLengthCm),float(SepalWidthCm),float(PetalLengthCm),float(PetalWidthCm))
    return str(predict_result)


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8502, debug= True)