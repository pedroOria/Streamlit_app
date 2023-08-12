import streamlit as st
import requests
from PIL import Image

with st.sidebar:
    st.title("Entrenamiento del modelo")
    st.write("Ingrese el valor de k en el siguiente campo")
    k_value = st.slider("k:",0,10)

    if k_value:
        
        response_acc = requests.get("http://127.0.0.1:8502/train",
                     params={
                         "k_value":k_value
                    })
        st.write(f"Accuracy: {response_acc.json()}")


with st.container():
    st.title("Predicción del modelo")
    st.write(f"Valor de k: {k_value}")
    SepalLengthCm = st.slider("SepalLengthCm:",0.0,10.0)
    SepalWidthCm = st.slider("SepalWidthCm:",0.0,10.0)
    PetalLengthCm = st.slider("PetalLengthCm:",0.0,10.0)
    PetalWidthCm = st.slider("PetalWidthCm:",0.0,10.0)
    response_pred = requests.get("http://127.0.0.1:8502/predict",
                     params={
                         "SepalLengthCm":SepalLengthCm,
                         "SepalWidthCm":SepalWidthCm,
                         "PetalLengthCm":PetalLengthCm,
                         "PetalWidthCm":PetalWidthCm,
                         "k_value":k_value
                    })
    resultado = response_pred.content.decode("utf-8")
    if resultado == "Iris-setosa":
        image = Image.open('media/setosa.png')
        st.image(image,caption='Resultado',width=150)
    elif resultado == "Iris-versicolor":
        image = Image.open('media/Iris_versicolor.png')
        st.image(image,caption='Resultado',width=150)
    elif resultado == "Iris-virginica":
        image = Image.open('media/iris_virginica.png')
        st.image(image,caption='Resultado',width=150)

    st.success(f"Predicción: {response_pred.content}", icon="📊")