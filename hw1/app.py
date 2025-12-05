import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from io import StringIO

from app_graphics import print_eda_graphics
from df_modification import *


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "optimal_ridge_name.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


st.title("Предсказание цен на машины")

df_train = pd.read_csv('https://raw.githubusercontent.com/LiyaKul/machine_learning_course/refs/heads/hw1/hw1/df_train.csv')
print_eda_graphics(df_train)
df_train.drop(['selling_price'], axis=1, inplace=True)
st.title("Ввод данных: CSV или ручной ввод")

mode = st.radio(
    "Выберите способ ввода данных:",
    ("Ручной ввод данных", "Загрузить CSV")
)
df = None

if mode == "Загрузить CSV":
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Файл успешно загружен!")
        st.write(df)

elif mode == "Ручной ввод данных":
    columns = list(df_train.columns)

    st.write(f"Введите данные без заголовка, колонки в порядке: {', '.join(columns)}")

    example = "0,Maruti Swift Dzire VDI,2014,145500,Diesel,Individual,Manual,First Owner,23.4,1248,88.7,5"
    text = st.text_area(
        "Каждая строка — отдельная запись, значения разделены запятой:",
        example,
        height=200
    )
    temp_df = pd.read_csv(StringIO(text), header=None)

    if temp_df.shape[1] != len(columns):
        st.error(f"Ожидалось {len(columns)} столбцов, а получено {temp_df.shape[1]}")
    else:
        temp_df.columns = columns
        df = temp_df
        st.success("Данные успешно преобразованы в DataFrame!")
        st.write(df)

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
modify(df)
print_eda_graphics(df, "test")

X_test_cat = df
add_name(df_train)
add_name(X_test_cat)
_, X_test_cat = modify_cat(df_train, X_test_cat)

ridge = load_model()
predict = ridge.predict(X_test_cat)

df['prediction'] = predict
st.subheader("Предсказание:")
st.write(df)

coefs = ridge.coef_[0]
feature_names = X_test_cat.columns
weights = pd.DataFrame({
    "feature": feature_names,
    "weight": ridge.coef_.flatten()
}).sort_values("weight", ascending=False)
st.subheader("Веса логистической регрессии")
st.dataframe(weights.style.background_gradient(cmap="coolwarm"))
