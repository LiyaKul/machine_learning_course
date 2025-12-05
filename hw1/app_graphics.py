import streamlit as st
import pandas as pd
import seaborn as sns
# from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns


def print_head(df):
    st.subheader("Первые строки датасета")
    st.write(df.head())


def print_num_statistic(num_df):
    st.subheader("Описательные статистики по числовым столбцам")
    st.write(num_df.describe().T)


def print_cat_statistic(cat_df, df_type="train"):
    st.subheader("Описательные статистики по категориальным столбцам")
    stats = {}
    for col in cat_df.columns:
        series = cat_df[col].astype("category")
        stats[col] = {
            "Тип": str(series.dtype),
            "Количество уникальных": series.nunique(),
            "Мода (самое частое)": series.mode().iloc[0] if not series.mode().empty else None,
        }
    st.write(pd.DataFrame(stats).T)

    st.subheader("Частоты категорий (value_counts)")
    cat_col = st.selectbox(
        "Выберите категоріальный столбец",
        cat_df.columns,
        key=f"{df_type}_cat_col_selectbox",
    )
    st.write(cat_df[cat_col].value_counts())
    st.write(cat_df[cat_col].value_counts(normalize=True).rename("proportion"))


def print_pairplot(num_df, df_type):
    st.subheader("Pairplot (Seaborn)")
    st.caption("Осторожно: на больших датасетах может быть очень медленно.")

    max_rows = st.slider(
        "Максимум строк для pairplot",
        100, 2000, 500,
        key=f"{df_type}_pairplot_max_rows_slider",
    )
    sample_df = num_df.sample(n=min(len(num_df), max_rows), random_state=42)

    if st.button("Построить pairplot", key=f"{df_type}_pairplot_button"):
        g = sns.pairplot(sample_df)
        st.pyplot(g.figure)


def print_heatmap(num_df):
    st.subheader("Тепловая карта корреляции (Пирсона)")
    corr = num_df.corr(method="pearson")

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Матрица корреляции Пирсона")
    st.pyplot(fig_corr)


# def print_profile_report(df, df_type="train"):
#     if st.button(
#         "Сгенерировать отчёт streamlit_ydata_profiling",
#         key=f"{df_type}_profile_button",
#     ):
#         profile = ProfileReport(df, explorative=True)
#         st_profile_report(profile)


def print_eda_graphics(df, df_type="train"):
    print_head(df)

    num_df = df.select_dtypes(include="number")
    print_num_statistic(num_df)

    cat_df = df.select_dtypes(include=["object"])
    if not cat_df.empty:
        print_cat_statistic(cat_df, df_type)

    print_pairplot(num_df, df_type)

    print_heatmap(num_df)

    # print_profile_report(df, df_type)
