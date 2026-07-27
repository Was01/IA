import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn import linear_model


#  Carregando os dados do arquivo CSV

df=pd.read_csv('homeprices.csv')

# Treinamento do modelo de regressão linear

model=linear_model.LinearRegression();

model.fit(df[['area']],df[['price']]);

st.title('Previsão de Preços de Casas')
st.divider()

area=st.number_input('Digite a área da casa em metros quadrados:', min_value=0.0, format="%.2f")

if area>0:
    price=model.predict(pd.DataFrame({'area':[area]}))
    st.write(f'O preço previsto para uma casa com {area} metros quadrados é: R$ {price[0][0]:.2f}')