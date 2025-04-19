import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import os

st.write("# Análise de Portfólio de Investimentos")

data = pd.read_parquet(path= 'data\processed\merged\merged_prices_10y.parquet')
tickers = json.load(open("data\lista_tickers.json", "r"))["stocks"]

seletor = st.multiselect("Ações", tickers)
filtered_data = data[seletor]
sel_data = st.date_input("Selecione o intervalo de datas", value=(filtered_data.index.min(), filtered_data.index.max()))
filtered_data = filtered_data.loc[sel_data[0]:sel_data[1]]
st.write("Intervalo de datas selecionado:")
st.write(sel_data[0], sel_data[1])


fig, ax = plt.subplots()
for column in filtered_data.columns:
    ax.plot(filtered_data.index, filtered_data[column], label=column)  # Adiciona uma linha para cada coluna

plt.title('Evolução das Ações')
plt.xlabel('Índice (Datas)')
plt.ylabel('Valores ajustados')
plt.legend(title='Colunas')  # Legenda com título
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo X (se necessário)
plt.tight_layout()  # Ajusta o layout para evitar sobreposição

st.pyplot(fig)

#st.line_chart(data=filtered_data, x=filtered_data.index, y=)

returns = filtered_data.pct_change().dropna()

fig, ax = plt.subplots()
for column in returns.columns:
    ax.plot(returns.index, returns[column], label=column)  # Adiciona uma linha para cada coluna

plt.title('Evolução das Ações')
plt.xlabel('Índice (Datas)')
plt.ylabel('Valores ajustados')
plt.legend(title='Colunas')  # Legenda com título
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo X (se necessário)
plt.tight_layout()  # Ajusta o layout para evitar sobreposição
st.pyplot(fig)


mu = mean_historical_return(filtered_data, frequency=252).to_frame()
mu = mu.reset_index()
mu.columns = ['Ticker', 'Retorno Anualizado']
S = CovarianceShrinkage(filtered_data).ledoit_wolf()
st.write(
    "Média de Retornos Anuais: ", mu, "Covariância: ", S
)
