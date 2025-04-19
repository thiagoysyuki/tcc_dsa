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

# Load Data
st.write("# Análise de Portfólio de Investimentos")

data = pd.read_parquet(path= 'https://github.com/thiagoysyuki/tcc_dsa/blob/main/data/processed/merged/merged_indexes_10y.parquet?raw=true')
tickers = json.load(open("https://github.com/thiagoysyuki/tcc_dsa/blob/main/data/lista_tickers.json?raw=true", "r"))["stocks"]

# Seletores

seletor = st.multiselect("Ações", tickers,placeholder="Selecione as ações", default=['ITUB4', 'B3SA3', 'PETR4'])
filtered_data = data[seletor]

sel_data = st.date_input("Selecione o intervalo de datas", value=(filtered_data.index.min(), filtered_data.index.max()), format="DD/MM/YYYY")

#Filtrar dados

filtered_data = filtered_data.loc[sel_data[0]:sel_data[1]]

filtered_data_norm = filtered_data.copy()

for i in filtered_data_norm.columns:
    filtered_data_norm[i] = (filtered_data_norm[i]) / (filtered_data_norm[i][0])

st.write("Ganho percentual:" ) 
st.write(filtered_data_norm.tail(1))


fig, ax = plt.subplots()
for column in filtered_data_norm.columns:
    ax.plot(filtered_data_norm.index, filtered_data_norm[column], label=column)  # Adiciona uma linha para cada coluna

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

#px.line(returns, x=returns.index, y=returns.columns, title='Evolução das Ações', labels={'x': 'Índice (Datas)', 'y': 'Valores ajustados'}).show()

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