import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json

from pypfopt.expected_returns import mean_historical_return, ema_historical_return

 

# Load Data
st.write("# Análise de Portfólio de Investimentos")

selic = pd.read_parquet(path= 'data/raw/selic/selic.parquet')
selic.set_index('date', inplace=True)
selic['value'] = selic['value'].astype(float)
stocks = pd.read_parquet(path= 'data/processed/merged/merged_prices_10y.parquet')
indices = pd.read_parquet(path= 'data/processed/merged/merged_indexes_10y.parquet')
indices.columns = indices.columns.str.replace('^', '', regex=False)

trading_days = stocks.index.to_series().groupby(stocks.index.year).nunique()
trading_days = pd.DataFrame({'ano':trading_days.index,'dias_uteis':trading_days.values})

st.write("Horizonte disponível:",trading_days.T)

tickers = stocks.columns
indexes = indices.columns

# Seletores

seletor_stock = st.multiselect("Ações", tickers,placeholder="Selecione as ações", default=['ITUB4', 'B3SA3', 'PETR4'])
seletor_index = st.multiselect("Índices", indexes,placeholder="Selecione os índices")

filtered_index_data = indices[seletor_index]
filtered_stock_data = stocks[seletor_stock]

sel_data = st.date_input("Selecione o intervalo de datas", value=(filtered_stock_data.index.min(), filtered_stock_data.index.max()), format="DD/MM/YYYY")

#Filtrar dados

filtered_stock_data = filtered_stock_data.loc[sel_data[0]:sel_data[1]]
filtered_index_data = filtered_index_data.loc[sel_data[0]:sel_data[1]]

filtered_data = pd.merge(filtered_stock_data, filtered_index_data, left_index=True, right_index=True, how='inner')


simple_returns = filtered_data.pct_change().dropna()
log_returns = np.log(filtered_data / filtered_data.shift(1)).dropna()

cum_return = filtered_data.copy()

for i in range(cum_return.shape[1]):  # Itera pelas colunas usando índices
    cum_return.iloc[:, i] = cum_return.iloc[:, i] / cum_return.iloc[0, i]

st.write("Retorno Total Acumulado" ) 
st.write(cum_return.tail(1))


stocks_graph = px.line(title='Retorno Acumulado', labels={'x': 'Índice (Datas)', 'y': 'Retorno simples'})
for column in cum_return.columns:
    stocks_graph.add_scatter(x=cum_return.index, y=cum_return[column], mode='lines', name=column)

# Grafico das ações

st.plotly_chart(stocks_graph, use_container_width=True)

st.write("# Análise dos Retornos:")

seletor_geometric = st.selectbox("Selecione o método de cálculo", ["Geométrico", "Aritmético"])
if seletor_geometric == "Geométrico":
    st.write("Retornos anualizados (Geométrico):")
    mu = mean_historical_return(filtered_data, frequency=252, compounding=True).to_frame()
else:
    st.write("Retornos anualizados (Aritmético):")
    mu = mean_historical_return(filtered_data, frequency=252, compounding=False).to_frame()

mu = mu.reset_index()
mu.columns = ['Ticker', 'Retorno Anualizado']
st.dataframe(mu, use_container_width=True)


st.write("Retorno Anualizados (EMA):")
span_select = st.slider("Selecione o período de suavização (EMA)", min_value=1, max_value=365, value=30, step=1)
st.write(f"Período de suavização (EMA): {span_select} dias")
ema_return = ema_historical_return(filtered_data, span=span_select, frequency=250).to_frame()
ema_return = ema_return.reset_index()
ema_return.columns = ['Ticker', 'Retorno Anualizado']
st.dataframe(ema_return, use_container_width=True)

st.write("Taxa SELIC")

selic_graph = px.line(selic, x=selic.index,y=selic['value'], title='Taxa SELIC', labels={'x': 'Índice (Datas)', 'y': 'Taxa SELIC'})
st.plotly_chart(selic_graph, use_container_width=True)


ir_analse_riscos = st.button("Analisar Risco", key="analyze_risk")

if ir_analse_riscos:
    st.session_state['filtered_data'] = filtered_data
    st.session_state['simple_returns'] = simple_returns
    st.session_state['log_returns'] = log_returns
    st.session_state['selic'] = selic
    st.session_state['trading_days'] = trading_days

    st.switch_page("analise_riscos.py")
    