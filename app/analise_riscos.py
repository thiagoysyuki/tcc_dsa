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
st.write("# Análise de Riscos")

data = st.session_state['filtered_data']
if data is None:
    st.error("Nenhum dado encontrado.")
else:
    st.success("Dados carregados com sucesso!")
    
prices_hist_graph = px.line(title='Distribuição diária de retornos', labels={'x': 'Retornos', 'y': 'Frequência'})
for column in data.columns:
    prices_hist_graph.add_scatter(x=data.index, y=data[column], name=column, mode='lines')

st.plotly_chart(prices_hist_graph, use_container_width=True)

st.write("## Drawdown Máximo")	

wealth_index =  (1 + st.session_state['simple_returns']).cumprod()
previous_peaks = wealth_index.cummax()
drawdown = (wealth_index - previous_peaks) / previous_peaks

tabs = st.tabs(list(drawdown.columns))
for i, tab_name in enumerate(drawdown.columns):
    with tabs[i]:
        fig, ax = plt.subplots()
        ax.plot(drawdown.index, drawdown[tab_name], label='Quedas', color='red')
        ax.plot(previous_peaks.index, previous_peaks[tab_name], label='Picos', color='green', linestyle='--', alpha=0.5)
        ax.plot(wealth_index.index, wealth_index[tab_name], label='Índice Saúde', color='blue')
        ax.vlines(drawdown.index[drawdown[tab_name] == drawdown[tab_name].min()], ymin=0, ymax=drawdown[tab_name].min(), color='black', lw=1, ls='--')
        ax.axhline(0, color='black', lw=1, ls='--')
        ax.fill_between(drawdown.index, drawdown[tab_name], color='red', alpha=0.5)
        ax.set_title(f"Drawdown Máximo - {tab_name}")
        ax.legend()
        st.write(f"**{tab_name}**")

        st.pyplot(fig)

        st.write("## Máximo Drawdown")
        st.write(f"Máximo Drawdown: {drawdown[tab_name].min()}")
        st.write(f"Data: {drawdown[tab_name].idxmin().date().strftime('%d/%m/%Y')}")
        

   