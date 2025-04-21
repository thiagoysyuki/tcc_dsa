import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from misc.otm_oportfolio import var_gaussian
import os

selic = st.session_state['selic']
data = st.session_state['filtered_data']
simple_returns = st.session_state['simple_returns']
log_returns = st.session_state['log_returns']
free_rate = selic['value'].tail(1).values[0]/100

if data is None:
    st.error("Nenhum dado encontrado.")
else:
    st.success("Dados carregados com sucesso!")

## Statisticas dos retornos + Sharpe ratio

simple_returns_stats = simple_returns.describe().T

simple_annualized_returns = pd.DataFrame({
    'Retorno Anualizado': simple_returns_stats['mean'] * 250,
    'Volatilidade Anualizada': simple_returns_stats['std'] * np.sqrt(250)
})

simple_annualized_returns['Sharpe ratio'] = simple_annualized_returns['Retorno Anualizado'] - free_rate / simple_annualized_returns['Volatilidade Anualizada']

log_returns_stats = log_returns.describe().T

log_annualized_returns = pd.DataFrame({
    'Retorno Anualizado': log_returns_stats['mean'] * 250,
    'Volatilidade Anualizada': log_returns_stats['std'] * np.sqrt(250)
})

log_annualized_returns['Sharpe ratio'] = log_annualized_returns['Retorno Anualizado'] - free_rate / log_annualized_returns['Volatilidade Anualizada']

st.write("Taxa SELIC atual é:", free_rate)

simples, log = st.tabs(["Simples","Logarítmico"])

with simples:
    st.write("# Análise de Riscos")

    tabs_desity = st.tabs(list(simple_returns.columns))
    for i, tab_name in enumerate(simple_returns.columns):
        with tabs_desity[i]:
            fig, ax = plt.subplots()
            sns.kdeplot(data=simple_returns[tab_name], ax=ax, label='Kernel Density', color='blue')
            sns.histplot(data=simple_returns[tab_name], ax=ax, bins=30, kde=True, stat='density', alpha=0.5)
            ax.set_title(f"Distribuição de Retornos - {tab_name}")
            ax.legend()
            st.pyplot(fig)


    st.write("Estatísticas dos retornos diários simples:")
    st.dataframe(simple_returns_stats, use_container_width=True)
    st.write("Estatísticas dos retornos anualizados:")
    st.dataframe(simple_annualized_returns, use_container_width=True)

    st.write("## Desvios da Normalidade")

    skew_table = pd.DataFrame(columns=[
        'Ação', 'Média', 'Desvio Padrão', 'Skewness', 'Kurtosis', 
        'Jarque-Bera', 'p-value', 'Distribuição'
    ])

    for column in simple_returns.columns:
        row = {
            'Ação': column,
            'Média': simple_returns[column].mean(),
            'Desvio Padrão': simple_returns[column].std(),
            'Skewness': simple_returns[column].skew(),
            'Kurtosis': simple_returns[column].kurtosis(),
            'Jarque-Bera': stats.jarque_bera(simple_returns[column])[0],
            'p-value': stats.jarque_bera(simple_returns[column])[1],
            'Distribuição': 'Normal' if abs(simple_returns[column].skew()) < 0.5 and abs(simple_returns[column].kurtosis()) < 3 else 'Não Normal'
        }
        skew_table = pd.concat([skew_table, pd.DataFrame([row])], ignore_index=True)

    # Format the 'p-value' column to scientific notation
    skew_table['p-value'] = skew_table['p-value'].apply(lambda x: f"{x:.2e}")

    st.write(skew_table)

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
            plt.xticks(rotation=90)  # Rotate x-axis labels 90 degrees
            st.write(f"**{tab_name}**")

            st.pyplot(fig)

    st.write("## Máximo Drawdown")
    st.write("Máximo Drawdown:")
    max_drawndow = pd.DataFrame({'Máximo Drawdown': drawdown.min(), 'Data do Máximo Drawdown': drawdown.idxmin()})
    max_drawndow['Data do Máximo Drawdown'] = max_drawndow['Data do Máximo Drawdown'].dt.strftime('%d/%m/%Y')
    st.write(max_drawndow)


with log:
    
    st.write("# Análise de Riscos")

    tabs_desity_lg = st.tabs(list(log_returns.columns))
    for i, tab_name in enumerate(log_returns.columns):
        with tabs_desity_lg[i]:
            fig, ax = plt.subplots()
            sns.kdeplot(data=log_returns[tab_name], ax=ax, label='Kernel Density', color='blue')
            sns.histplot(data=log_returns[tab_name], ax=ax, bins=30, kde=True, stat='density', alpha=0.5)
            ax.set_title(f"Distribuição de Retornos - {tab_name}")
            ax.legend()
            st.pyplot(fig)



    st.write("Estatísticas dos retornos diários logarítmicos:")
    st.dataframe(log_returns_stats, use_container_width=True)
    st.write("Estatísticas dos retornos logarítmicos anualizados:")
    st.dataframe(log_annualized_returns, use_container_width=True)
    
    st.write("## Desvios da Normalidade")
    skew_table_lg = pd.DataFrame(columns=[
        'Ação', 'Média', 'Desvio Padrão', 'Skewness', 'Kurtosis', 
        'Jarque-Bera', 'p-value', 'Distribuição'
    ])

    for column in log_returns.columns:
        row = {
            'Ação': column,
            'Média': log_returns[column].mean(),
            'Desvio Padrão': log_returns[column].std(),
            'Skewness': log_returns[column].skew(),
            'Kurtosis': log_returns[column].kurtosis(),
            'Jarque-Bera': stats.jarque_bera(log_returns[column])[0],
            'p-value': stats.jarque_bera(log_returns[column])[1],
            'Distribuição': 'Normal' if abs(log_returns[column].skew()) < 0.5 and abs(log_returns[column].kurtosis()) < 3 else 'Não Normal'
        }
        skew_table_lg = pd.concat([skew_table_lg, pd.DataFrame([row])], ignore_index=True)

    # Format the 'p-value' column to scientific notation
    skew_table_lg['p-value'] = skew_table_lg['p-value'].apply(lambda x: f"{x:.2e}")

    st.write(skew_table_lg)

    st.write("## Drawdown Máximo")	

    wealth_index_lg =  (1 + st.session_state['log_returns']).cumprod()
    previous_peaks_lg = wealth_index.cummax()
    drawdown_lg = (wealth_index - previous_peaks) / previous_peaks

    tabs_lg = st.tabs(list(drawdown_lg.columns))
    for i, tab_name in enumerate(drawdown_lg.columns):
        with tabs_lg[i]:
            fig, ax = plt.subplots()
            ax.plot(drawdown_lg.index, drawdown_lg[tab_name], label='Quedas', color='red')
            ax.plot(previous_peaks.index, previous_peaks[tab_name], label='Picos', color='green', linestyle='--', alpha=0.5)
            ax.plot(wealth_index.index, wealth_index[tab_name], label='Índice Saúde', color='blue')
            ax.vlines(drawdown_lg.index[drawdown_lg[tab_name] == drawdown_lg[tab_name].min()], ymin=0, ymax=drawdown_lg[tab_name].min(), color='black', lw=1, ls='--')
            ax.axhline(0, color='black', lw=1, ls='--')
            ax.fill_between(drawdown_lg.index, drawdown_lg[tab_name], color='red', alpha=0.5)
            ax.set_title(f"Drawdown Máximo - {tab_name}")
            ax.legend()
            plt.xticks(rotation=90)  # Rotate x-axis labels 90 degrees
            st.write(f"**{tab_name}**")

            st.pyplot(fig)

    st.write("## Máximo Drawdown")
    st.write("Máximo Drawdown:")
    max_drawndow_lg = pd.DataFrame({'Máximo Drawdown': drawdown_lg.min(), 'Data do Máximo Drawdown': drawdown_lg.idxmin()})
    max_drawndow_lg['Data do Máximo Drawdown'] = max_drawndow_lg['Data do Máximo Drawdown'].dt.strftime('%d/%m/%Y')

    st.write(max_drawndow_lg)

    st.write("## Value at Risk (VaR)")

    historical_var = pd.DataFrame(log_returns.quantile(0.05))
    historical_var.reset_index(inplace=True)
    historical_var.columns = ['Ticker', 'Histórico']
    historical_var['Histórico'] = historical_var['Histórico'] * -1

    z = stats.norm.ppf(0.05)
    VaR_parametric = pd.DataFrame(
        (log_returns.mean() - z * log_returns.std())
    )

    VaR_parametric.reset_index(inplace=True)
    VaR_parametric.columns = ['Ticker', 'Gaussiano']

    VaR_fs = pd.DataFrame(var_gaussian(log_returns, level=0.05, modified=True))
    VaR_fs.reset_index(inplace=True)
    VaR_fs.columns = ['Ticker', 'Cornish-Fisher']
    
    VaR_log = pd.merge(historical_var, VaR_parametric, on='Ticker')
    VaR_log = pd.merge(VaR_log, VaR_fs, on='Ticker')

    VaR_graph =px.bar(VaR_log, x='Ticker', y=['Histórico', 'Gaussiano', 'Cornish-Fisher'], title='Value at Risk (VaR)', labels={'x': 'Ação', 'y': 'VaR'})
    VaR_graph.update_layout(barmode='group', xaxis_title='Ação', yaxis_title='VaR')

    st.plotly_chart(VaR_graph, use_container_width=True)

    st.write(VaR_log)







