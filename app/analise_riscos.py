import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from pypfopt.expected_returns import mean_historical_return
from misc.otm_oportfolio import var_gaussian,cvar_historic, semideviation
import os
import statsmodels.api as sm

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

st.write("Taxa SELIC atual é:", free_rate)

st.write("Estatísticas dos retornos diários simples:")


simples, log = st.tabs(["Simples","Logarítmico"])

with simples:

    st.write("# Análise de Riscos")

    retorno_anualizado = pd.DataFrame(mean_historical_return(data, log_returns=False, frequency=252))
    retorno_anualizado.reset_index(inplace=True)
    retorno_anualizado.columns = ['Ações','Retorno Anualizado']
    volatilidade_anualizada = pd.DataFrame(simple_returns.std() * 252 ** 0.5)
    volatilidade_anualizada.reset_index(inplace=True)
    volatilidade_anualizada.columns = ['Ações','Volatilidade Anualizada']
    dados_anualizados = pd.merge(retorno_anualizado, volatilidade_anualizada, on='Ações')
    dados_anualizados['Sharpe Ratio'] = (dados_anualizados['Retorno Anualizado'] - free_rate) / dados_anualizados['Volatilidade Anualizada']

        # Add a regression line to the scatter plot
    fig = px.scatter(
        title='Retorno vs. Volatilidade',
        data_frame=dados_anualizados,
        x='Retorno Anualizado',
        y='Volatilidade Anualizada',
        labels={'x': 'Retorno Anual', 'y': 'Volatilidade Anual'},
        trendline="ols",  # Add Ordinary Least Squares (OLS) regression line
        trendline_color_override='red',  # Change the color of the regression line

    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(dados_anualizados, use_container_width=True)

    tabs_desity = st.tabs(list(simple_returns.columns))
    for i, tab_name in enumerate(simple_returns.columns):
        with tabs_desity[i]:
            fig, ax = plt.subplots()
            sns.kdeplot(data=simple_returns[tab_name], ax=ax, label='Kernel Density', color='blue')
            sns.histplot(data=simple_returns[tab_name], ax=ax, bins=30, kde=True, stat='density', alpha=0.5)
            ax.set_title(f"Distribuição de Retornos - {tab_name}")
            ax.legend()
            st.pyplot(fig)
    
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

    st.write("## Value at Risk (VaR)")

    historical_var = pd.DataFrame(simple_returns.quantile(0.05))
    historical_var.reset_index(inplace=True)
    historical_var.columns = ['Ação', 'Histórico']
    historical_var['Histórico'] = historical_var['Histórico'] * -1

    z = stats.norm.ppf(0.05)
    VaR_parametric = pd.DataFrame(
        (simple_returns.mean() - z * simple_returns.std())
    )

    VaR_parametric.reset_index(inplace=True)
    VaR_parametric.columns = ['Ação', 'Gaussiano']

    VaR_fs = pd.DataFrame(var_gaussian(simple_returns, level=0.05, modified=True))
    VaR_fs.reset_index(inplace=True)
    VaR_fs.columns = ['Ação', 'Cornish-Fisher']

    beyond_var_lg =pd.DataFrame(cvar_historic(simple_returns, level=0.05))
    beyond_var_lg.reset_index(inplace=True)
    beyond_var_lg.columns = ['Ação', 'Beyond VaR']
    
    VaR_simple = pd.merge(historical_var, VaR_parametric, on='Ação')
    VaR_simple = pd.merge(VaR_simple, VaR_fs, on='Ação')
    VaR_simple = pd.merge(VaR_simple, beyond_var_lg, on='Ação')

    VaR_graph =px.bar(VaR_simple, x='Ação', y=['Histórico', 'Gaussiano', 'Cornish-Fisher'], title='Value at Risk (VaR)', labels={'x': 'Ação', 'y': 'VaR'})
    VaR_graph.update_layout(barmode='group', xaxis_title='Ação', yaxis_title='VaR')

    st.plotly_chart(VaR_graph, use_container_width=True)

    st.write(VaR_simple)



with log:
    
    st.write("# Análise de Riscos")    

    retorno_anualizado_lg = pd.DataFrame(mean_historical_return(data, log_returns=True, frequency=252))
    retorno_anualizado_lg.reset_index(inplace=True)
    retorno_anualizado_lg.columns = ['Ações','Retorno Anualizado']
    volatilidade_anualizada_lg = pd.DataFrame(log_returns.std() * 252 ** 0.5)
    volatilidade_anualizada_lg.reset_index(inplace=True)
    volatilidade_anualizada_lg.columns = ['Ações','Volatilidade Anualizada']
    dados_anualizados_lg = pd.merge(retorno_anualizado_lg, volatilidade_anualizada_lg, on='Ações')
    dados_anualizados_lg['Sharpe Ratio'] = (dados_anualizados_lg['Retorno Anualizado'] - free_rate) / dados_anualizados_lg['Volatilidade Anualizada']

    # Add a regression line to the scatter plot
    fig = px.scatter(
        title='Retorno vs. Volatilidade',
        data_frame=dados_anualizados_lg,
        x='Retorno Anualizado',
        y='Volatilidade Anualizada',
        labels={'x': 'Retorno Anual', 'y': 'Volatilidade Anual'},
        trendline="ols",  # Add Ordinary Least Squares (OLS) regression line
        trendline_color_override='red',  # Change the color of the regression line
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(dados_anualizados_lg, use_container_width=True)

    tabs_desity_lg = st.tabs(list(log_returns.columns))
    for i, tab_name in enumerate(log_returns.columns):
        with tabs_desity_lg[i]:
            fig, ax = plt.subplots()
            sns.kdeplot(data=log_returns[tab_name], ax=ax, label='Kernel Density', color='blue')
            sns.histplot(data=log_returns[tab_name], ax=ax, bins=30, kde=True, stat='density', alpha=0.5)
            ax.set_title(f"Distribuição de Retornos - {tab_name}")
            ax.legend()
            st.pyplot(fig)
    
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
            'Distribuição': 'Normal' if abs(log_returns[column].skew()) < 0.5 and abs(log_returns[column].kurtosis()) < 3 else 'Não Normal',
            'Desvio Semi-condicional': semideviation(log_returns[column])
            
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

    historical_var_lg = pd.DataFrame(log_returns.quantile(0.05))
    historical_var_lg.reset_index(inplace=True)
    historical_var_lg.columns = ['Ação', 'Histórico']
    historical_var_lg['Histórico'] = historical_var['Histórico'] * -1

    z = stats.norm.ppf(0.05)
    VaR_parametric_lg = pd.DataFrame(
        (log_returns.mean() - z * log_returns.std())
    )

    VaR_parametric_lg.reset_index(inplace=True)
    VaR_parametric_lg.columns = ['Ação', 'Gaussiano']

    VaR_fs_lg = pd.DataFrame(var_gaussian(log_returns, level=0.05, modified=True))
    VaR_fs_lg.reset_index(inplace=True)
    VaR_fs_lg.columns = ['Ação', 'Cornish-Fisher']

    beyond_var_lg =pd.DataFrame(cvar_historic(log_returns, level=0.05))
    beyond_var_lg.reset_index(inplace=True)
    beyond_var_lg.columns = ['Ação', 'Beyond VaR']
    
    VaR_log = pd.merge(historical_var_lg, VaR_parametric_lg, on='Ação')
    VaR_log = pd.merge(VaR_log, VaR_fs_lg, on='Ação')
    VaR_log = pd.merge(VaR_log, beyond_var_lg, on='Ação')

    VaR_graph =px.bar(VaR_log, x='Ação', y=['Histórico', 'Gaussiano', 'Cornish-Fisher'], title='Value at Risk (VaR)', labels={'x': 'Ação', 'y': 'VaR'})
    VaR_graph.update_layout(barmode='group', xaxis_title='Ação', yaxis_title='VaR')

    st.plotly_chart(VaR_graph, use_container_width=True)

    st.write(VaR_log)

    semideviation(log_returns)

   









