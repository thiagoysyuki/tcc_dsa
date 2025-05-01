import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pypfopt.expected_returns import mean_historical_return, ema_historical_return
import scipy.stats as stats
from misc.otm_oportfolio import var_gaussian,cvar_historic, semideviation
from plotnine import *
from misc.otm_oportfolio import backtest_markowitz
from datetime import datetime, timedelta

plt.rcParams.update({'figure.max_open_warning': 0})

url_indices ="https://drive.google.com/uc?export=download&id=1nk7adCx-VGXKkicD2vmqGxYF4tBZw-1O"
url_stocks = "https://drive.google.com/uc?export=download&id=1erTCQsKviL5wY1D2ZcrDpXE75AT5cyre"
url_selic ="https://drive.google.com/uc?export=download&id=1q1BhAcBipfqPbR4-udpg7yfbwPMsMrWt"

@st.cache_data
def trazer_parquet(path):
    data = pd.read_parquet(path)
    return data
    

selic = trazer_parquet(path= url_selic)
selic.set_index('date', inplace=True)
selic['value'] = selic['value'].astype(float)
stocks = trazer_parquet(path= url_stocks)
stocks = stocks.astype('float')
indices = trazer_parquet(path= url_indices)
indices.columns = indices.columns.str.replace('^', '', regex=False)
trading_days = stocks.index.to_series().groupby(stocks.index.year).nunique()
trading_days = pd.DataFrame({'ano':trading_days.index,'dias_uteis':trading_days.values})
tickers = stocks.columns
indexes = indices.columns



tickers_list = [
    "VALE3",
    "ITUB4",
    "PETR4",
    "ELET3",
    "BBAS3"
]



with st.sidebar:
    st.image("app/img/logo_app.png")
    st.write("## Otimizador de Investimentos")
    st.write("Feito por Thiago Yuki")
    st.write("### Selecione o perído de Análise")
    seletor_stock = st.multiselect("Ações", tickers,placeholder="Selecione as ações", default=tickers_list)
    seletor_index = st.multiselect("Índices", indexes,placeholder="Selecione os índices")
    seletor_selic = st.slider("Selic", min_value=selic['value'].min(),max_value=selic['value'].max(), value=11.75)
    sel_data_inicio = st.date_input("Selecione data de início", value= datetime(2021,1,1), format="DD/MM/YYYY")
    sel_data_fim = st.date_input("Selecione data de fim", value= datetime(2024,12,31), format="DD/MM/YYYY")

retorno_anualizado = pd.DataFrame(mean_historical_return(stocks, log_returns=False, frequency=252))
retorno_anualizado.reset_index(inplace=True)
retorno_anualizado.columns = ['Ações','Retorno Anualizado']
volatilidade_anualizada = pd.DataFrame(stocks.pct_change().dropna().std() * 252 ** 0.5)
volatilidade_anualizada.reset_index(inplace=True)
volatilidade_anualizada.columns = ['Ações','Volatilidade Anualizada']
dados_anualizados = pd.merge(retorno_anualizado, volatilidade_anualizada, on='Ações')
dados_anualizados['Sharpe Ratio'] = (dados_anualizados['Retorno Anualizado'] - seletor_selic) / dados_anualizados['Volatilidade Anualizada']

st.write("## Performance histórica das Ações")
st.write("utilize os indicadores para selecionar as ações que vão compor seu portifólio.")
st.write(dados_anualizados)

#Filtrar dados
filtered_index_data = indices[seletor_index]
filtered_stock_data = stocks[seletor_stock].dropna()
filtered_stock_data = filtered_stock_data[sel_data_inicio:sel_data_fim]
filtered_index_data = filtered_index_data[sel_data_inicio:sel_data_fim]
filtered_data = pd.merge(filtered_stock_data, filtered_index_data, left_index=True, right_index=True, how='inner')

###--------------------------------------------------------------------------------------------------------##

Retornos, Riscos, Otimização = st.tabs(["Análise dos Retornos","Análise Riscos","Otimização"])

with Retornos:    
    simple_returns = filtered_data.pct_change().dropna()
    log_returns = np.log(filtered_data / filtered_data.shift(1)).dropna()

    st.write("## Tabela dos Preços")

    st.dataframe(filtered_data)

    cum_return = filtered_data.copy()

    for i in range(cum_return.shape[1]):  # Itera pelas colunas usando índices
        cum_return.iloc[:, i] = cum_return.iloc[:, i] / cum_return.iloc[0, i]

    st.write("Retorno Total Acumulado" ) 
    st.write(cum_return.tail(1))


    stocks_graph = px.line(title='Preço das Ações', labels={'x': 'Índice (Datas)', 'y': 'Preço R$'})
    for column in filtered_data.columns:
        stocks_graph.add_scatter(x=filtered_data.index, y=filtered_data[column], mode='lines', name=column)

    # Grafico das ações

    st.plotly_chart(stocks_graph)

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
    st.dataframe(mu)


    st.write("Retorno Anualizados (EMA):")
    span_select = st.slider("Selecione o período de suavização (EMA)", min_value=1, max_value=365, value=30, step=1)
    st.write(f"Período de suavização (EMA): {span_select} dias")
    ema_return = ema_historical_return(filtered_data, span=span_select, frequency=250).to_frame()
    ema_return = ema_return.reset_index()
    ema_return.columns = ['Ticker', 'Retorno Anualizado']
    st.dataframe(ema_return)

    st.write("Taxa SELIC")

    selic_graph = px.line(selic, x=selic.index,y=selic['value'], title='Taxa SELIC', labels={'x': 'Índice (Datas)', 'y': 'Taxa SELIC'})
    st.plotly_chart(selic_graph)

with Riscos:
    simples, log = st.tabs(["Simples","Logarítmico"])

    with simples:

        st.write("# Análise de Riscos")

        retorno_anualizado = pd.DataFrame(mean_historical_return(filtered_data, log_returns=False, frequency=252))
        retorno_anualizado.reset_index(inplace=True)
        retorno_anualizado.columns = ['Ações','Retorno Anualizado']
        volatilidade_anualizada = pd.DataFrame(simple_returns.std() * 252 ** 0.5)
        volatilidade_anualizada.reset_index(inplace=True)
        volatilidade_anualizada.columns = ['Ações','Volatilidade Anualizada']
        dados_anualizados = pd.merge(retorno_anualizado, volatilidade_anualizada, on='Ações')
        dados_anualizados['Sharpe Ratio'] = (dados_anualizados['Retorno Anualizado'] - seletor_selic) / dados_anualizados['Volatilidade Anualizada']

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
        st.plotly_chart(fig)

        st.dataframe(dados_anualizados)

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
            #skew_table = pd.concat([skew_table, pd.DataFrame([row])], ignore_index=True, axis=0)

        # Format the 'p-value' column to scientific notation
        #skew_table['p-value'] = skew_table['p-value'].apply(lambda x: f"{x:.2e}")

        #st.write(skew_table)

        st.write("## Drawdown Máximo")	

        wealth_index =  (1 + simple_returns).cumprod()
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

        st.plotly_chart(VaR_graph)

        st.write(VaR_simple)



    with log:

        st.write("# Análise de Riscos")    

        retorno_anualizado_lg = pd.DataFrame(mean_historical_return(filtered_data, log_returns=True, frequency=252))
        retorno_anualizado_lg.reset_index(inplace=True)
        retorno_anualizado_lg.columns = ['Ações','Retorno Anualizado']
        volatilidade_anualizada_lg = pd.DataFrame(log_returns.std() * 252 ** 0.5)
        volatilidade_anualizada_lg.reset_index(inplace=True)
        volatilidade_anualizada_lg.columns = ['Ações','Volatilidade Anualizada']
        dados_anualizados_lg = pd.merge(retorno_anualizado_lg, volatilidade_anualizada_lg, on='Ações')
        dados_anualizados_lg['Sharpe Ratio'] = (dados_anualizados_lg['Retorno Anualizado'] - seletor_selic) / dados_anualizados_lg['Volatilidade Anualizada']

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
        st.plotly_chart(fig)
        st.write(dados_anualizados_lg)

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
            #skew_table_lg = pd.concat([skew_table_lg, pd.DataFrame([row])], ignore_index=True)

        # Format the 'p-value' column to scientific notation
        #skew_table_lg['p-value'] = skew_table_lg['p-value'].apply(lambda x: f"{x:.2e}")

        #st.write(skew_table_lg)

        st.write("## Drawdown Máximo")	

        wealth_index_lg =  (1 + log_returns).cumprod()
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

        st.plotly_chart(VaR_graph)

        st.write(VaR_log)

        semideviation(log_returns)

with Otimização:
    st.write("## Fronteira Eficiênte")
    
    col_data_inicio,col_data_fim = st.columns(2)

    with col_data_inicio:
        data_historico_inicio = st.date_input(label="Período Historico inicio", value=filtered_data.index.min(), format="DD/MM/YYYY")
        data_historico_fim = st.date_input(label="Período Historico fim", value= datetime(2023,12,31), format="DD/MM/YYYY")
        
    
    with col_data_fim: 
       data_backtest_inicio = st.date_input(label="início Backtest", value = data_historico_fim + timedelta(days=1), format="DD/MM/YYYY")
       data_backtest_fim = st.date_input(label="Fim Backtest", value= filtered_data.index.max(),format="DD/MM/YYYY")

    
    
    
    historico = filtered_data[data_historico_inicio:data_historico_fim].dropna()
      
    backtest = filtered_data[data_backtest_inicio:data_backtest_fim].dropna()

    investimento_input = st.number_input(label="Investimento R$", value=1000)

    st.write(seletor_selic)

    experimento = backtest_markowitz(prices=historico, backtest=backtest,investimento=investimento_input, risk_free=seletor_selic/100)
    experimento.optimization_mv()
    experimento.backtest_performance()

    weights = experimento.weights
    performance = experimento.performance
    retornos_cumulativos = experimento.cum_return

    fig,ax = experimento.fronteira_eficiente_plot()
    st.pyplot(fig)

    weights_long = weights.reset_index().melt(id_vars= "Ticker",var_name='Carteira',value_name='Fração Percentual') 
    weights_long['Fração Percentual'] = round(weights_long['Fração Percentual'] * 100,2)

    st.write("## Composição das Carteiras")

    bart_carteira = ggplot(weights_long, aes(x="Carteira", y="Fração Percentual", fill="Ticker"))+ geom_col()

    st.pyplot(ggplot.draw(bart_carteira))

    st.write("## BackTest da Carteira")

    st.write(performance)    

    evol_retornos = px.line(title="Retornos Cumulativos")
    for col in retornos_cumulativos:
        evol_retornos.add_scatter(x=retornos_cumulativos.index, y=retornos_cumulativos[col], mode="lines", name=col)

    st.plotly_chart(evol_retornos)

    variancia = retornos_cumulativos.std()

    st.write(variancia)

    

   










