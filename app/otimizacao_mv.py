import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
from pypfopt.expected_returns import mean_historical_return
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import sample_cov
from pypfopt import plotting
import os

selic = st.session_state['selic']
data = st.session_state['filtered_data']
simple_returns = st.session_state['simple_returns']
log_returns = st.session_state['log_returns']
free_rate = selic['value'].tail(1).values[0]/100

st.write("# Otimização de Portfólio")

seletor_stock = st.multiselect("Ações", data.columns,placeholder="Selecione as ações", default=['ITUB4', 'B3SA3', 'PETR4'])

mu = mean_historical_return(data, frequency=252)  # expected returns
S = sample_cov(data, frequency=252)  # covariance matrix
ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

fig, ax = plt.subplots(figsize=(10, 6))
ef_max_sharpe = ef.deepcopy()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True, )

# Find the tangency portfolio
ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 30000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()

st.pyplot(fig)
