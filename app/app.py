import streamlit as st

pg = st.navigation(
    [
        st.Page("analise_hist.py", title="1 - Análise Histórica", icon=":material/analytics:"),
        st.Page("analise_riscos.py", title="2 - Análise de riscos", icon=":material/analytics:"),
        st.Page("otimizacao_randomica.py", title="3 - Análise de impacto", icon=":material/analytics:"),
        st.Page("otimizacao_portfolio.py", title="4 - Otimização Portifólio", icon=":material/analytics:"),
    ]
)

pg.run()