import streamlit as st

pg = st.navigation(
    [
        st.Page("analise_retornos.py", title="1 - Análise Histórica dos Retornos", icon=":material/analytics:"),
        st.Page("analise_riscos.py", title="2 - Análise Histórica dos riscos", icon=":material/analytics:"),
        st.Page("otimizacao_mv.py", title="3 - Otimização Portifólio", icon=":material/analytics:")
    ]
)

if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = None


pg.run()