"""
Calculadora de Manutenção Preventiva com Age Replacement
Autor: Sistema de Engenharia de Confiabilidade
Versão: 1.0.0 (MVP)
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import io

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Calculadora de Manutenção Preventiva",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== NÚCLEO DE CÁLCULO - KPIs BÁSICOS ====================

def calcular_kpis_basicos(HO: float, HF: float, Nf: int, HD: float, HP: float) -> dict:
    """
    Calcula KPIs básicos de confiabilidade.
    
    Args:
        HO: Horas operadas
        HF: Horas em falha (downtime corretivo)
        Nf: Número de falhas
        HD: Horas disponíveis
        HP: Horas paradas programadas
        
    Returns:
        Dicionário com MTBF, MTTR, Ai, DF, UF
    """
    if Nf == 0:
        raise ValueError("Número de falhas não pode ser zero")
    
    MTBF = HO / Nf
    MTTR = HF / Nf
    Ai = MTBF / (MTBF + MTTR)
    
    # DF: Fator de disponibilidade (tempo disponível para operar)
    DF = (HD - HF) / HD if HD > 0 else 0
    
    # UF: Fator de utilização (quanto do tempo disponível foi usado)
    tempo_disponivel_liquido = HD - HP
    UF = HO / tempo_disponivel_liquido if tempo_disponivel_liquido > 0 else 0
    
    return {
        'MTBF': MTBF,
        'MTTR': MTTR,
        'Ai': Ai,
        'DF': DF,
        'UF': UF
    }

# ==================== MODELO EXPONENCIAL ====================

def exponencial_sobrevida(T: float, MTBF: float) -> float:
    """Função de sobrevivência para distribuição exponencial."""
    return np.exp(-T / MTBF)

def exponencial_falha(T: float, MTBF: float) -> float:
    """Probabilidade de falha antes de T (distribuição exponencial)."""
    return 1 - exponencial_sobrevida(T, MTBF)

def exponencial_uptime_medio(T: float, MTBF: float) -> float:
    """
    Uptime médio por ciclo de manutenção (modelo exponencial).
    E[L] = MTBF * (1 - exp(-T/MTBF))
    """
    return MTBF * (1 - np.exp(-T / MTBF))

def exponencial_disponibilidade(T: float, MTBF: float, MTTR_c: float, d_PM: float) -> float:
    """
    Disponibilidade média para Age Replacement com modelo exponencial.
    
    Args:
        T: Intervalo de PM (horas operadas)
        MTBF: Mean Time Between Failures
        MTTR_c: Tempo médio de reparo corretivo
        d_PM: Duração da manutenção preventiva
    """
    S_T = exponencial_sobrevida(T, MTBF)
    F_T = exponencial_falha(T, MTBF)
    
    E_L = exponencial_uptime_medio(T, MTBF)
    D_T = MTTR_c * F_T + d_PM * S_T
    
    return E_L / (E_L + D_T) if (E_L + D_T) > 0 else 0

def exponencial_custo_hora(T: float, MTBF: float, MTTR_c: float, C_PM: float, C_CM: float) -> float:
    """
    Custo por hora operada (modelo exponencial).
    
    g(T) = (C_PM + C_CM * F(T)) / (T + MTTR_c * F(T))
    """
    F_T = exponencial_falha(T, MTBF)
    
    numerador = C_PM + C_CM * F_T
    denominador = T + MTTR_c * F_T
    
    return numerador / denominador if denominador > 0 else float('inf')

# ==================== MODELO WEIBULL ====================

def weibull_sobrevida(T: float, beta: float, eta: float) -> float:
    """Função de sobrevivência para distribuição Weibull."""
    return np.exp(-(T / eta) ** beta)

def weibull_falha(T: float, beta: float, eta: float) -> float:
    """Probabilidade de falha antes de T (distribuição Weibull)."""
    return 1 - weibull_sobrevida(T, beta, eta)

def weibull_uptime_medio(T: float, beta: float, eta: float, n_pontos: int = 1000) -> float:
    """
    Uptime médio por ciclo (modelo Weibull).
    E[L] = ∫_0^T S(t) dt (integração numérica)
    """
    t_vals = np.linspace(0, T, n_pontos)
    S_vals = weibull_sobrevida(t_vals, beta, eta)
    
    # Integração pelo método dos trapézios
    return np.trapz(S_vals, t_vals)

def weibull_disponibilidade(T: float, beta: float, eta: float, MTTR_c: float, d_PM: float) -> float:
    """Disponibilidade média para Age Replacement com modelo Weibull."""
    S_T = weibull_sobrevida(T, beta, eta)
    F_T = weibull_falha(T, beta, eta)
    
    E_L = weibull_uptime_medio(T, beta, eta)
    D_T = MTTR_c * F_T + d_PM * S_T
    
    return E_L / (E_L + D_T) if (E_L + D_T) > 0 else 0

def weibull_custo_hora(T: float, beta: float, eta: float, MTTR_c: float, C_PM: float, C_CM: float) -> float:
    """Custo por hora operada (modelo Weibull)."""
    F_T = weibull_falha(T, beta, eta)
    E_L = weibull_uptime_medio(T, beta, eta)
    
    numerador = C_PM + C_CM * F_T
    denominador = E_L + MTTR_c * F_T
    
    return numerador / denominador if denominador > 0 else float('inf')

# ==================== OTIMIZAÇÃO ====================

def buscar_T_meta_disponibilidade(
    A_meta: float,
    MTBF: float,
    MTTR_c: float,
    d_PM: float,
    modelo: str = "Exponencial",
    beta: float = 1.0,
    eta: float = 1000.0,
    tol: float = 0.0001,
    max_iter: int = 100
) -> Optional[float]:
    """
    Busca binária para encontrar T que atinge A_meta.
    
    Args:
        A_meta: Disponibilidade alvo
        MTBF: Mean Time Between Failures
        MTTR_c: Tempo médio de reparo corretivo
        d_PM: Duração da PM
        modelo: "Exponencial" ou "Weibull"
        beta, eta: Parâmetros Weibull (se aplicável)
        tol: Tolerância para convergência
        max_iter: Máximo de iterações
        
    Returns:
        T ótimo (horas operadas) ou None se não convergir
    """
    # Limites de busca
    T_min = d_PM  # Mínimo razoável
    T_max = MTBF * 10  # Máximo razoável
    
    for _ in range(max_iter):
        T_mid = (T_min + T_max) / 2
        
        if modelo == "Exponencial":
            A_atual = exponencial_disponibilidade(T_mid, MTBF, MTTR_c, d_PM)
        else:  # Weibull
            A_atual = weibull_disponibilidade(T_mid, beta, eta, MTTR_c, d_PM)
        
        if abs(A_atual - A_meta) < tol:
            return T_mid
        
        # Disponibilidade aumenta com T (geralmente)
        if A_atual < A_meta:
            T_min = T_mid
        else:
            T_max = T_mid
    
    return None  # Não convergiu

def encontrar_T_custo_minimo(
    MTBF: float,
    MTTR_c: float,
    C_PM: float,
    C_CM: float,
    modelo: str = "Exponencial",
    beta: float = 1.0,
    eta: float = 1000.0,
    n_pontos: int = 500
) -> Tuple[float, float]:
    """
    Encontra T que minimiza custo por hora operada (varredura).
    
    Returns:
        (T_otimo, custo_minimo)
    """
    T_vals = np.linspace(MTBF * 0.1, MTBF * 5, n_pontos)
    custos = []
    
    for T in T_vals:
        if modelo == "Exponencial":
            custo = exponencial_custo_hora(T, MTBF, MTTR_c, C_PM, C_CM)
        else:  # Weibull
            custo = weibull_custo_hora(T, beta, eta, MTTR_c, C_PM, C_CM)
        custos.append(custo)
    
    idx_min = np.argmin(custos)
    return T_vals[idx_min], custos[idx_min]

# ==================== CONVERSÃO PARA CALENDÁRIO ====================

def converter_para_calendario(T_operado: float, DF: float, UF: float) -> float:
    """
    Converte intervalo de PM de horas operadas para horas calendário.
    
    T_cal = T / (DF * UF)
    """
    fator = DF * UF
    if fator <= 0:
        raise ValueError("DF * UF deve ser maior que zero")
    
    return T_operado / fator

# ==================== GERAÇÃO DE DADOS PARA GRÁFICOS ====================

def gerar_curvas(
    MTBF: float,
    MTTR_c: float,
    d_PM: float,
    C_PM: float,
    C_CM: float,
    modelo: str,
    beta: float,
    eta: float,
    n_pontos: int = 200
) -> pd.DataFrame:
    """Gera dados para plotar A(T) e g(T)."""
    T_vals = np.linspace(MTBF * 0.1, MTBF * 5, n_pontos)
    
    A_vals = []
    g_vals = []
    
    for T in T_vals:
        if modelo == "Exponencial":
            A = exponencial_disponibilidade(T, MTBF, MTTR_c, d_PM)
            g = exponencial_custo_hora(T, MTBF, MTTR_c, C_PM, C_CM)
        else:  # Weibull
            A = weibull_disponibilidade(T, beta, eta, MTTR_c, d_PM)
            g = weibull_custo_hora(T, beta, eta, MTTR_c, C_PM, C_CM)
        
        A_vals.append(A)
        g_vals.append(g)
    
    return pd.DataFrame({
        'T (horas operadas)': T_vals,
        'Disponibilidade A(T)': A_vals,
        'Custo/hora g(T)': g_vals
    })

# ==================== INTERFACE STREAMLIT ====================

def main():
    st.title("🔧 Calculadora de Manutenção Preventiva")
    st.markdown("""
    **Sistema de otimização de intervalos de manutenção preventiva baseado em Age Replacement.**
    
    Esta ferramenta calcula o intervalo ótimo de PM considerando:
    - **Meta de disponibilidade**: Encontra o intervalo que atinge a disponibilidade desejada
    - **Custo mínimo**: Encontra o intervalo que minimiza o custo total por hora operada
    """)
    
    # ==================== SIDEBAR - INPUTS ====================
    
    st.sidebar.header("📊 Dados do Ativo")
    
    # Dados históricos para KPIs
    st.sidebar.subheader("Histórico Operacional")
    HO = st.sidebar.number_input("Horas Operadas (HO)", min_value=1.0, value=8760.0, step=100.0,
                                  help="Total de horas que o equipamento operou")
    HF = st.sidebar.number_input("Horas em Falha (HF)", min_value=0.0, value=87.6, step=10.0,
                                  help="Total de horas em manutenção corretiva")
    Nf = st.sidebar.number_input("Número de Falhas (Nf)", min_value=1, value=10, step=1,
                                  help="Quantidade de falhas ocorridas")
    
    st.sidebar.subheader("Disponibilidade de Tempo")
    HD = st.sidebar.number_input("Horas Disponíveis (HD)", min_value=1.0, value=8760.0, step=100.0,
                                  help="Total de horas no período (ex: 365 dias = 8760h)")
    HP = st.sidebar.number_input("Horas Paradas Programadas (HP)", min_value=0.0, value=0.0, step=10.0,
                                  help="Paradas programadas (não PM, ex: feriados)")
    
    # Parâmetros de manutenção
    st.sidebar.subheader("Parâmetros de Manutenção")
    MTTR_c = st.sidebar.number_input("MTTR Corretivo (horas)", min_value=0.1, value=8.76, step=0.5,
                                      help="Tempo médio de reparo corretivo")
    d_PM = st.sidebar.number_input("Duração da PM (horas)", min_value=0.1, value=4.0, step=0.5,
                                    help="Tempo necessário para executar PM")
    
    # Custos
    st.sidebar.subheader("Custos")
    C_PM = st.sidebar.number_input("Custo da PM (R$)", min_value=0.0, value=5000.0, step=100.0,
                                    help="Custo de uma manutenção preventiva")
    C_CM = st.sidebar.number_input("Custo da Corretiva (R$)", min_value=0.0, value=20000.0, step=100.0,
                                    help="Custo médio de uma falha + reparo")
    
    # Modelo de falha
    st.sidebar.subheader("Modelo de Falha")
    modelo = st.sidebar.selectbox("Distribuição", ["Exponencial", "Weibull"],
                                   help="Exponencial: taxa de falha constante (β=1). Weibull: permite desgaste/envelhecimento")
    
    beta = 1.0
    eta = 1000.0
    
    if modelo == "Weibull":
        beta = st.sidebar.number_input("Parâmetro β (forma)", min_value=0.1, value=2.0, step=0.1,
                                        help="β<1: taxa decrescente, β=1: exponencial, β>1: desgaste")
        eta = st.sidebar.number_input("Parâmetro η (escala)", min_value=1.0, value=1000.0, step=10.0,
                                       help="Vida característica (aproximadamente MTBF para β próximo de 1)")
    
    # Modo de otimização
    st.sidebar.subheader("Modo de Otimização")
    modo = st.sidebar.radio("Objetivo", ["Meta de Disponibilidade", "Custo Mínimo"])
    
    A_meta = 0.95
    if modo == "Meta de Disponibilidade":
        A_meta = st.sidebar.slider("Disponibilidade Alvo (%)", min_value=80.0, max_value=99.9, value=95.0, step=0.1) / 100
    
    # ==================== CÁLCULOS ====================
    
    try:
        # KPIs básicos
        kpis = calcular_kpis_basicos(HO, HF, Nf, HD, HP)
        MTBF = kpis['MTBF']
        DF = kpis['DF']
        UF = kpis['UF']
        
        # Validações
        if DF <= 0 or DF > 1:
            st.error("⚠️ DF (Fator de Disponibilidade) deve estar entre 0 e 1. Verifique os dados de entrada.")
            return
        
        if UF <= 0 or UF > 1:
            st.error("⚠️ UF (Fator de Utilização) deve estar entre 0 e 1. Verifique os dados de entrada.")
            return
        
        # Ajustar eta para Weibull se necessário (aproximação inicial)
        if modelo == "Weibull" and eta == 1000.0:
            eta = MTBF  # Usar MTBF como estimativa inicial
        
        # ==================== CARDS DE KPIs ====================
        
        st.header("📈 Indicadores de Confiabilidade")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("MTBF", f"{MTBF:.1f} h", help="Mean Time Between Failures")
        
        with col2:
            st.metric("MTTR", f"{kpis['MTTR']:.1f} h", help="Mean Time To Repair")
        
        with col3:
            st.metric("Disponibilidade Intrínseca", f"{kpis['Ai']*100:.2f}%", help="Ai = MTBF/(MTBF+MTTR)")
        
        with col4:
            st.metric("DF", f"{DF*100:.2f}%", help="Fator de Disponibilidade")
        
        with col5:
            st.metric("UF", f"{UF*100:.2f}%", help="Fator de Utilização")
        
        # ==================== OTIMIZAÇÃO ====================
        
        st.header("🎯 Intervalo Ótimo de Manutenção Preventiva")
        
        T_otimo = None
        A_otimo = None
        g_otimo = None
        
        if modo == "Meta de Disponibilidade":
            T_otimo = buscar_T_meta_disponibilidade(
                A_meta, MTBF, MTTR_c, d_PM, modelo, beta, eta
            )
            
            if T_otimo is None:
                st.warning("⚠️ Não foi possível encontrar um intervalo que atinja a meta de disponibilidade. Tente ajustar os parâmetros.")
            else:
                # Calcular disponibilidade e custo para o T ótimo
                if modelo == "Exponencial":
                    A_otimo = exponencial_disponibilidade(T_otimo, MTBF, MTTR_c, d_PM)
                    g_otimo = exponencial_custo_hora(T_otimo, MTBF, MTTR_c, C_PM, C_CM)
                else:
                    A_otimo = weibull_disponibilidade(T_otimo, beta, eta, MTTR_c, d_PM)
                    g_otimo = weibull_custo_hora(T_otimo, beta, eta, MTTR_c, C_PM, C_CM)
        
        else:  # Custo Mínimo
            T_otimo, g_otimo = encontrar_T_custo_minimo(
                MTBF, MTTR_c, C_PM, C_CM, modelo, beta, eta
            )
            
            # Calcular disponibilidade para o T ótimo
            if modelo == "Exponencial":
                A_otimo = exponencial_disponibilidade(T_otimo, MTBF, MTTR_c, d_PM)
            else:
                A_otimo = weibull_disponibilidade(T_otimo, beta, eta, MTTR_c, d_PM)
        
        # ==================== RESULTADOS ====================
        
        if T_otimo is not None:
            T_cal = converter_para_calendario(T_otimo, DF, UF)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Intervalo Ótimo (horas operadas)",
                    f"{T_otimo:.1f} h",
                    help="Tempo de operação entre PMs"
                )
            
            with col2:
                st.metric(
                    "Intervalo Calendário",
                    f"{T_cal:.1f} h ({T_cal/24:.1f} dias)",
                    help="Convertido considerando DF e UF"
                )
            
            with col3:
                st.metric(
                    "Disponibilidade Resultante",
                    f"{A_otimo*100:.2f}%",
                    delta=f"{(A_otimo - kpis['Ai'])*100:+.2f}% vs Ai"
                )
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Custo por Hora Operada",
                    f"R$ {g_otimo:.2f}/h",
                    help="Custo total (PM + falhas) por hora operada"
                )
            
            with col2:
                custo_anual = g_otimo * HO
                st.metric(
                    "Custo Anual Estimado",
                    f"R$ {custo_anual:,.2f}",
                    help=f"Baseado em {HO:.0f} horas operadas/ano"
                )
            
            # ==================== DETALHAMENTO ====================
            
            with st.expander("📋 Detalhamento dos Cálculos"):
                st.markdown(f"""
                **Modelo utilizado:** {modelo}
                
                **Parâmetros do modelo:**
                - MTBF: {MTBF:.2f} horas
                - MTTR corretivo: {MTTR_c:.2f} horas
                - Duração da PM: {d_PM:.2f} horas
                """)
                
                if modelo == "Weibull":
                    st.markdown(f"""
                    - β (forma): {beta:.2f}
                    - η (escala): {eta:.2f}
                    """)
                
                st.markdown(f"""
                **Fatores operacionais:**
                - DF (Fator de Disponibilidade): {DF:.4f}
                - UF (Fator de Utilização): {UF:.4f}
                - DF × UF: {DF*UF:.4f}
                
                **Custos:**
                - Custo PM: R$ {C_PM:,.2f}
                - Custo Corretiva: R$ {C_CM:,.2f}
                - Razão C_CM/C_PM: {C_CM/C_PM:.2f}
                
                **Resultados:**
                - Probabilidade de falha antes de T: {(1 - (exponencial_sobrevida(T_otimo, MTBF) if modelo == 'Exponencial' else weibull_sobrevida(T_otimo, beta, eta)))*100:.2f}%
                - Número estimado de PMs/ano: {HO/T_otimo:.1f}
                """)
        
        # ==================== GRÁFICOS ====================
        
        st.header("📊 Análise Gráfica")
        
        df_curvas = gerar_curvas(MTBF, MTTR_c, d_PM, C_PM, C_CM, modelo, beta, eta)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Disponibilidade vs Intervalo de PM")
            st.line_chart(df_curvas.set_index('T (horas operadas)')['Disponibilidade A(T)'])
            
            if T_otimo is not None and A_otimo is not None:
                st.caption(f"✓ Ponto ótimo: T = {T_otimo:.1f}h, A = {A_otimo*100:.2f}%")
        
        with col2:
            st.subheader("Custo/Hora vs Intervalo de PM")
            st.line_chart(df_curvas.set_index('T (horas operadas)')['Custo/hora g(T)'])
            
            if T_otimo is not None and g_otimo is not None:
                st.caption(f"✓ Ponto ótimo: T = {T_otimo:.1f}h, g = R$ {g_otimo:.2f}/h")
        
        # ==================== TABELA DE RESULTADOS ====================
        
        st.header("📋 Tabela de Resultados")
        
        # Criar DataFrame com resultados principais
        resultados = {
            'Parâmetro': [
                'MTBF', 'MTTR', 'Disponibilidade Intrínseca (Ai)',
                'DF', 'UF', 'Intervalo PM Ótimo (horas operadas)',
                'Intervalo PM Calendário (horas)', 'Intervalo PM Calendário (dias)',
                'Disponibilidade Resultante', 'Custo por Hora Operada',
                'Custo Anual Estimado'
            ],
            'Valor': [
                f"{MTBF:.2f} h",
                f"{kpis['MTTR']:.2f} h",
                f"{kpis['Ai']*100:.2f}%",
                f"{DF*100:.2f}%",
                f"{UF*100:.2f}%",
                f"{T_otimo:.2f} h" if T_otimo else "N/A",
                f"{T_cal:.2f} h" if T_otimo else "N/A",
                f"{T_cal/24:.2f} dias" if T_otimo else "N/A",
                f"{A_otimo*100:.2f}%" if A_otimo else "N/A",
                f"R$ {g_otimo:.2f}/h" if g_otimo else "N/A",
                f"R$ {g_otimo * HO:,.2f}" if g_otimo else "N/A"
            ]
        }
        
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados, use_container_width=True, hide_index=True)
        
        # ==================== EXPORT CSV ====================
        
        st.header("💾 Exportar Resultados")
        
        # Combinar resultados e curvas
        df_export = df_resultados.copy()
        
        # Converter para CSV
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_data = csv_buffer.getvalue()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Resultados (CSV)",
                data=csv_data,
                file_name="resultados_manutencao.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export das curvas
            csv_curvas = df_curvas.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Download Curvas (CSV)",
                data=csv_curvas,
                file_name="curvas_analise.csv",
                mime="text/csv"
            )
    
    except ValueError as e:
        st.error(f"❌ Erro nos dados de entrada: {str(e)}")
    except Exception as e:
        st.error(f"❌ Erro inesperado: {str(e)}")
        st.exception(e)
    
    # ==================== RODAPÉ ====================
    
    st.divider()
    st.markdown("""
    **Sobre esta ferramenta:**
    
    Sistema de otimização de manutenção preventiva baseado em Age Replacement Policy. 
    Calcula o intervalo ótimo de PM considerando modelos de confiabilidade (Exponencial e Weibull)
    e objetivos de disponibilidade ou custo mínimo.
    
    **Referências:**
    - Barlow, R. E., & Proschan, F. (1965). Mathematical Theory of Reliability
    - Nakagawa, T. (2005). Maintenance Theory of Reliability
    """)

# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    main()
