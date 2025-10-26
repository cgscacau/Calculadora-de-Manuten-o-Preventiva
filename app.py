"""
Calculadora de Manutenção Preventiva - Foco em Disponibilidade Operacional
Versão: 3.0.0 (Sem Custos - Foco em Produção)
Autor: Sistema de Engenharia de Confiabilidade
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Calculadora de Disponibilidade Operacional",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSTANTES ====================
HORAS_POR_MES = 730.0
DIAS_POR_MES = 30.44

# ==================== CÁLCULOS DE DISPONIBILIDADE ====================

def calcular_disponibilidade_intrinseca(MTBF: float, MTTR: float) -> float:
    """
    Calcula disponibilidade intrínseca (inerente ao equipamento).
    
    Ai = MTBF / (MTBF + MTTR)
    """
    return MTBF / (MTBF + MTTR) if (MTBF + MTTR) > 0 else 0

def calcular_disponibilidade_alcancada(Ai: float, DF: float) -> float:
    """
    Calcula disponibilidade alcançada (considerando paradas programadas).
    
    Aa = Ai × DF
    """
    return Ai * DF

def calcular_disponibilidade_operacional(Aa: float, UF: float) -> float:
    """
    Calcula disponibilidade operacional (real, considerando utilização).
    
    Ao = Aa × UF = Ai × DF × UF
    """
    return Aa * UF

def calcular_horas_disponiveis_producao(HD: float, HP: float, HF: float) -> float:
    """
    Calcula horas realmente disponíveis para produção.
    
    HD_prod = HD - HP - HF
    """
    return HD - HP - HF

def calcular_DF_necessario(Ai: float, UF: float, Ao_meta: float) -> float:
    """
    Calcula DF necessário para atingir Ao_meta.
    
    DF = Ao_meta / (Ai × UF)
    """
    denominador = Ai * UF
    if denominador <= 0:
        return 0
    return Ao_meta / denominador

def calcular_UF_necessario(Ai: float, DF: float, Ao_meta: float) -> float:
    """
    Calcula UF necessário para atingir Ao_meta.
    
    UF = Ao_meta / (Ai × DF)
    """
    denominador = Ai * DF
    if denominador <= 0:
        return 0
    return Ao_meta / denominador

def calcular_MTBF_necessario(MTTR: float, DF: float, UF: float, Ao_meta: float) -> float:
    """
    Calcula MTBF necessário para atingir Ao_meta.
    
    Ao = (MTBF / (MTBF + MTTR)) × DF × UF
    
    Resolvendo para MTBF:
    MTBF = (Ao_meta × MTTR) / (DF × UF - Ao_meta)
    """
    denominador = DF * UF - Ao_meta
    if denominador <= 0:
        return float('inf')
    return (Ao_meta * MTTR) / denominador

def calcular_MTTR_maximo(MTBF: float, DF: float, UF: float, Ao_meta: float) -> float:
    """
    Calcula MTTR máximo permitido para atingir Ao_meta.
    
    MTTR = MTBF × ((DF × UF - Ao_meta) / Ao_meta)
    """
    if Ao_meta <= 0:
        return 0
    return MTBF * ((DF * UF - Ao_meta) / Ao_meta)

# ==================== MODELO DE DEGRADAÇÃO ====================

def taxa_falha_degradacao(t: float, lambda_base: float, beta_desgaste: float, t_inicio_desgaste: float) -> float:
    """Taxa de falha com degradação progressiva."""
    if t <= t_inicio_desgaste:
        return lambda_base
    else:
        t_desgaste = t - t_inicio_desgaste
        return lambda_base * (1 + (t_desgaste / t_inicio_desgaste) ** beta_desgaste)

def confiabilidade_degradacao(t: float, lambda_base: float, beta_desgaste: float, t_inicio_desgaste: float, n_pontos: int = 1000) -> float:
    """Confiabilidade considerando degradação."""
    if t <= 0:
        return 1.0
    
    t_vals = np.linspace(0, t, n_pontos)
    lambda_vals = np.array([taxa_falha_degradacao(ti, lambda_base, beta_desgaste, t_inicio_desgaste) for ti in t_vals])
    integral_lambda = np.trapz(lambda_vals, t_vals)
    
    return np.exp(-integral_lambda)

def disponibilidade_ao_longo_tempo(
    t: float, 
    lambda_base: float, 
    beta_desgaste: float, 
    t_inicio_desgaste: float,
    MTTR: float,
    DF: float,
    UF: float
) -> float:
    """Disponibilidade operacional instantânea no tempo t."""
    R_t = confiabilidade_degradacao(t, lambda_base, beta_desgaste, t_inicio_desgaste)
    
    # Disponibilidade intrínseca no tempo t
    tempo_total = t + MTTR * (1 - R_t)
    Ai_t = (t * R_t) / tempo_total if tempo_total > 0 else 0
    
    # Disponibilidade operacional
    Ao_t = Ai_t * DF * UF
    
    return Ao_t

def encontrar_intervalo_PM_por_disponibilidade(
    lambda_base: float,
    beta_desgaste: float,
    t_inicio_desgaste: float,
    MTTR: float,
    DF: float,
    UF: float,
    Ao_minima: float,
    t_max: float = None
) -> dict:
    """
    Encontra intervalo ótimo de PM baseado em disponibilidade operacional mínima.
    """
    if t_max is None:
        t_max = t_inicio_desgaste * 3
    
    t_vals = np.linspace(1, t_max, 500)
    
    disponibilidades = []
    confiabilidades = []
    taxas_falha = []
    
    for t in t_vals:
        Ao_t = disponibilidade_ao_longo_tempo(t, lambda_base, beta_desgaste, t_inicio_desgaste, MTTR, DF, UF)
        disponibilidades.append(Ao_t)
        
        R_t = confiabilidade_degradacao(t, lambda_base, beta_desgaste, t_inicio_desgaste)
        confiabilidades.append(R_t)
        
        lambda_t = taxa_falha_degradacao(t, lambda_base, beta_desgaste, t_inicio_desgaste)
        taxas_falha.append(lambda_t)
    
    disponibilidades = np.array(disponibilidades)
    confiabilidades = np.array(confiabilidades)
    taxas_falha = np.array(taxas_falha)
    
    # Encontrar último ponto onde disponibilidade >= Ao_minima
    idx_acima_minima = np.where(disponibilidades >= Ao_minima)[0]
    
    if len(idx_acima_minima) > 0:
        T_otimo = t_vals[idx_acima_minima[-1]]
        idx_otimo = idx_acima_minima[-1]
    else:
        # Se nunca atinge, escolher ponto de maior disponibilidade
        idx_otimo = np.argmax(disponibilidades)
        T_otimo = t_vals[idx_otimo]
    
    return {
        'T_otimo': T_otimo,
        'disponibilidade': disponibilidades[idx_otimo],
        'confiabilidade': confiabilidades[idx_otimo],
        'taxa_falha': taxas_falha[idx_otimo],
        't_vals': t_vals,
        'disponibilidades': disponibilidades,
        'confiabilidades': confiabilidades,
        'taxas_falha': taxas_falha
    }

# ==================== MATRIZ DE DISPONIBILIDADE ====================

def gerar_matriz_disponibilidade(
    parametro_fixo: str,
    valor_fixo: float,
    range_param1: Tuple[float, float],
    range_param2: Tuple[float, float],
    DF: float,
    UF: float,
    n_pontos: int = 20
) -> pd.DataFrame:
    """
    Gera matriz relacionando MTBF, MTTR e DF/Ai.
    
    Args:
        parametro_fixo: 'MTBF', 'MTTR', 'DF' ou 'Ai'
        valor_fixo: Valor do parâmetro fixo
        range_param1: (min, max) do primeiro parâmetro variável
        range_param2: (min, max) do segundo parâmetro variável
        DF, UF: Fatores para cálculo de Ao
    """
    param1_vals = np.linspace(range_param1[0], range_param1[1], n_pontos)
    param2_vals = np.linspace(range_param2[0], range_param2[1], n_pontos)
    
    matriz = np.zeros((n_pontos, n_pontos))
    
    for i, p1 in enumerate(param1_vals):
        for j, p2 in enumerate(param2_vals):
            if parametro_fixo == 'MTBF':
                MTBF = valor_fixo
                MTTR = p1
                DF_calc = p2
                UF_calc = UF
            elif parametro_fixo == 'MTTR':
                MTBF = p1
                MTTR = valor_fixo
                DF_calc = p2
                UF_calc = UF
            elif parametro_fixo == 'DF':
                MTBF = p1
                MTTR = p2
                DF_calc = valor_fixo
                UF_calc = UF
            else:  # Ai fixo
                MTBF = p1
                MTTR = p2
                Ai = valor_fixo
                DF_calc = DF
                UF_calc = UF
                # Recalcular baseado em Ai fixo
                if Ai > 0:
                    MTBF = (Ai * MTTR) / (1 - Ai) if Ai < 1 else MTBF
            
            Ai = calcular_disponibilidade_intrinseca(MTBF, MTTR)
            Ao = calcular_disponibilidade_operacional(Ai, DF_calc, UF_calc)
            matriz[i, j] = Ao * 100
    
    return matriz, param1_vals, param2_vals

# ==================== PLOTAGEM ====================

def criar_grafico_degradacao_disponibilidade(resultado: dict, T_otimo: float, Ao_minima: float) -> go.Figure:
    """Gráfico de degradação focado em disponibilidade."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Disponibilidade Operacional ao Longo do Tempo',
            'Confiabilidade (Probabilidade de Não Falhar)',
            'Taxa de Falha Instantânea',
            'Disponibilidade vs Confiabilidade'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    t_vals = resultado['t_vals']
    
    # Subplot 1: Disponibilidade Operacional
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=resultado['disponibilidades'] * 100,
            mode='lines',
            name='Ao(t)',
            line=dict(color='blue', width=3),
            hovertemplate='Tempo: %{x:.1f}h<br>Ao: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Linha de disponibilidade mínima
    fig.add_hline(
        y=Ao_minima * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Ao mínima: {Ao_minima*100:.1f}%",
        row=1, col=1
    )
    
    # Subplot 2: Confiabilidade
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=resultado['confiabilidades'] * 100,
            mode='lines',
            name='R(t)',
            line=dict(color='green', width=3),
            hovertemplate='Tempo: %{x:.1f}h<br>R(t): %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Subplot 3: Taxa de Falha
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=resultado['taxas_falha'],
            mode='lines',
            name='λ(t)',
            line=dict(color='red', width=3),
            hovertemplate='Tempo: %{x:.1f}h<br>λ(t): %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Subplot 4: Ao vs R (scatter)
    fig.add_trace(
        go.Scatter(
            x=resultado['confiabilidades'] * 100,
            y=resultado['disponibilidades'] * 100,
            mode='markers',
            name='Ao vs R',
            marker=dict(
                size=4,
                color=t_vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Tempo (h)", x=1.15)
            ),
            hovertemplate='R(t): %{x:.2f}%<br>Ao(t): %{y:.2f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Linha vertical no ponto ótimo
    for row in [1, 2]:
        for col in [1, 2]:
            if col == 1:  # Apenas nos gráficos com tempo no eixo x
                fig.add_vline(
                    x=T_otimo,
                    line_dash="dash",
                    line_color="purple",
                    opacity=0.7,
                    annotation_text=f"PM: {T_otimo:.0f}h",
                    row=row, col=col
                )
    
    # Atualizar eixos
    fig.update_xaxes(title_text="Horas Operadas", row=1, col=1)
    fig.update_xaxes(title_text="Horas Operadas", row=1, col=2)
    fig.update_xaxes(title_text="Horas Operadas", row=2, col=1)
    fig.update_xaxes(title_text="Confiabilidade (%)", row=2, col=2)
    
    fig.update_yaxes(title_text="Ao (%)", row=1, col=1)
    fig.update_yaxes(title_text="R(t) (%)", row=1, col=2)
    fig.update_yaxes(title_text="λ(t)", row=2, col=1)
    fig.update_yaxes(title_text="Ao (%)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Análise de Degradação - Disponibilidade Operacional",
        title_x=0.5
    )
    
    return fig

def criar_heatmap_disponibilidade(
    matriz: np.ndarray,
    param1_vals: np.ndarray,
    param2_vals: np.ndarray,
    param1_name: str,
    param2_name: str,
    titulo: str
) -> go.Figure:
    """Cria heatmap da matriz de disponibilidade."""
    fig = go.Figure(data=go.Heatmap(
        z=matriz,
        x=param2_vals,
        y=param1_vals,
        colorscale='RdYlGn',
        text=np.round(matriz, 1),
        texttemplate='%{text}%',
        textfont={"size": 8},
        colorbar=dict(title="Ao (%)"),
        hovertemplate=f'{param2_name}: %{{x:.1f}}<br>{param1_name}: %{{y:.1f}}<br>Ao: %{{z:.1f}}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title=param2_name,
        yaxis_title=param1_name,
        height=600
    )
    
    return fig

# ==================== INTERFACE STREAMLIT ====================

def main():
    st.title("🎯 Calculadora de Disponibilidade Operacional")
    st.markdown("""
    **Sistema focado em disponibilidade para cumprimento de metas de produção - BASE MENSAL.**
    
    Esta ferramenta permite:
    - ✅ **Calcular disponibilidade operacional** (Ao) baseada em MTBF, MTTR, DF e UF
    - ✅ **Determinar DF e UF necessários** para atingir meta de produção
    - ✅ **Analisar degradação** e encontrar intervalo ótimo de PM
    - ✅ **Matriz de disponibilidade** relacionando MTBF, MTTR e DF
    """)
    
    # ==================== TABS ====================
    
    tab1, tab2, tab3 = st.tabs([
        "🎯 Cálculo de Disponibilidade",
        "📊 Análise de Degradação",
        "🗺️ Matriz de Disponibilidade"
    ])
    
    # ==================== TAB 1: CÁLCULO DE DISPONIBILIDADE ====================
    
    with tab1:
        st.header("🎯 Cálculo de Disponibilidade Operacional")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Dados Operacionais (Base Mensal)")
            
            modo_calculo = st.radio(
                "Modo de Cálculo:",
                ["Calcular Disponibilidade (Ao)", "Calcular DF Necessário", "Calcular UF Necessário", "Calcular MTBF Necessário", "Calcular MTTR Máximo"],
                help="Escolha o que deseja calcular"
            )
            
            st.divider()
            
            # Inputs baseados no modo
            if modo_calculo == "Calcular Disponibilidade (Ao)":
                MTBF = st.number_input("MTBF (horas)", min_value=1.0, value=300.0, step=10.0, key="ao_mtbf")
                MTTR = st.number_input("MTTR (horas)", min_value=0.1, value=5.0, step=0.5, key="ao_mttr")
                DF = st.slider("DF - Fator de Disponibilidade (%)", min_value=50.0, max_value=100.0, value=95.0, step=0.5) / 100
                UF = st.slider("UF - Fator de Utilização (%)", min_value=50.0, max_value=100.0, value=85.0, step=0.5) / 100
                
            elif modo_calculo == "Calcular DF Necessário":
                MTBF = st.number_input("MTBF (horas)", min_value=1.0, value=300.0, step=10.0, key="df_mtbf")
                MTTR = st.number_input("MTTR (horas)", min_value=0.1, value=5.0, step=0.5, key="df_mttr")
                UF = st.slider("UF - Fator de Utilização (%)", min_value=50.0, max_value=100.0, value=85.0, step=0.5) / 100
                Ao_meta = st.slider("Ao Meta - Disponibilidade Operacional Desejada (%)", min_value=50.0, max_value=99.0, value=85.0, step=0.5) / 100
                
            elif modo_calculo == "Calcular UF Necessário":
                MTBF = st.number_input("MTBF (horas)", min_value=1.0, value=300.0, step=10.0, key="uf_mtbf")
                MTTR = st.number_input("MTTR (horas)", min_value=0.1, value=5.0, step=0.5, key="uf_mttr")
                DF = st.slider("DF - Fator de Disponibilidade (%)", min_value=50.0, max_value=100.0, value=95.0, step=0.5) / 100
                Ao_meta = st.slider("Ao Meta - Disponibilidade Operacional Desejada (%)", min_value=50.0, max_value=99.0, value=85.0, step=0.5) / 100
                
            elif modo_calculo == "Calcular MTBF Necessário":
                MTTR = st.number_input("MTTR (horas)", min_value=0.1, value=5.0, step=0.5, key="mtbf_mttr")
                DF = st.slider("DF - Fator de Disponibilidade (%)", min_value=50.0, max_value=100.0, value=95.0, step=0.5) / 100
                UF = st.slider("UF - Fator de Utilização (%)", min_value=50.0, max_value=100.0, value=85.0, step=0.5) / 100
                Ao_meta = st.slider("Ao Meta - Disponibilidade Operacional Desejada (%)", min_value=50.0, max_value=99.0, value=85.0, step=0.5) / 100
                
            else:  # Calcular MTTR Máximo
                MTBF = st.number_input("MTBF (horas)", min_value=1.0, value=300.0, step=10.0, key="mttr_mtbf")
                DF = st.slider("DF - Fator de Disponibilidade (%)", min_value=50.0, max_value=100.0, value=95.0, step=0.5) / 100
                UF = st.slider("UF - Fator de Utilização (%)", min_value=50.0, max_value=100.0, value=85.0, step=0.5) / 100
                Ao_meta = st.slider("Ao Meta - Disponibilidade Operacional Desejada (%)", min_value=50.0, max_value=99.0, value=85.0, step=0.5) / 100
        
        with col2:
            st.subheader("📈 Resultados")
            
            try:
                if modo_calculo == "Calcular Disponibilidade (Ao)":
                    Ai = calcular_disponibilidade_intrinseca(MTBF, MTTR)
                    Aa = calcular_disponibilidade_alcancada(Ai, DF)
                    Ao = calcular_disponibilidade_operacional(Aa, UF)
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            "Ai - Intrínseca",
                            f"{Ai*100:.2f}%",
                            help="Disponibilidade inerente ao equipamento"
                        )
                    
                    with col_b:
                        st.metric(
                            "Aa - Alcançada",
                            f"{Aa*100:.2f}%",
                            help="Considerando paradas programadas"
                        )
                    
                    with col_c:
                        st.metric(
                            "Ao - Operacional",
                            f"{Ao*100:.2f}%",
                            help="Disponibilidade real para produção"
                        )
                    
                    st.divider()
                    
                    # Cálculo de horas produtivas
                    horas_operacao_teorica = HORAS_POR_MES * Ao
                    
                    st.info(f"""
                    **Interpretação:**
                    
                    Com os parâmetros atuais:
                    - **{horas_operacao_teorica:.0f} horas/mês** disponíveis para produção
                    - **{(HORAS_POR_MES - horas_operacao_teorica):.0f} horas/mês** de indisponibilidade total
                    - **Meta de produção**: Se precisar de X horas/mês, verifique se Ao é suficiente
                    """)
                    
                    # Gráfico de composição
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Disponibilidade'],
                            y=[Ai*100],
                            name='Ai',
                            marker_color='lightgreen',
                            text=f'{Ai*100:.1f}%',
                            textposition='inside'
                        ),
                        go.Bar(
                            x=['Disponibilidade'],
                            y=[(Aa-Ai)*100],
                            name='Perda por DF',
                            marker_color='yellow',
                            text=f'{(Aa-Ai)*100:.1f}%',
                            textposition='inside'
                        ),
                        go.Bar(
                            x=['Disponibilidade'],
                            y=[(Ao-Aa)*100],
                            name='Perda por UF',
                            marker_color='orange',
                            text=f'{(Ao-Aa)*100:.1f}%',
                            textposition='inside'
                        ),
                        go.Bar(
                            x=['Disponibilidade'],
                            y=[(100-Ao*100)],
                            name='Indisponibilidade',
                            marker_color='red',
                            text=f'{(100-Ao*100):.1f}%',
                            textposition='inside'
                        )
                    ])
                    
                    fig.update_layout(
                        barmode='stack',
                        title='Composição da Disponibilidade',
                        yaxis_title='Percentual (%)',
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif modo_calculo == "Calcular DF Necessário":
                    Ai = calcular_disponibilidade_intrinseca(MTBF, MTTR)
                    DF_necessario = calcular_DF_necessario(Ai, UF, Ao_meta)
                    
                    if DF_necessario > 1:
                        st.error(f"⚠️ **Impossível atingir Ao = {Ao_meta*100:.1f}%**")
                        st.warning(f"""
                        Com MTBF={MTBF:.0f}h, MTTR={MTTR:.1f}h e UF={UF*100:.0f}%:
                        - Ai máximo = {Ai*100:.2f}%
                        - Ao máximo possível = {(Ai*UF)*100:.2f}%
                        
                        **Sugestões:**
                        - Aumentar MTBF (melhorar confiabilidade)
                        - Reduzir MTTR (melhorar manutenibilidade)
                        - Aumentar UF (melhorar utilização)
                        - Reduzir meta de Ao
                        """)
                    else:
                        st.success(f"✅ **DF Necessário: {DF_necessario*100:.2f}%**")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("DF Necessário", f"{DF_necessario*100:.2f}%")
                        
                        with col_b:
                            horas_paradas_max = HORAS_POR_MES * (1 - DF_necessario)
                            st.metric("Paradas Permitidas", f"{horas_paradas_max:.0f}h/mês")
                        
                        st.info(f"""
                        **Interpretação:**
                        
                        Para atingir Ao = {Ao_meta*100:.1f}%, você precisa:
                        - **DF ≥ {DF_necessario*100:.2f}%**
                        - Máximo de **{horas_paradas_max:.0f} horas/mês** de paradas programadas
                        - Isso equivale a **{horas_paradas_max/24:.1f} dias/mês** de paradas
                        """)
                
                elif modo_calculo == "Calcular UF Necessário":
                    Ai = calcular_disponibilidade_intrinseca(MTBF, MTTR)
                    UF_necessario = calcular_UF_necessario(Ai, DF, Ao_meta)
                    
                    if UF_necessario > 1:
                        st.error(f"⚠️ **Impossível atingir Ao = {Ao_meta*100:.1f}%**")
                        st.warning(f"""
                        Com MTBF={MTBF:.0f}h, MTTR={MTTR:.1f}h e DF={DF*100:.0f}%:
                        - Ai = {Ai*100:.2f}%
                        - Aa = {(Ai*DF)*100:.2f}%
                        - Ao máximo possível = {(Ai*DF)*100:.2f}%
                        
                        **Sugestões:**
                        - Aumentar MTBF ou reduzir MTTR
                        - Aumentar DF (reduzir paradas programadas)
                        - Reduzir meta de Ao
                        """)
                    else:
                        st.success(f"✅ **UF Necessário: {UF_necessario*100:.2f}%**")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("UF Necessário", f"{UF_necessario*100:.2f}%")
                        
                        with col_b:
                            horas_operacao_necessaria = HORAS_POR_MES * DF * UF_necessario
                            st.metric("Horas Operação Necessárias", f"{horas_operacao_necessaria:.0f}h/mês")
                        
                        st.info(f"""
                        **Interpretação:**
                        
                        Para atingir Ao = {Ao_meta*100:.1f}%, você precisa:
                        - **UF ≥ {UF_necessario*100:.2f}%**
                        - Operar pelo menos **{horas_operacao_necessaria:.0f} horas/mês**
                        - Do tempo disponível ({HORAS_POR_MES*DF:.0f}h), usar {UF_necessario*100:.1f}%
                        """)
                
                elif modo_calculo == "Calcular MTBF Necessário":
                    MTBF_necessario = calcular_MTBF_necessario(MTTR, DF, UF, Ao_meta)
                    
                    if MTBF_necessario == float('inf') or MTBF_necessario < 0:
                        st.error(f"⚠️ **Impossível atingir Ao = {Ao_meta*100:.1f}%**")
                        st.warning(f"""
                        Com os parâmetros atuais, não é possível atingir a meta.
                        
                        **O problema:**
                        - Ao_meta ({Ao_meta*100:.1f}%) > DF × UF ({(DF*UF)*100:.2f}%)
                        
                        **Sugestões:**
                        - Aumentar DF (reduzir paradas programadas)
                        - Aumentar UF (melhorar utilização)
                        - Reduzir meta de Ao
                        - Reduzir MTTR
                        """)
                    else:
                        st.success(f"✅ **MTBF Necessário: {MTBF_necessario:.0f} horas**")
                        
                        Ai_necessario = calcular_disponibilidade_intrinseca(MTBF_necessario, MTTR)
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("MTBF Necessário", f"{MTBF_necessario:.0f}h")
                        
                        with col_b:
                            st.metric("Ai Necessário", f"{Ai_necessario*100:.2f}%")
                        
                        with col_c:
                            falhas_max_mes = HORAS_POR_MES / MTBF_necessario
                            st.metric("Falhas Máx/Mês", f"{falhas_max_mes:.2f}")
                        
                        st.info(f"""
                        **Interpretação:**
                        
                        Para atingir Ao = {Ao_meta*100:.1f}%, você precisa:
                        - **MTBF ≥ {MTBF_necessario:.0f} horas**
                        - Ai ≥ {Ai_necessario*100:.2f}%
                        - Máximo de **{falhas_max_mes:.2f} falhas/mês**
                        - Intervalo médio entre falhas: **{MTBF_necessario/24:.1f} dias**
                        """)
                
                else:  # Calcular MTTR Máximo
                    MTTR_maximo = calcular_MTTR_maximo(MTBF, DF, UF, Ao_meta)
                    
                    if MTTR_maximo < 0:
                        st.error(f"⚠️ **Impossível atingir Ao = {Ao_meta*100:.1f}%**")
                        st.warning(f"""
                        Com MTBF={MTBF:.0f}h, DF={DF*100:.0f}% e UF={UF*100:.0f}%:
                        
                        **O problema:**
                        - Ao_meta muito alta para o MTBF atual
                        
                        **Sugestões:**
                        - Aumentar MTBF
                        - Aumentar DF ou UF
                        - Reduzir meta de Ao
                        """)
                    else:
                        st.success(f"✅ **MTTR Máximo: {MTTR_maximo:.1f} horas**")
                        
                        Ai_minimo = calcular_disponibilidade_intrinseca(MTBF, MTTR_maximo)
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("MTTR Máximo", f"{MTTR_maximo:.1f}h")
                        
                        with col_b:
                            st.metric("Ai Mínimo", f"{Ai_minimo*100:.2f}%")
                        
                        st.info(f"""
                        **Interpretação:**
                        
                        Para atingir Ao = {Ao_meta*100:.1f}%, você pode ter:
                        - **MTTR ≤ {MTTR_maximo:.1f} horas**
                        - Ai ≥ {Ai_minimo*100:.2f}%
                        - Tempo máximo de reparo por falha: **{MTTR_maximo:.1f} horas**
                        """)
                
                # Tabela resumo
                st.divider()
                st.subheader("📋 Resumo dos Parâmetros")
                
                if modo_calculo == "Calcular Disponibilidade (Ao)":
                    dados_resumo = {
                        'Parâmetro': ['MTBF', 'MTTR', 'DF', 'UF', 'Ai', 'Aa', 'Ao'],
                        'Valor': [
                            f"{MTBF:.1f}h",
                            f"{MTTR:.1f}h",
                            f"{DF*100:.2f}%",
                            f"{UF*100:.2f}%",
                            f"{Ai*100:.2f}%",
                            f"{Aa*100:.2f}%",
                            f"{Ao*100:.2f}%"
                        ],
                        'Descrição': [
                            'Mean Time Between Failures',
                            'Mean Time To Repair',
                            'Fator de Disponibilidade',
                            'Fator de Utilização',
                            'Disponibilidade Intrínseca',
                            'Disponibilidade Alcançada',
                            'Disponibilidade Operacional'
                        ]
                    }
                elif modo_calculo == "Calcular DF Necessário":
                    if DF_necessario <= 1:
                        dados_resumo = {
                            'Parâmetro': ['MTBF', 'MTTR', 'UF', 'Ao Meta', 'DF Necessário', 'Paradas Máx'],
                            'Valor': [
                                f"{MTBF:.1f}h",
                                f"{MTTR:.1f}h",
                                f"{UF*100:.2f}%",
                                f"{Ao_meta*100:.2f}%",
                                f"{DF_necessario*100:.2f}%",
                                f"{HORAS_POR_MES * (1 - DF_necessario):.0f}h/mês"
                            ]
                        }
                    else:
                        dados_resumo = None
                elif modo_calculo == "Calcular UF Necessário":
                    if UF_necessario <= 1:
                        dados_resumo = {
                            'Parâmetro': ['MTBF', 'MTTR', 'DF', 'Ao Meta', 'UF Necessário', 'Horas Op. Necessárias'],
                            'Valor': [
                                f"{MTBF:.1f}h",
                                f"{MTTR:.1f}h",
                                f"{DF*100:.2f}%",
                                f"{Ao_meta*100:.2f}%",
                                f"{UF_necessario*100:.2f}%",
                                f"{HORAS_POR_MES * DF * UF_necessario:.0f}h/mês"
                            ]
                        }
                    else:
                        dados_resumo = None
                elif modo_calculo == "Calcular MTBF Necessário":
                    if MTBF_necessario != float('inf') and MTBF_necessario > 0:
                        dados_resumo = {
                            'Parâmetro': ['MTTR', 'DF', 'UF', 'Ao Meta', 'MTBF Necessário', 'Falhas Máx/Mês'],
                            'Valor': [
                                f"{MTTR:.1f}h",
                                f"{DF*100:.2f}%",
                                f"{UF*100:.2f}%",
                                f"{Ao_meta*100:.2f}%",
                                f"{MTBF_necessario:.0f}h",
                                f"{HORAS_POR_MES / MTBF_necessario:.2f}"
                            ]
                        }
                    else:
                        dados_resumo = None
                else:  # MTTR Máximo
                    if MTTR_maximo >= 0:
                        dados_resumo = {
                            'Parâmetro': ['MTBF', 'DF', 'UF', 'Ao Meta', 'MTTR Máximo', 'Ai Mínimo'],
                            'Valor': [
                                f"{MTBF:.1f}h",
                                f"{DF*100:.2f}%",
                                f"{UF*100:.2f}%",
                                f"{Ao_meta*100:.2f}%",
                                f"{MTTR_maximo:.1f}h",
                                f"{Ai_minimo*100:.2f}%"
                            ]
                        }
                    else:
                        dados_resumo = None
                
                if dados_resumo:
                    df_resumo = pd.DataFrame(dados_resumo)
                    st.dataframe(df_resumo, use_container_width=True, hide_index=True)
                    
                    # Export
                    csv_data = df_resumo.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Download Resultados (CSV)",
                        data=csv_data,
                        file_name="disponibilidade_operacional.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Erro no cálculo: {str(e)}")
                st.exception(e)
    
    # ==================== TAB 2: ANÁLISE DE DEGRADAÇÃO ====================
    
    with tab2:
        st.header("📊 Análise de Degradação e Intervalo de PM")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Parâmetros")
            
            # Dados históricos
            st.markdown("**Dados Operacionais:**")
            
            HO_deg = st.number_input("Horas Operadas/Mês", min_value=1.0, value=600.0, step=10.0, key="deg2_HO")
            HF_deg = st.number_input("Horas em Falha/Mês", min_value=0.0, value=10.0, step=1.0, key="deg2_HF")
            Nf_deg = st.number_input("Número de Falhas/Mês", min_value=1, value=2, step=1, key="deg2_Nf")
            HD_deg = st.number_input("Horas Disponíveis/Mês", min_value=1.0, value=HORAS_POR_MES, step=10.0, key="deg2_HD")
            HP_deg = st.number_input("Horas Paradas Programadas/Mês", min_value=0.0, value=0.0, step=5.0, key="deg2_HP")
            
            st.divider()
            
            st.markdown("**Fatores Operacionais:**")
            
            DF_deg = st.slider("DF - Fator de Disponibilidade (%)", min_value=50.0, max_value=100.0, value=95.0, step=0.5, key="deg2_DF") / 100
            UF_deg = st.slider("UF - Fator de Utilização (%)", min_value=50.0, max_value=100.0, value=85.0, step=0.5, key="deg2_UF") / 100
            
            st.divider()
            
            st.markdown("**Parâmetros de Degradação:**")
            
            t_inicio_deg = st.slider("Tempo até Início do Desgaste (h)", min_value=50.0, max_value=500.0, value=200.0, step=10.0, key="deg2_t_inicio")
            beta_deg = st.slider("Intensidade da Degradação (β)", min_value=1.0, max_value=5.0, value=2.5, step=0.1, key="deg2_beta")
            Ao_minima = st.slider("Ao Mínima Aceitável (%)", min_value=70.0, max_value=95.0, value=85.0, step=1.0, key="deg2_Ao_min") / 100
        
        with col2:
            try:
                # Calcular KPIs
                kpis_deg = calcular_kpis_basicos(HO_deg, HF_deg, Nf_deg, HD_deg, HP_deg)
                MTBF_deg = kpis_deg['MTBF']
                MTTR_deg = kpis_deg['MTTR']
                
                lambda_base_deg = 1 / MTBF_deg
                
                # Encontrar intervalo ótimo
                resultado_deg = encontrar_intervalo_PM_por_disponibilidade(
                    lambda_base=lambda_base_deg,
                    beta_desgaste=beta_deg,
                    t_inicio_desgaste=t_inicio_deg,
                    MTTR=MTTR_deg,
                    DF=DF_deg,
                    UF=UF_deg,
                    Ao_minima=Ao_minima,
                    t_max=t_inicio_deg * 3
                )
                
                # Converter para calendário
                T_cal_deg = resultado_deg['T_otimo'] / (DF_deg * UF_deg)
                
                # Métricas
                st.subheader("🎯 Intervalo Ótimo de PM")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Intervalo PM", f"{resultado_deg['T_otimo']:.0f}h operadas")
                
                with col_b:
                    st.metric("Calendário", f"{T_cal_deg/24:.1f} dias")
                
                with col_c:
                    st.metric("PMs/Mês", f"{HO_deg/resultado_deg['T_otimo']:.2f}")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Ao no Ponto", f"{resultado_deg['disponibilidade']*100:.1f}%")
                
                with col_b:
                    st.metric("Confiabilidade", f"{resultado_deg['confiabilidade']*100:.1f}%")
                
                with col_c:
                    st.metric("Taxa de Falha", f"{resultado_deg['taxa_falha']:.4f}")
                
                st.divider()
                
                # Gráfico
                fig_deg = criar_grafico_degradacao_disponibilidade(resultado_deg, resultado_deg['T_otimo'], Ao_minima)
                st.plotly_chart(fig_deg, use_container_width=True)
                
                # Análise
                with st.expander("📋 Análise Detalhada"):
                    Ai_deg = calcular_disponibilidade_intrinseca(MTBF_deg, MTTR_deg)
                    Aa_deg = calcular_disponibilidade_alcancada(Ai_deg, DF_deg)
                    
                    st.markdown(f"""
                    **Parâmetros Base:**
                    - MTBF: {MTBF_deg:.1f}h
                    - MTTR: {MTTR_deg:.1f}h
                    - Ai: {Ai_deg*100:.2f}%
                    - DF: {DF_deg*100:.2f}%
                    - UF: {UF_deg*100:.2f}%
                    - Aa: {Aa_deg*100:.2f}%
                    
                    **Degradação:**
                    - Tempo até desgaste: {t_inicio_deg:.0f}h
                    - Intensidade (β): {beta_deg:.1f}
                    - Ao mínima: {Ao_minima*100:.1f}%
                    
                    **Resultado:**
                    - Intervalo PM: {resultado_deg['T_otimo']:.0f}h operadas ({T_cal_deg/24:.1f} dias calendário)
                    - Ao no ponto ótimo: {resultado_deg['disponibilidade']*100:.2f}%
                    - Confiabilidade: {resultado_deg['confiabilidade']*100:.2f}%
                    
                    **Interpretação:**
                    - A cada **{T_cal_deg/24:.1f} dias**, fazer PM para restaurar confiabilidade
                    - Isso garante Ao ≥ {Ao_minima*100:.1f}% durante todo o ciclo
                    - Frequência: **{(HO_deg/resultado_deg['T_otimo'])*12:.1f} PMs/ano**
                    """)
            
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
                st.exception(e)
    
    # ==================== TAB 3: MATRIZ DE DISPONIBILIDADE ====================
    
    with tab3:
        st.header("🗺️ Matriz de Disponibilidade")
        
        st.markdown("""
        **Visualize a relação entre MTBF, MTTR e DF/UF através de mapas de calor.**
        
        Escolha qual parâmetro fixar e veja como os outros se relacionam para atingir diferentes níveis de Ao.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuração da Matriz")
            
            parametro_fixo = st.selectbox(
                "Parâmetro Fixo:",
                ["MTBF", "MTTR", "DF"],
                help="Qual parâmetro manter constante"
            )
            
            if parametro_fixo == "MTBF":
                valor_fixo_matriz = st.number_input("Valor do MTBF (horas)", min_value=10.0, value=300.0, step=10.0, key="matriz_mtbf")
                
                st.markdown("**Parâmetros Variáveis:**")
                mttr_min = st.number_input("MTTR Mínimo (h)", min_value=0.1, value=1.0, step=0.5, key="matriz_mttr_min")
                mttr_max = st.number_input("MTTR Máximo (h)", min_value=0.1, value=20.0, step=0.5, key="matriz_mttr_max")
                
                df_min = st.number_input("DF Mínimo (%)", min_value=50.0, value=70.0, step=1.0, key="matriz_df_min") / 100
                df_max = st.number_input("DF Máximo (%)", min_value=50.0, value=100.0, step=1.0, key="matriz_df_max") / 100
                
                param1_name = "MTTR (horas)"
                param2_name = "DF (%)"
                
            elif parametro_fixo == "MTTR":
                valor_fixo_matriz = st.number_input("Valor do MTTR (horas)", min_value=0.1, value=5.0, step=0.5, key="matriz_mttr")
                
                st.markdown("**Parâmetros Variáveis:**")
                mtbf_min = st.number_input("MTBF Mínimo (h)", min_value=10.0, value=100.0, step=10.0, key="matriz_mtbf_min")
                mtbf_max = st.number_input("MTBF Máximo (h)", min_value=10.0, value=500.0, step=10.0, key="matriz_mtbf_max")
                
                df_min = st.number_input("DF Mínimo (%)", min_value=50.0, value=70.0, step=1.0, key="matriz_df_min2") / 100
                df_max = st.number_input("DF Máximo (%)", min_value=50.0, value=100.0, step=1.0, key="matriz_df_max2") / 100
                
                param1_name = "MTBF (horas)"
                param2_name = "DF (%)"
                
            else:  # DF fixo
                valor_fixo_matriz = st.slider("Valor do DF (%)", min_value=50.0, max_value=100.0, value=95.0, step=1.0, key="matriz_df") / 100
                
                st.markdown("**Parâmetros Variáveis:**")
                mtbf_min = st.number_input("MTBF Mínimo (h)", min_value=10.0, value=100.0, step=10.0, key="matriz_mtbf_min2")
                mtbf_max = st.number_input("MTBF Máximo (h)", min_value=10.0, value=500.0, step=10.0, key="matriz_mtbf_max2")
                
                mttr_min = st.number_input("MTTR Mínimo (h)", min_value=0.1, value=1.0, step=0.5, key="matriz_mttr_min2")
                mttr_max = st.number_input("MTTR Máximo (h)", min_value=0.1, value=20.0, step=0.5, key="matriz_mttr_max2")
                
                param1_name = "MTBF (horas)"
                param2_name = "MTTR (horas)"
            
            UF_matriz = st.slider("UF - Fator de Utilização (%)", min_value=50.0, max_value=100.0, value=85.0, step=1.0, key="matriz_uf") / 100
            
            resolucao = st.slider("Resolução da Matriz", min_value=10, max_value=50, value=25, step=5, help="Mais pontos = mais preciso mas mais lento")
        
        with col2:
            try:
                st.subheader(f"🗺️ Mapa de Calor: Ao vs {param1_name} vs {param2_name}")
                st.caption(f"Parâmetro fixo: {parametro_fixo} = {valor_fixo_matriz if parametro_fixo != 'DF' else valor_fixo_matriz*100}{'h' if parametro_fixo != 'DF' else '%'}, UF = {UF_matriz*100:.0f}%")
                
                # Gerar matriz
                if parametro_fixo == "MTBF":
                    range_param1 = (mttr_min, mttr_max)
                    range_param2 = (df_min, df_max)
                elif parametro_fixo == "MTTR":
                    range_param1 = (mtbf_min, mtbf_max)
                    range_param2 = (df_min, df_max)
                else:  # DF
                    range_param1 = (mtbf_min, mtbf_max)
                    range_param2 = (mttr_min, mttr_max)
                
                matriz, param1_vals, param2_vals = gerar_matriz_disponibilidade(
                    parametro_fixo=parametro_fixo,
                    valor_fixo=valor_fixo_matriz,
                    range_param1=range_param1,
                    range_param2=range_param2,
                    DF=valor_fixo_matriz if parametro_fixo == "DF" else 0.95,
                    UF=UF_matriz,
                    n_pontos=resolucao
                )
                
                # Converter param2_vals para percentual se for DF
                if parametro_fixo in ["MTBF", "MTTR"]:
                    param2_vals_display = param2_vals * 100
                    param2_name_display = "DF (%)"
                else:
                    param2_vals_display = param2_vals
                    param2_name_display = param2_name
                
                # Criar heatmap
                fig_matriz = criar_heatmap_disponibilidade(
                    matriz=matriz,
                    param1_vals=param1_vals,
                    param2_vals=param2_vals_display,
                    param1_name=param1_name,
                    param2_name=param2_name_display,
                    titulo=f"Disponibilidade Operacional (Ao) - {parametro_fixo} fixo"
                )
                
                st.plotly_chart(fig_matriz, use_container_width=True)
                
                # Análise
                with st.expander("📊 Análise da Matriz"):
                    Ao_max = np.max(matriz)
                    Ao_min = np.min(matriz)
                    Ao_media = np.mean(matriz)
                    
                    # Encontrar pontos de interesse
                    idx_max = np.unravel_index(np.argmax(matriz), matriz.shape)
                    idx_min = np.unravel_index(np.argmin(matriz), matriz.shape)
                    
                    st.markdown(f"""
                    **Estatísticas da Matriz:**
                    - Ao Máximo: **{Ao_max:.1f}%** (melhor cenário)
                    - Ao Mínimo: **{Ao_min:.1f}%** (pior cenário)
                    - Ao Médio: **{Ao_media:.1f}%**
                    
                    **Melhor Cenário:**
                    - {param1_name}: {param1_vals[idx_max[0]]:.1f}
                    - {param2_name_display}: {param2_vals_display[idx_max[1]]:.1f}
                    - Ao: {Ao_max:.1f}%
                    
                    **Pior Cenário:**
                    - {param1_name}: {param1_vals[idx_min[0]]:.1f}
                    - {param2_name_display}: {param2_vals_display[idx_min[1]]:.1f}
                    - Ao: {Ao_min:.1f}%
                    
                    **Interpretação:**
                    - Zonas **verdes**: Alta disponibilidade operacional (>90%)
                    - Zonas **amarelas**: Disponibilidade moderada (80-90%)
                    - Zonas **vermelhas**: Baixa disponibilidade (<80%)
                    
                    Use esta matriz para:
                    - Definir metas de MTBF/MTTR baseadas em Ao desejado
                    - Avaliar impacto de melhorias na confiabilidade
                    - Planejar ações de manutenção
                    """)
                
                # Export da matriz
                df_matriz = pd.DataFrame(
                    matriz,
                    index=[f"{param1_name}: {v:.1f}" for v in param1_vals],
                    columns=[f"{param2_name_display}: {v:.1f}" for v in param2_vals_display]
                )
                
                csv_matriz = df_matriz.to_csv(encoding='utf-8-sig')
                st.download_button(
                    label="📥 Download Matriz (CSV)",
                    data=csv_matriz,
                    file_name="matriz_disponibilidade.csv",
                    mime="text/csv"
                )
            
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
                st.exception(e)
    
    # ==================== RODAPÉ ====================
    
    st.divider()
    st.markdown("""
    **Sobre esta ferramenta v3.0:**
    
    Calculadora focada em **disponibilidade operacional** para cumprimento de metas de produção.
    
    **Funcionalidades:**
    - 🎯 Cálculo de Ao baseado em MTBF, MTTR, DF e UF
    - 🔄 Cálculo reverso: determine parâmetros necessários para meta de Ao
    - 📊 Análise de degradação com intervalo ótimo de PM
    - 🗺️ Matriz interativa relacionando MTBF, MTTR e DF
    
    **Fórmulas:**

    
    $$A_i = \\frac{MTBF}{MTBF + MTTR}$$

    
    $$A_a = A_i \\times DF$$

    
    $$A_o = A_i \\times DF \\times UF$$
    
    **Referências:**
    - IEC 60300-3-1: Dependability management
    - MIL-HDBK-338B: Electronic Reliability Design Handbook
    """)

if __name__ == "__main__":
    main()
