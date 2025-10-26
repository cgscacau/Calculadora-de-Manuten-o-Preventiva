"""
Calculadora de Disponibilidade Operacional - Versão Simplificada
Versão: 4.0.0 (Meta vs Realizado)
Autor: Sistema de Engenharia de Confiabilidade
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Calculadora de Disponibilidade - Meta vs Realizado",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSTANTES ====================
HORAS_POR_MES = 730.0
DIAS_POR_MES = 30.44

# ==================== FUNÇÕES DE CÁLCULO ====================

def calcular_disponibilidade_intrinseca(MTBF: float, MTTR: float) -> float:
    """Ai = MTBF / (MTBF + MTTR)"""
    return MTBF / (MTBF + MTTR) if (MTBF + MTTR) > 0 else 0

def calcular_disponibilidade_alcancada(Ai: float, DF: float) -> float:
    """Aa = Ai × DF"""
    return Ai * DF

def calcular_disponibilidade_operacional(Ai: float, DF: float, UF: float) -> float:
    """Ao = Ai × DF × UF"""
    return Ai * DF * UF

def calcular_horas_producao(Ao: float, horas_mes: float = HORAS_POR_MES) -> float:
    """Horas disponíveis para produção no mês"""
    return Ao * horas_mes

def calcular_MTBF_necessario(MTTR: float, DF: float, UF: float, Ao_meta: float) -> float:
    """
    Calcula MTBF necessário para atingir Ao_meta.
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

def calcular_gap_analise(Ao_atual: float, Ao_meta: float, MTBF_atual: float, MTTR_atual: float, 
                         DF_meta: float, UF_meta: float) -> dict:
    """
    Analisa o gap entre situação atual e meta.
    Fornece recomendações de melhoria.
    """
    gap_percentual = ((Ao_meta - Ao_atual) / Ao_atual * 100) if Ao_atual > 0 else 0
    
    # Calcular MTBF necessário mantendo MTTR atual
    MTBF_necessario = calcular_MTBF_necessario(MTTR_atual, DF_meta, UF_meta, Ao_meta)
    
    # Calcular MTTR máximo mantendo MTBF atual
    MTTR_maximo = calcular_MTTR_maximo(MTBF_atual, DF_meta, UF_meta, Ao_meta)
    
    # Calcular melhorias necessárias
    melhoria_MTBF_percentual = ((MTBF_necessario - MTBF_atual) / MTBF_atual * 100) if MTBF_necessario != float('inf') else float('inf')
    melhoria_MTTR_percentual = ((MTTR_atual - MTTR_maximo) / MTTR_atual * 100) if MTTR_maximo >= 0 else 0
    
    return {
        'gap_percentual': gap_percentual,
        'MTBF_necessario': MTBF_necessario,
        'MTTR_maximo': MTTR_maximo,
        'melhoria_MTBF_percentual': melhoria_MTBF_percentual,
        'melhoria_MTTR_percentual': melhoria_MTTR_percentual,
        'atingivel': MTBF_necessario != float('inf') and MTTR_maximo >= 0
    }

def calcular_numero_falhas(MTBF: float, horas_operadas: float) -> float:
    """Número esperado de falhas no período"""
    return horas_operadas / MTBF if MTBF > 0 else 0

def calcular_tempo_reparo_total(MTTR: float, num_falhas: float) -> float:
    """Tempo total em reparo no período"""
    return MTTR * num_falhas

# ==================== MODELO DE DEGRADAÇÃO ====================

def taxa_falha_degradacao(t: float, lambda_base: float, beta_desgaste: float, t_inicio_desgaste: float) -> float:
    """Taxa de falha com degradação progressiva."""
    if t <= t_inicio_desgaste:
        return lambda_base
    else:
        t_desgaste = t - t_inicio_desgaste
        return lambda_base * (1 + (t_desgaste / t_inicio_desgaste) ** beta_desgaste)

def confiabilidade_degradacao(t: float, lambda_base: float, beta_desgaste: float, t_inicio_desgaste: float, n_pontos: int = 500) -> float:
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
    
    tempo_total = t + MTTR * (1 - R_t)
    Ai_t = (t * R_t) / tempo_total if tempo_total > 0 else 0
    
    Ao_t = Ai_t * DF * UF
    
    return Ao_t

def encontrar_intervalo_PM_otimo(
    MTBF: float,
    MTTR: float,
    DF: float,
    UF: float,
    Ao_minima: float,
    t_inicio_desgaste: float = None,
    beta_desgaste: float = 2.5
) -> dict:
    """
    Encontra intervalo ótimo de PM baseado em disponibilidade operacional mínima.
    """
    if t_inicio_desgaste is None:
        t_inicio_desgaste = MTBF * 0.7  # 70% do MTBF como padrão
    
    lambda_base = 1 / MTBF
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
        'taxas_falha': taxas_falha,
        't_inicio_desgaste': t_inicio_desgaste
    }

# ==================== PLOTAGEM ====================

def criar_grafico_comparativo(Ao_atual: float, Ao_meta: float, Ai_atual: float, Ai_meta: float,
                               DF_atual: float, DF_meta: float, UF_atual: float, UF_meta: float) -> go.Figure:
    """Cria gráfico comparativo entre situação atual e meta."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Disponibilidade Operacional',
            'Disponibilidade Intrínseca',
            'Fator de Disponibilidade (DF)',
            'Fator de Utilização (UF)'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Ao
    fig.add_trace(
        go.Bar(x=['Atual', 'Meta'], y=[Ao_atual*100, Ao_meta*100],
               marker_color=['#FF6B6B', '#4ECDC4'],
               text=[f'{Ao_atual*100:.1f}%', f'{Ao_meta*100:.1f}%'],
               textposition='outside'),
        row=1, col=1
    )
    
    # Ai
    fig.add_trace(
        go.Bar(x=['Atual', 'Meta'], y=[Ai_atual*100, Ai_meta*100],
               marker_color=['#FF6B6B', '#4ECDC4'],
               text=[f'{Ai_atual*100:.1f}%', f'{Ai_meta*100:.1f}%'],
               textposition='outside'),
        row=1, col=2
    )
    
    # DF
    fig.add_trace(
        go.Bar(x=['Atual', 'Meta'], y=[DF_atual*100, DF_meta*100],
               marker_color=['#FF6B6B', '#4ECDC4'],
               text=[f'{DF_atual*100:.1f}%', f'{DF_meta*100:.1f}%'],
               textposition='outside'),
        row=2, col=1
    )
    
    # UF
    fig.add_trace(
        go.Bar(x=['Atual', 'Meta'], y=[UF_atual*100, UF_meta*100],
               marker_color=['#FF6B6B', '#4ECDC4'],
               text=[f'{UF_atual*100:.1f}%', f'{UF_meta*100:.1f}%'],
               textposition='outside'),
        row=2, col=2
    )
    
    fig.update_yaxes(title_text="%", range=[0, 105])
    fig.update_layout(height=600, showlegend=False, title_text="Comparação: Atual vs Meta")
    
    return fig

def criar_grafico_degradacao(resultado: dict, T_otimo: float, Ao_minima: float) -> go.Figure:
    """Gráfico de degradação com intervalo de PM."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Disponibilidade Operacional ao Longo do Tempo',
            'Confiabilidade (Probabilidade de Não Falhar)',
            'Taxa de Falha Instantânea',
            'Evolução Combinada'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    t_vals = resultado['t_vals']
    
    # Subplot 1: Disponibilidade
    fig.add_trace(
        go.Scatter(x=t_vals, y=resultado['disponibilidades']*100, mode='lines',
                   name='Ao(t)', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    fig.add_hline(y=Ao_minima*100, line_dash="dash", line_color="red",
                  annotation_text=f"Ao mínima: {Ao_minima*100:.1f}%", row=1, col=1)
    
    # Subplot 2: Confiabilidade
    fig.add_trace(
        go.Scatter(x=t_vals, y=resultado['confiabilidades']*100, mode='lines',
                   name='R(t)', line=dict(color='green', width=3)),
        row=1, col=2
    )
    
    # Subplot 3: Taxa de Falha
    fig.add_trace(
        go.Scatter(x=t_vals, y=resultado['taxas_falha'], mode='lines',
                   name='λ(t)', line=dict(color='red', width=3)),
        row=2, col=1
    )
    
    # Subplot 4: Evolução combinada normalizada
    Ao_norm = (resultado['disponibilidades'] - resultado['disponibilidades'].min()) / (resultado['disponibilidades'].max() - resultado['disponibilidades'].min()) * 100
    R_norm = (resultado['confiabilidades'] - resultado['confiabilidades'].min()) / (resultado['confiabilidades'].max() - resultado['confiabilidades'].min()) * 100
    
    fig.add_trace(
        go.Scatter(x=t_vals, y=Ao_norm, mode='lines', name='Ao normalizado',
                   line=dict(color='blue', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=t_vals, y=R_norm, mode='lines', name='R normalizado',
                   line=dict(color='green', width=2, dash='dash')),
        row=2, col=2
    )
    
    # Linha vertical no ponto ótimo
    for row in [1, 2]:
        for col in [1, 2]:
            if col == 1 or (row == 2 and col == 2):
                fig.add_vline(x=T_otimo, line_dash="dash", line_color="purple",
                             opacity=0.7, annotation_text=f"PM: {T_otimo:.0f}h",
                             row=row, col=col)
    
    fig.update_xaxes(title_text="Horas Operadas")
    fig.update_yaxes(title_text="Ao (%)", row=1, col=1)
    fig.update_yaxes(title_text="R(t) (%)", row=1, col=2)
    fig.update_yaxes(title_text="λ(t)", row=2, col=1)
    fig.update_yaxes(title_text="Normalizado (%)", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, 
                     title_text="Análise de Degradação - Intervalo Ótimo de PM")
    
    return fig

def criar_grafico_matriz_mtbf_mttr(DF: float, UF: float, 
                                     mtbf_range: Tuple[float, float] = (100, 500),
                                     mttr_range: Tuple[float, float] = (1, 20),
                                     n_pontos: int = 30) -> go.Figure:
    """Cria matriz de disponibilidade MTBF vs MTTR."""
    
    mtbf_vals = np.linspace(mtbf_range[0], mtbf_range[1], n_pontos)
    mttr_vals = np.linspace(mttr_range[0], mttr_range[1], n_pontos)
    
    matriz = np.zeros((n_pontos, n_pontos))
    
    for i, mtbf in enumerate(mtbf_vals):
        for j, mttr in enumerate(mttr_vals):
            Ai = calcular_disponibilidade_intrinseca(mtbf, mttr)
            Ao = calcular_disponibilidade_operacional(Ai, DF, UF)
            matriz[i, j] = Ao * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=matriz,
        x=mttr_vals,
        y=mtbf_vals,
        colorscale='RdYlGn',
        colorbar=dict(title="Ao (%)"),
        hovertemplate='MTTR: %{x:.1f}h<br>MTBF: %{y:.1f}h<br>Ao: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Matriz de Disponibilidade (DF={DF*100:.0f}%, UF={UF*100:.0f}%)",
        xaxis_title="MTTR (horas)",
        yaxis_title="MTBF (horas)",
        height=600
    )
    
    return fig

# ==================== INTERFACE STREAMLIT ====================

def main():
    st.title("🎯 Calculadora de Disponibilidade Operacional")
    st.markdown("### Meta vs Realizado - Análise Completa")
    
    st.markdown("""
    **Entre com sua meta mensal e o desempenho atual - o sistema calcula tudo!**
    
    📊 O que você obtém:
    - ✅ Análise completa de disponibilidade (Ai, Aa, Ao)
    - ✅ Gap entre situação atual e meta
    - ✅ Recomendações de melhoria (MTBF e MTTR)
    - ✅ Intervalo ótimo de manutenção preventiva
    - ✅ Matriz de disponibilidade interativa
    """)
    
    st.divider()
    
    # ==================== INPUTS ====================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 META MENSAL")
        
        DF_meta = st.slider(
            "DF Meta - Fator de Disponibilidade (%)",
            min_value=80.0,
            max_value=100.0,
            value=95.0,
            step=0.5,
            help="Meta de tempo disponível para operação (sem paradas programadas excessivas)"
        ) / 100
        
        UF_meta = st.slider(
            "UF Meta - Fator de Utilização (%)",
            min_value=70.0,
            max_value=100.0,
            value=90.0,
            step=0.5,
            help="Meta de utilização do tempo disponível"
        ) / 100
        
        Ao_meta = DF_meta * UF_meta  # Simplificação: assumindo Ai = 1 para meta
        
        st.info(f"""
        **Meta Combinada:**
        - Ao Meta (assumindo Ai=100%): **{Ao_meta*100:.2f}%**
        - Horas produção/mês: **{Ao_meta*HORAS_POR_MES:.0f}h**
        """)
    
    with col2:
        st.subheader("📊 DESEMPENHO ATUAL (Mês Anterior)")
        
        MTBF_atual = st.number_input(
            "MTBF Atual (horas)",
            min_value=1.0,
            value=300.0,
            step=10.0,
            help="Mean Time Between Failures do mês anterior"
        )
        
        MTTR_atual = st.number_input(
            "MTTR Atual (horas)",
            min_value=0.1,
            value=5.0,
            step=0.5,
            help="Mean Time To Repair do mês anterior"
        )
        
        st.markdown("**Fatores Operacionais Atuais:**")
        
        DF_atual = st.slider(
            "DF Atual (%)",
            min_value=50.0,
            max_value=100.0,
            value=92.0,
            step=0.5
        ) / 100
        
        UF_atual = st.slider(
            "UF Atual (%)",
            min_value=50.0,
            max_value=100.0,
            value=85.0,
            step=0.5
        ) / 100
    
    st.divider()
    
    # ==================== CÁLCULOS ====================
    
    # Situação Atual
    Ai_atual = calcular_disponibilidade_intrinseca(MTBF_atual, MTTR_atual)
    Aa_atual = calcular_disponibilidade_alcancada(Ai_atual, DF_atual)
    Ao_atual = calcular_disponibilidade_operacional(Ai_atual, DF_atual, UF_atual)
    horas_prod_atual = calcular_horas_producao(Ao_atual)
    
    # Situação Meta (assumindo melhorias em MTBF/MTTR)
    Ai_meta = 1.0  # Meta ideal
    Aa_meta = calcular_disponibilidade_alcancada(Ai_meta, DF_meta)
    horas_prod_meta = calcular_horas_producao(Ao_meta)
    
    # Análise de Gap
    gap_analise = calcular_gap_analise(Ao_atual, Ao_meta, MTBF_atual, MTTR_atual, DF_meta, UF_meta)
    
    # Número de falhas e tempo de reparo
    horas_operadas_mes = HORAS_POR_MES * DF_atual * UF_atual
    num_falhas_atual = calcular_numero_falhas(MTBF_atual, horas_operadas_mes)
    tempo_reparo_total_atual = calcular_tempo_reparo_total(MTTR_atual, num_falhas_atual)
    
    # ==================== RESULTADOS PRINCIPAIS ====================
    
    st.header("📈 RESULTADOS DA ANÁLISE")
    
    # Métricas principais
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_Ao = (Ao_atual - Ao_meta) * 100
        st.metric(
            "Ao Atual",
            f"{Ao_atual*100:.2f}%",
            delta=f"{delta_Ao:.2f}%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Ao Meta",
            f"{Ao_meta*100:.2f}%",
            help="Meta de disponibilidade operacional"
        )
    
    with col3:
        gap = Ao_meta - Ao_atual
        st.metric(
            "Gap",
            f"{gap*100:.2f}%",
            delta=f"{gap_analise['gap_percentual']:.1f}% relativo",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Horas Prod. Atual",
            f"{horas_prod_atual:.0f}h/mês"
        )
    
    with col5:
        delta_horas = horas_prod_meta - horas_prod_atual
        st.metric(
            "Horas Prod. Meta",
            f"{horas_prod_meta:.0f}h/mês",
            delta=f"{delta_horas:+.0f}h"
        )
    
    st.divider()
    
    # Detalhamento em colunas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Situação Atual Detalhada")
        
        dados_atual = pd.DataFrame({
            'Indicador': [
                'MTBF',
                'MTTR',
                'Ai - Disponibilidade Intrínseca',
                'DF - Fator de Disponibilidade',
                'UF - Fator de Utilização',
                'Aa - Disponibilidade Alcançada',
                'Ao - Disponibilidade Operacional',
                'Horas Produção/Mês',
                'Falhas Esperadas/Mês',
                'Tempo Total em Reparo/Mês'
            ],
            'Valor Atual': [
                f"{MTBF_atual:.1f}h",
                f"{MTTR_atual:.1f}h",
                f"{Ai_atual*100:.2f}%",
                f"{DF_atual*100:.2f}%",
                f"{UF_atual*100:.2f}%",
                f"{Aa_atual*100:.2f}%",
                f"{Ao_atual*100:.2f}%",
                f"{horas_prod_atual:.0f}h",
                f"{num_falhas_atual:.2f}",
                f"{tempo_reparo_total_atual:.1f}h"
            ]
        })
        
        st.dataframe(dados_atual, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🎯 Para Atingir a Meta")
        
        if gap_analise['atingivel']:
            st.success("✅ Meta atingível com melhorias!")
            
            recomendacoes = pd.DataFrame({
                'Estratégia': [
                    'Opção 1: Melhorar MTBF',
                    'Opção 2: Melhorar MTTR',
                    'Opção 3: Melhorar DF',
                    'Opção 4: Melhorar UF'
                ],
                'Ação Necessária': [
                    f"MTBF ≥ {gap_analise['MTBF_necessario']:.0f}h (+{gap_analise['melhoria_MTBF_percentual']:.1f}%)" if gap_analise['MTBF_necessario'] != float('inf') else "Não aplicável",
                    f"MTTR ≤ {gap_analise['MTTR_maximo']:.1f}h (-{gap_analise['melhoria_MTTR_percentual']:.1f}%)" if gap_analise['MTTR_maximo'] >= 0 else "Não aplicável",
                    f"DF ≥ {DF_meta*100:.1f}% (atual: {DF_atual*100:.1f}%)",
                    f"UF ≥ {UF_meta*100:.1f}% (atual: {UF_atual*100:.1f}%)"
                ]
            })
            
            st.dataframe(recomendacoes, use_container_width=True, hide_index=True)
            
            st.info(f"""
            **Interpretação:**
            
            Para atingir **Ao = {Ao_meta*100:.2f}%**, você pode:
            
            1. **Aumentar MTBF** de {MTBF_atual:.0f}h para {gap_analise['MTBF_necessario']:.0f}h
               - Melhoria necessária: **{gap_analise['melhoria_MTBF_percentual']:.1f}%**
               - Ações: Melhorar confiabilidade, manutenção preditiva, substituição de componentes críticos
            
            2. **Reduzir MTTR** de {MTTR_atual:.1f}h para {gap_analise['MTTR_maximo']:.1f}h
               - Redução necessária: **{gap_analise['melhoria_MTTR_percentual']:.1f}%**
               - Ações: Treinamento, peças em estoque, procedimentos otimizados
            
            3. **Aumentar DF** de {DF_atual*100:.1f}% para {DF_meta*100:.1f}%
               - Ações: Reduzir paradas programadas, otimizar setup
            
            4. **Aumentar UF** de {UF_atual*100:.1f}% para {UF_meta*100:.1f}%
               - Ações: Melhorar planejamento de produção, reduzir ociosidade
            
            💡 **Recomendação:** Combine melhorias em MTBF e MTTR para resultados mais sustentáveis.
            """)
        else:
            st.error("⚠️ Meta muito ambiciosa com parâmetros atuais!")
            st.warning(f"""
            A meta de **Ao = {Ao_meta*100:.2f}%** não é atingível com:
            - DF meta: {DF_meta*100:.1f}%
            - UF meta: {UF_meta*100:.1f}%
            
            **Ao máximo teórico** com DF e UF metas: **{(DF_meta * UF_meta)*100:.2f}%**
            
            **Sugestões:**
            1. Revisar metas de DF e UF (aumentar)
            2. Aceitar meta de Ao mais realista
            3. Investir em melhorias radicais de confiabilidade
            """)
    
    st.divider()
    
    # ==================== GRÁFICOS ====================
    
    st.header("📊 VISUALIZAÇÕES")
    
    tab1, tab2, tab3 = st.tabs(["Comparativo", "Degradação e PM", "Matriz MTBF vs MTTR"])
    
    with tab1:
        st.subheader("Comparação: Atual vs Meta")
        
        fig_comp = criar_grafico_comparativo(
            Ao_atual, Ao_meta, Ai_atual, Ai_meta,
            DF_atual, DF_meta, UF_atual, UF_meta
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Análise adicional
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Situação Atual:**")
            st.write(f"- Ai: {Ai_atual*100:.2f}%")
            st.write(f"- Aa: {Aa_atual*100:.2f}%")
            st.write(f"- Ao: {Ao_atual*100:.2f}%")
            st.write(f"- Produção: {horas_prod_atual:.0f}h/mês")
        
        with col2:
            st.markdown("**🟢 Meta:**")
            st.write(f"- Ai: {Ai_meta*100:.2f}% (ideal)")
            st.write(f"- Aa: {Aa_meta*100:.2f}%")
            st.write(f"- Ao: {Ao_meta*100:.2f}%")
            st.write(f"- Produção: {horas_prod_meta:.0f}h/mês")
    
    with tab2:
        st.subheader("Análise de Degradação e Intervalo Ótimo de PM")
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            Ao_minima_pm = st.slider(
                "Ao Mínima Aceitável para PM (%)",
                min_value=70.0,
                max_value=95.0,
                value=85.0,
                step=1.0,
                help="Disponibilidade mínima antes de fazer PM"
            ) / 100
        
        with col_config2:
            beta_desgaste = st.slider(
                "Intensidade de Degradação (β)",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                step=0.1,
                help="Quanto maior, mais rápida é a degradação"
            )
        
        # Calcular intervalo de PM
        resultado_pm = encontrar_intervalo_PM_otimo(
            MTBF=MTBF_atual,
            MTTR=MTTR_atual,
            DF=DF_meta,  # Usar meta para planejamento futuro
            UF=UF_meta,
            Ao_minima=Ao_minima_pm,
            beta_desgaste=beta_desgaste
        )
        
        T_cal = resultado_pm['T_otimo'] / (DF_meta * UF_meta)
        frequencia_pm_mes = horas_operadas_mes / resultado_pm['T_otimo']
        
        col_pm1, col_pm2, col_pm3, col_pm4 = st.columns(4)
        
        with col_pm1:
            st.metric("Intervalo PM", f"{resultado_pm['T_otimo']:.0f}h operadas")
        
        with col_pm2:
            st.metric("Calendário", f"{T_cal/24:.1f} dias")
        
        with col_pm3:
            st.metric("PMs/Mês", f"{frequencia_pm_mes:.2f}")
        
        with col_pm4:
            st.metric("PMs/Ano", f"{frequencia_pm_mes*12:.1f}")
        
        fig_pm = criar_grafico_degradacao(resultado_pm, resultado_pm['T_otimo'], Ao_minima_pm)
        st.plotly_chart(fig_pm, use_container_width=True)
        
        with st.expander("📋 Interpretação do Intervalo de PM"):
            st.markdown(f"""
            **Análise do Ciclo de Degradação:**
            
            1. **Fase Estável** (0 a {resultado_pm['t_inicio_desgaste']:.0f}h):
               - Taxa de falha constante
               - Disponibilidade alta e estável
               - Período ideal de operação
            
            2. **Início da Degradação** ({resultado_pm['t_inicio_desgaste']:.0f}h):
               - Componentes começam a desgastar
               - Taxa de falha aumenta gradualmente
            
            3. **Ponto Ótimo de PM** ({resultado_pm['T_otimo']:.0f}h operadas):
               - Disponibilidade: {resultado_pm['disponibilidade']*100:.2f}%
               - Confiabilidade: {resultado_pm['confiabilidade']*100:.2f}%
               - Taxa de falha: {resultado_pm['taxa_falha']:.4f}
               - **Momento ideal para intervenção preventiva**
            
            **Recomendação de Manutenção:**
            - Fazer PM a cada **{T_cal/24:.1f} dias** (calendário)
            - Ou a cada **{resultado_pm['T_otimo']:.0f} horas** de operação
            - Frequência: **{frequencia_pm_mes:.2f} PMs/mês** ou **{frequencia_pm_mes*12:.1f} PMs/ano**
            - Isso garante Ao ≥ {Ao_minima_pm*100:.1f}% durante todo o ciclo
            
            💡 **Benefício:** Evita degradação severa e mantém alta disponibilidade.
            """)
    
    with tab3:
        st.subheader("Matriz de Disponibilidade: MTBF vs MTTR")
        
        st.markdown(f"""
        **Explore diferentes cenários de MTBF e MTTR**
        
        Configuração atual: DF = {DF_meta*100:.0f}%, UF = {UF_meta*100:.0f}%
        """)
        
        col_matriz1, col_matriz2 = st.columns(2)
        
        with col_matriz1:
            mtbf_min = st.number_input("MTBF Mínimo (h)", min_value=5.0, value=100.0, step=10.0, key="mtbf_min")
            mtbf_max = st.number_input("MTBF Máximo (h)", min_value=50.0, value=500.0, step=10.0, key="mtbf_max")
        
        with col_matriz2:
            mttr_min = st.number_input("MTTR Mínimo (h)", min_value=0.5, value=1.0, step=0.5, key="mttr_min")
            mttr_max = st.number_input("MTTR Máximo (h)", min_value=0.5, value=20.0, step=0.5, key="mttr_max")
        
        fig_matriz = criar_grafico_matriz_mtbf_mttr(
            DF=DF_meta,
            UF=UF_meta,
            mtbf_range=(mtbf_min, mtbf_max),
            mttr_range=(mttr_min, mttr_max),
            n_pontos=30
        )
        st.plotly_chart(fig_matriz, use_container_width=True)
        
        # Marcar ponto atual e meta
        st.info(f"""
        **Sua Posição na Matriz:**
        
        🔴 **Atual:** MTBF = {MTBF_atual:.0f}h, MTTR = {MTTR_atual:.1f}h → Ao = {Ao_atual*100:.2f}%
        
        🟢 **Para atingir meta (Ao = {Ao_meta*100:.2f}%):**
        - Opção 1: MTBF ≥ {gap_analise['MTBF_necessario']:.0f}h (mantendo MTTR = {MTTR_atual:.1f}h)
        - Opção 2: MTTR ≤ {gap_analise['MTTR_maximo']:.1f}h (mantendo MTBF = {MTBF_atual:.0f}h)
        - Opção 3: Combinação de melhorias em ambos
        
        **Interpretação das Cores:**
        - 🟢 Verde (>90%): Excelente disponibilidade
        - 🟡 Amarelo (80-90%): Disponibilidade adequada
        - 🟠 Laranja (70-80%): Atenção necessária
        - 🔴 Vermelho (<70%): Crítico - ação urgente
        """)
    
    st.divider()
    
    # ==================== EXPORT ====================
    
    st.header("💾 Exportar Resultados")
    
    # Criar DataFrame consolidado
    df_export = pd.DataFrame({
        'Categoria': [
            'MTBF Atual', 'MTTR Atual', 'DF Atual', 'UF Atual',
            'Ai Atual', 'Aa Atual', 'Ao Atual', 'Horas Produção Atual',
            'DF Meta', 'UF Meta', 'Ao Meta', 'Horas Produção Meta',
            'Gap Ao (%)', 'Gap Horas',
            'MTBF Necessário', 'MTTR Máximo',
            'Melhoria MTBF (%)', 'Melhoria MTTR (%)',
            'Falhas Esperadas/Mês', 'Tempo Reparo Total/Mês',
            'Intervalo PM Ótimo (h)', 'Intervalo PM (dias)',
            'Frequência PM/Mês', 'Frequência PM/Ano'
        ],
        'Valor': [
            f"{MTBF_atual:.1f}h", f"{MTTR_atual:.1f}h", f"{DF_atual*100:.2f}%", f"{UF_atual*100:.2f}%",
            f"{Ai_atual*100:.2f}%", f"{Aa_atual*100:.2f}%", f"{Ao_atual*100:.2f}%", f"{horas_prod_atual:.0f}h",
            f"{DF_meta*100:.2f}%", f"{UF_meta*100:.2f}%", f"{Ao_meta*100:.2f}%", f"{horas_prod_meta:.0f}h",
            f"{gap*100:.2f}%", f"{delta_horas:.0f}h",
            f"{gap_analise['MTBF_necessario']:.0f}h" if gap_analise['MTBF_necessario'] != float('inf') else "N/A",
            f"{gap_analise['MTTR_maximo']:.1f}h" if gap_analise['MTTR_maximo'] >= 0 else "N/A",
            f"{gap_analise['melhoria_MTBF_percentual']:.1f}%" if gap_analise['melhoria_MTBF_percentual'] != float('inf') else "N/A",
            f"{gap_analise['melhoria_MTTR_percentual']:.1f}%",
            f"{num_falhas_atual:.2f}", f"{tempo_reparo_total_atual:.1f}h",
            f"{resultado_pm['T_otimo']:.0f}h", f"{T_cal/24:.1f} dias",
            f"{frequencia_pm_mes:.2f}", f"{frequencia_pm_mes*12:.1f}"
        ]
    })
    
    csv_export = df_export.to_csv(index=False, encoding='utf-8-sig')
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.download_button(
            label="📥 Download Análise Completa (CSV)",
            data=csv_export,
            file_name="analise_disponibilidade_completa.csv",
            mime="text/csv"
        )
    
    with col_exp2:
        # Criar relatório em texto
        relatorio = f"""
RELATÓRIO DE DISPONIBILIDADE OPERACIONAL
==========================================

DATA: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

SITUAÇÃO ATUAL
--------------
MTBF: {MTBF_atual:.1f}h
MTTR: {MTTR_atual:.1f}h
DF: {DF_atual*100:.2f}%
UF: {UF_atual*100:.2f}%

Disponibilidade Intrínseca (Ai): {Ai_atual*100:.2f}%
Disponibilidade Alcançada (Aa): {Aa_atual*100:.2f}%
Disponibilidade Operacional (Ao): {Ao_atual*100:.2f}%

Horas Produção/Mês: {horas_prod_atual:.0f}h
Falhas Esperadas/Mês: {num_falhas_atual:.2f}
Tempo Total em Reparo/Mês: {tempo_reparo_total_atual:.1f}h

META MENSAL
-----------
DF Meta: {DF_meta*100:.2f}%
UF Meta: {UF_meta*100:.2f}%
Ao Meta: {Ao_meta*100:.2f}%
Horas Produção Meta: {horas_prod_meta:.0f}h

GAP DE DESEMPENHO
-----------------
Gap Ao: {gap*100:.2f}% ({gap_analise['gap_percentual']:.1f}% relativo)
Gap Horas: {delta_horas:+.0f}h/mês

RECOMENDAÇÕES PARA ATINGIR META
--------------------------------
Opção 1 - Melhorar MTBF:
  MTBF Necessário: {gap_analise['MTBF_necessario']:.0f}h
  Melhoria: +{gap_analise['melhoria_MTBF_percentual']:.1f}%

Opção 2 - Melhorar MTTR:
  MTTR Máximo: {gap_analise['MTTR_maximo']:.1f}h
  Redução: -{gap_analise['melhoria_MTTR_percentual']:.1f}%

PLANO DE MANUTENÇÃO PREVENTIVA
-------------------------------
Intervalo PM Ótimo: {resultado_pm['T_otimo']:.0f}h operadas ({T_cal/24:.1f} dias calendário)
Frequência: {frequencia_pm_mes:.2f} PMs/mês ({frequencia_pm_mes*12:.1f} PMs/ano)
Ao no Ponto Ótimo: {resultado_pm['disponibilidade']*100:.2f}%
Confiabilidade: {resultado_pm['confiabilidade']*100:.2f}%

==========================================
Fim do Relatório
        """
        
        st.download_button(
            label="📄 Download Relatório (TXT)",
            data=relatorio,
            file_name="relatorio_disponibilidade.txt",
            mime="text/plain"
        )
    
    # ==================== RODAPÉ ====================
    
    st.divider()
    st.markdown("""
    **Sobre esta ferramenta v4.0:**
    
    Calculadora simplificada focada em **comparação entre meta e realizado**.
    
    **Entradas necessárias:**
    - 🎯 Meta mensal: DF e UF desejados
    - 📊 Desempenho atual: MTBF e MTTR do mês anterior
    
    **Saídas fornecidas:**
    - ✅ Análise completa de disponibilidade (Ai, Aa, Ao)
    - ✅ Gap e recomendações de melhoria
    - ✅ Intervalo ótimo de manutenção preventiva
    - ✅ Matriz interativa MTBF vs MTTR
    - ✅ Visualizações comparativas
    
    **Fórmulas:**

    
    $$A_i = \\frac{MTBF}{MTBF + MTTR}$$

    
    $$A_a = A_i \\times DF$$

    
    $$A_o = A_i \\times DF \\times UF$$
    
    **Referências:**
    - IEC 60300-3-1: Dependability management
    - MIL-HDBK-338B: Electronic Reliability Design Handbook
    - ISO 14224: Petroleum and natural gas industries - Collection and exchange of reliability and maintenance data
    """)

if __name__ == "__main__":
    main()
