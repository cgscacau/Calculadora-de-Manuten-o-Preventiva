"""
Calculadora de Disponibilidade Operacional - Meta vs Realizado
Versão: 5.0.0 (Versão Final Completa)
Autor: Sistema de Engenharia de Confiabilidade
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Calculadora de Disponibilidade Operacional",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSTANTES ====================
HORAS_POR_MES = 730.0
DIAS_POR_MES = 30.44

# ==================== FUNÇÕES DE CÁLCULO DE DISPONIBILIDADE ====================

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

def calcular_disponibilidade_operacional(Ai: float, DF: float, UF: float) -> float:
    """
    Calcula disponibilidade operacional (real, considerando utilização).
    Ao = Ai × DF × UF
    """
    return Ai * DF * UF

def calcular_horas_producao(Ao: float, horas_mes: float = HORAS_POR_MES) -> float:
    """Calcula horas disponíveis para produção no mês."""
    return Ao * horas_mes

# ==================== FUNÇÕES DE CÁLCULO REVERSO ====================

def calcular_MTBF_necessario(MTTR: float, DF: float, UF: float, Ao_meta: float) -> float:
    """
    Calcula MTBF necessário para atingir Ao_meta.
    
    Derivação:
    Ao = (MTBF/(MTBF+MTTR)) × DF × UF
    Resolvendo para MTBF:
    MTBF = (Ao × MTTR) / (DF × UF - Ao)
    """
    denominador = DF * UF - Ao_meta
    if denominador <= 0:
        return float('inf')
    return (Ao_meta * MTTR) / denominador

def calcular_MTTR_maximo(MTBF: float, DF: float, UF: float, Ao_meta: float) -> float:
    """
    Calcula MTTR máximo permitido para atingir Ao_meta.
    
    Derivação:
    MTTR = MTBF × ((DF × UF - Ao) / Ao)
    """
    if Ao_meta <= 0:
        return 0.0
    
    numerador = DF * UF - Ao_meta
    if numerador < 0:
        return -1.0  # Indica que é impossível
    
    return MTBF * (numerador / Ao_meta)

def calcular_DF_necessario(MTBF: float, MTTR: float, UF: float, Ao_meta: float) -> float:
    """
    Calcula DF necessário para atingir Ao_meta.
    
    Derivação:
    DF = Ao / (Ai × UF)
    """
    Ai = calcular_disponibilidade_intrinseca(MTBF, MTTR)
    denominador = Ai * UF
    
    if denominador <= 0:
        return float('inf')
    
    df_calculado = Ao_meta / denominador
    return df_calculado if df_calculado <= 1.0 else float('inf')

# ==================== ANÁLISE DE GAP ====================

def calcular_gap_analise(
    Ao_atual: float, 
    Ao_meta: float, 
    MTBF_atual: float, 
    MTTR_atual: float, 
    DF_atual: float, 
    DF_meta: float, 
    UF_meta: float
) -> Dict:
    """
    Analisa o gap entre situação atual e meta.
    Retorna recomendações de melhoria validadas.
    """
    # Gap percentual
    gap_percentual = ((Ao_meta - Ao_atual) / Ao_atual * 100) if Ao_atual > 0 else 0
    
    # Calcular valores necessários
    MTBF_necessario = calcular_MTBF_necessario(MTTR_atual, DF_meta, UF_meta, Ao_meta)
    MTTR_maximo = calcular_MTTR_maximo(MTBF_atual, DF_meta, UF_meta, Ao_meta)
    DF_necessario = calcular_DF_necessario(MTBF_atual, MTTR_atual, UF_meta, Ao_meta)
    
    # Validar e calcular melhorias percentuais
    mtbf_viavel = MTBF_necessario != float('inf') and MTBF_necessario > MTBF_atual
    if mtbf_viavel:
        melhoria_MTBF_percentual = ((MTBF_necessario - MTBF_atual) / MTBF_atual * 100)
    else:
        melhoria_MTBF_percentual = None
    
    mttr_viavel = MTTR_maximo >= 0 and MTTR_maximo < MTTR_atual
    if mttr_viavel:
        melhoria_MTTR_percentual = ((MTTR_atual - MTTR_maximo) / MTTR_atual * 100)
    else:
        melhoria_MTTR_percentual = None
    
    df_viavel = DF_necessario != float('inf') and DF_necessario > DF_atual and DF_necessario <= 1.0
    if df_viavel:
        melhoria_DF_percentual = ((DF_necessario - DF_atual) / DF_atual * 100)
    else:
        melhoria_DF_percentual = None
    
    return {
        'gap_percentual': gap_percentual,
        'MTBF_necessario': MTBF_necessario if mtbf_viavel else None,
        'MTTR_maximo': MTTR_maximo if mttr_viavel else None,
        'DF_necessario': DF_necessario if df_viavel else None,
        'melhoria_MTBF_percentual': melhoria_MTBF_percentual,
        'melhoria_MTTR_percentual': melhoria_MTTR_percentual,
        'melhoria_DF_percentual': melhoria_DF_percentual,
        'mtbf_viavel': mtbf_viavel,
        'mttr_viavel': mttr_viavel,
        'df_viavel': df_viavel,
        'atingivel': mtbf_viavel or mttr_viavel or df_viavel
    }

# ==================== CÁLCULOS OPERACIONAIS ====================

def calcular_numero_falhas(MTBF: float, horas_operadas: float) -> float:
    """Número esperado de falhas no período."""
    return horas_operadas / MTBF if MTBF > 0 else 0

def calcular_tempo_reparo_total(MTTR: float, num_falhas: float) -> float:
    """Tempo total em reparo no período."""
    return MTTR * num_falhas

# ==================== MODELO DE DEGRADAÇÃO ====================

def taxa_falha_degradacao(t: float, lambda_base: float, beta_desgaste: float, t_inicio_desgaste: float) -> float:
    """
    Taxa de falha com degradação progressiva.
    
    λ(t) = λ_base                                    se t ≤ t_inicio
    λ(t) = λ_base × (1 + ((t-t_inicio)/t_inicio)^β)  se t > t_inicio
    """
    if t <= t_inicio_desgaste:
        return lambda_base
    else:
        t_desgaste = t - t_inicio_desgaste
        return lambda_base * (1 + (t_desgaste / t_inicio_desgaste) ** beta_desgaste)

def confiabilidade_degradacao(t: float, lambda_base: float, beta_desgaste: float, t_inicio_desgaste: float, n_pontos: int = 500) -> float:
    """
    Confiabilidade considerando degradação.
    R(t) = exp(-∫[0,t] λ(τ) dτ)
    """
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
    t_inicio_desgaste: Optional[float] = None,
    beta_desgaste: float = 2.5
) -> Dict:
    """
    Encontra intervalo ótimo de PM baseado em disponibilidade operacional mínima.
    """
    if t_inicio_desgaste is None:
        t_inicio_desgaste = MTBF * 0.7
    
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

# ==================== GRÁFICOS ====================

def criar_grafico_comparativo(
    Ao_atual: float, 
    Ao_meta: float, 
    Ai_atual: float, 
    Ai_meta: float,
    DF_atual: float, 
    DF_meta: float, 
    UF_atual: float, 
    UF_meta: float
) -> go.Figure:
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
        go.Bar(
            x=['Atual', 'Meta'], 
            y=[Ao_atual*100, Ao_meta*100],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{Ao_atual*100:.1f}%', f'{Ao_meta*100:.1f}%'],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Ai
    fig.add_trace(
        go.Bar(
            x=['Atual', 'Meta'], 
            y=[Ai_atual*100, Ai_meta*100],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{Ai_atual*100:.1f}%', f'{Ai_meta*100:.1f}%'],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # DF
    fig.add_trace(
        go.Bar(
            x=['Atual', 'Meta'], 
            y=[DF_atual*100, DF_meta*100],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{DF_atual*100:.1f}%', f'{DF_meta*100:.1f}%'],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # UF
    fig.add_trace(
        go.Bar(
            x=['Atual', 'Meta'], 
            y=[UF_atual*100, UF_meta*100],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{UF_atual*100:.1f}%', f'{UF_meta*100:.1f}%'],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_yaxes(title_text="%", range=[0, 105])
    fig.update_layout(height=600, showlegend=False, title_text="Comparação: Atual vs Meta")
    
    return fig

def criar_grafico_degradacao(resultado: Dict, T_otimo: float, Ao_minima: float) -> go.Figure:
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
        go.Scatter(
            x=t_vals, 
            y=resultado['disponibilidades']*100, 
            mode='lines',
            name='Ao(t)', 
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    fig.add_hline(
        y=Ao_minima*100, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Ao mínima: {Ao_minima*100:.1f}%", 
        row=1, col=1
    )
    
    # Subplot 2: Confiabilidade
    fig.add_trace(
        go.Scatter(
            x=t_vals, 
            y=resultado['confiabilidades']*100, 
            mode='lines',
            name='R(t)', 
            line=dict(color='green', width=3)
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
            line=dict(color='red', width=3)
        ),
        row=2, col=1
    )
    
    # Subplot 4: Evolução combinada normalizada
    Ao_norm = (resultado['disponibilidades'] - resultado['disponibilidades'].min()) / \
              (resultado['disponibilidades'].max() - resultado['disponibilidades'].min()) * 100
    R_norm = (resultado['confiabilidades'] - resultado['confiabilidades'].min()) / \
             (resultado['confiabilidades'].max() - resultado['confiabilidades'].min()) * 100
    
    fig.add_trace(
        go.Scatter(
            x=t_vals, 
            y=Ao_norm, 
            mode='lines', 
            name='Ao normalizado',
            line=dict(color='blue', width=2)
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=t_vals, 
            y=R_norm, 
            mode='lines', 
            name='R normalizado',
            line=dict(color='green', width=2, dash='dash')
        ),
        row=2, col=2
    )
    
    # Linha vertical no ponto ótimo
    for row in [1, 2]:
        for col in [1, 2]:
            if col == 1 or (row == 2 and col == 2):
                fig.add_vline(
                    x=T_otimo, 
                    line_dash="dash", 
                    line_color="purple",
                    opacity=0.7, 
                    annotation_text=f"PM: {T_otimo:.0f}h",
                    row=row, col=col
                )
    
    fig.update_xaxes(title_text="Horas Operadas")
    fig.update_yaxes(title_text="Ao (%)", row=1, col=1)
    fig.update_yaxes(title_text="R(t) (%)", row=1, col=2)
    fig.update_yaxes(title_text="λ(t)", row=2, col=1)
    fig.update_yaxes(title_text="Normalizado (%)", row=2, col=2)
    
    fig.update_layout(
        height=700, 
        showlegend=True, 
        title_text="Análise de Degradação - Intervalo Ótimo de PM"
    )
    
    return fig

def criar_matriz_disponibilidade(
    parametro_fixo: str,
    valor_fixo: float,
    UF: float,
    MTBF_atual: float,
    MTTR_atual: float,
    DF_atual: float,
    valores_meta: Dict,
    ranges: Dict,
    n_pontos: int = 30
) -> go.Figure:
    """
    Cria matriz de disponibilidade correlacionando MTBF, MTTR e DF.
    """
    
    if parametro_fixo == 'MTBF':
        # Fixo MTBF, varia MTTR e DF
        param1_vals = np.linspace(ranges['mttr_min'], ranges['mttr_max'], n_pontos)
        param2_vals = np.linspace(ranges['df_min'], ranges['df_max'], n_pontos)
        param1_name = "MTTR (horas)"
        param2_name = "DF (%)"
        
        matriz = np.zeros((n_pontos, n_pontos))
        for i, mttr in enumerate(param1_vals):
            for j, df in enumerate(param2_vals):
                Ai = calcular_disponibilidade_intrinseca(valor_fixo, mttr)
                Ao = calcular_disponibilidade_operacional(Ai, df, UF)
                matriz[i, j] = Ao * 100
        
        x_atual = DF_atual
        y_atual = MTTR_atual
        x_meta_mttr = DF_atual
        y_meta_mttr = valores_meta.get('MTTR_maximo')
        x_meta_df = valores_meta.get('DF_necessario')
        y_meta_df = MTTR_atual
        
        param2_vals_display = param2_vals * 100
        x_atual_display = x_atual * 100
        x_meta_mttr_display = x_meta_mttr * 100 if x_meta_mttr else None
        x_meta_df_display = x_meta_df * 100 if x_meta_df else None
        
    elif parametro_fixo == 'MTTR':
        # Fixo MTTR, varia MTBF e DF
        param1_vals = np.linspace(ranges['mtbf_min'], ranges['mtbf_max'], n_pontos)
        param2_vals = np.linspace(ranges['df_min'], ranges['df_max'], n_pontos)
        param1_name = "MTBF (horas)"
        param2_name = "DF (%)"
        
        matriz = np.zeros((n_pontos, n_pontos))
        for i, mtbf in enumerate(param1_vals):
            for j, df in enumerate(param2_vals):
                Ai = calcular_disponibilidade_intrinseca(mtbf, valor_fixo)
                Ao = calcular_disponibilidade_operacional(Ai, df, UF)
                matriz[i, j] = Ao * 100
        
        x_atual = DF_atual
        y_atual = MTBF_atual
        x_meta_mtbf = DF_atual
        y_meta_mtbf = valores_meta.get('MTBF_necessario')
        x_meta_df = valores_meta.get('DF_necessario')
        y_meta_df = MTBF_atual
        
        param2_vals_display = param2_vals * 100
        x_atual_display = x_atual * 100
        x_meta_mtbf_display = x_meta_mtbf * 100 if x_meta_mtbf else None
        x_meta_df_display = x_meta_df * 100 if x_meta_df else None
        
    else:  # DF fixo
        # Fixo DF, varia MTBF e MTTR
        param1_vals = np.linspace(ranges['mtbf_min'], ranges['mtbf_max'], n_pontos)
        param2_vals = np.linspace(ranges['mttr_min'], ranges['mttr_max'], n_pontos)
        param1_name = "MTBF (horas)"
        param2_name = "MTTR (horas)"
        
        matriz = np.zeros((n_pontos, n_pontos))
        for i, mtbf in enumerate(param1_vals):
            for j, mttr in enumerate(param2_vals):
                Ai = calcular_disponibilidade_intrinseca(mtbf, mttr)
                Ao = calcular_disponibilidade_operacional(Ai, valor_fixo, UF)
                matriz[i, j] = Ao * 100
        
        x_atual = MTTR_atual
        y_atual = MTBF_atual
        x_meta_mttr = valores_meta.get('MTTR_maximo')
        y_meta_mttr = MTBF_atual
        x_meta_mtbf = MTTR_atual
        y_meta_mtbf = valores_meta.get('MTBF_necessario')
        
        param2_vals_display = param2_vals
        x_atual_display = x_atual
        x_meta_mttr_display = x_meta_mttr
        x_meta_mtbf_display = x_meta_mtbf
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar heatmap
    fig.add_trace(go.Heatmap(
        z=matriz,
        x=param2_vals_display,
        y=param1_vals,
        colorscale='RdYlGn',
        colorbar=dict(title="Ao (%)"),
        hovertemplate=f'{param2_name}: %{{x:.1f}}<br>{param1_name}: %{{y:.1f}}<br>Ao: %{{z:.1f}}%<extra></extra>',
        name='Disponibilidade'
    ))
    
    # Marcador: Posição ATUAL
    fig.add_trace(go.Scatter(
        x=[x_atual_display],
        y=[y_atual],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='x', line=dict(width=3, color='white')),
        text=['ATUAL'],
        textposition='top center',
        textfont=dict(size=12, color='white', family='Arial Black'),
        name='Posição Atual',
        hovertemplate=f'<b>ATUAL</b><br>{param2_name}: {x_atual_display:.1f}<br>{param1_name}: {y_atual:.1f}<extra></extra>'
    ))
    
    # Adicionar marcadores de meta baseados no parâmetro fixo
    if parametro_fixo == 'MTBF':
        # Meta melhorando MTTR
        if y_meta_mttr is not None and ranges['mttr_min'] <= y_meta_mttr <= ranges['mttr_max']:
            fig.add_trace(go.Scatter(
                x=[x_meta_mttr_display],
                y=[y_meta_mttr],
                mode='markers+text',
                marker=dict(size=18, color='yellow', symbol='diamond', line=dict(width=2, color='black')),
                text=['META MTTR'],
                textposition='bottom center',
                textfont=dict(size=10, color='black', family='Arial Black'),
                name='Meta MTTR',
                hovertemplate=f'<b>META MTTR</b><br>DF: {x_meta_mttr_display:.1f}%<br>MTTR: {y_meta_mttr:.1f}h<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[x_atual_display, x_meta_mttr_display],
                y=[y_atual, y_meta_mttr],
                mode='lines',
                line=dict(color='yellow', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Meta melhorando DF
        if x_meta_df_display is not None and ranges['df_min'] <= (x_meta_df_display/100) <= ranges['df_max']:
            fig.add_trace(go.Scatter(
                x=[x_meta_df_display],
                y=[y_meta_df],
                mode='markers+text',
                marker=dict(size=18, color='lime', symbol='square', line=dict(width=2, color='black')),
                text=['META DF'],
                textposition='top right',
                textfont=dict(size=10, color='black', family='Arial Black'),
                name='Meta DF',
                hovertemplate=f'<b>META DF</b><br>DF: {x_meta_df_display:.1f}%<br>MTTR: {y_meta_df:.1f}h<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[x_atual_display, x_meta_df_display],
                y=[y_atual, y_meta_df],
                mode='lines',
                line=dict(color='lime', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    elif parametro_fixo == 'MTTR':
        # Meta melhorando MTBF
        if y_meta_mtbf is not None and ranges['mtbf_min'] <= y_meta_mtbf <= ranges['mtbf_max']:
            fig.add_trace(go.Scatter(
                x=[x_meta_mtbf_display],
                y=[y_meta_mtbf],
                mode='markers+text',
                marker=dict(size=18, color='cyan', symbol='star', line=dict(width=2, color='white')),
                text=['META MTBF'],
                textposition='bottom center',
                textfont=dict(size=10, color='white', family='Arial Black'),
                name='Meta MTBF',
                hovertemplate=f'<b>META MTBF</b><br>DF: {x_meta_mtbf_display:.1f}%<br>MTBF: {y_meta_mtbf:.0f}h<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[x_atual_display, x_meta_mtbf_display],
                y=[y_atual, y_meta_mtbf],
                mode='lines',
                line=dict(color='cyan', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Meta melhorando DF
        if x_meta_df_display is not None and ranges['df_min'] <= (x_meta_df_display/100) <= ranges['df_max']:
            fig.add_trace(go.Scatter(
                x=[x_meta_df_display],
                y=[y_meta_df],
                mode='markers+text',
                marker=dict(size=18, color='lime', symbol='square', line=dict(width=2, color='black')),
                text=['META DF'],
                textposition='top right',
                textfont=dict(size=10, color='black', family='Arial Black'),
                name='Meta DF',
                hovertemplate=f'<b>META DF</b><br>DF: {x_meta_df_display:.1f}%<br>MTBF: {y_meta_df:.0f}h<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[x_atual_display, x_meta_df_display],
                y=[y_atual, y_meta_df],
                mode='lines',
                line=dict(color='lime', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    else:  # DF fixo
        # Meta melhorando MTBF
        if y_meta_mtbf is not None and ranges['mtbf_min'] <= y_meta_mtbf <= ranges['mtbf_max']:
            fig.add_trace(go.Scatter(
                x=[x_meta_mtbf_display],
                y=[y_meta_mtbf],
                mode='markers+text',
                marker=dict(size=18, color='cyan', symbol='star', line=dict(width=2, color='white')),
                text=['META MTBF'],
                textposition='bottom center',
                textfont=dict(size=10, color='white', family='Arial Black'),
                name='Meta MTBF',
                hovertemplate=f'<b>META MTBF</b><br>MTTR: {x_meta_mtbf_display:.1f}h<br>MTBF: {y_meta_mtbf:.0f}h<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[x_atual_display, x_meta_mtbf_display],
                y=[y_atual, y_meta_mtbf],
                mode='lines',
                line=dict(color='cyan', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Meta melhorando MTTR
        if x_meta_mttr_display is not None and ranges['mttr_min'] <= x_meta_mttr_display <= ranges['mttr_max']:
            fig.add_trace(go.Scatter(
                x=[x_meta_mttr_display],
                y=[y_meta_mttr],
                mode='markers+text',
                marker=dict(size=18, color='yellow', symbol='diamond', line=dict(width=2, color='black')),
                text=['META MTTR'],
                textposition='top right',
                textfont=dict(size=10, color='black', family='Arial Black'),
                name='Meta MTTR',
                hovertemplate=f'<b>META MTTR</b><br>MTTR: {x_meta_mttr_display:.1f}h<br>MTBF: {y_meta_mttr:.0f}h<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[x_atual_display, x_meta_mttr_display],
                y=[y_atual, y_meta_mttr],
                mode='lines',
                line=dict(color='yellow', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Layout
    titulo = f"Matriz de Disponibilidade - {parametro_fixo} Fixo = "
    if parametro_fixo == 'DF':
        titulo += f"{valor_fixo*100:.1f}%"
    else:
        titulo += f"{valor_fixo:.1f}h"
    titulo += f" | UF = {UF*100:.0f}%"
    
    fig.update_layout(
        title=titulo,
        xaxis_title=param2_name,
        yaxis_title=param1_name,
        height=700,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
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
    - ✅ Recomendações de melhoria (MTBF, MTTR e DF)
    - ✅ Intervalo ótimo de manutenção preventiva
    - ✅ **Matriz configurável: MTBF × MTTR × DF**
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
            help="Meta de tempo disponível para operação"
        ) / 100
        
        UF_meta = st.slider(
            "UF Meta - Fator de Utilização (%)",
            min_value=70.0,
            max_value=100.0,
            value=90.0,
            step=0.5,
            help="Meta de utilização do tempo disponível"
        ) / 100
        
        Ao_meta = DF_meta * UF_meta
        
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
            value=80.0,
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
    
    # Situação Meta
    Ai_meta = 1.0
    Aa_meta = calcular_disponibilidade_alcancada(Ai_meta, DF_meta)
    horas_prod_meta = calcular_horas_producao(Ao_meta)
    
    # Análise de Gap
    gap_analise = calcular_gap_analise(Ao_atual, Ao_meta, MTBF_atual, MTTR_atual, DF_atual, DF_meta, UF_meta)
    
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
            
            # Criar tabela de recomendações
            recomendacoes_lista = []
            
            if gap_analise['mtbf_viavel']:
                recomendacoes_lista.append({
                    'Estratégia': 'Opção 1: Melhorar MTBF',
                    'Ação Necessária': f"MTBF ≥ {gap_analise['MTBF_necessario']:.0f}h (+{gap_analise['melhoria_MTBF_percentual']:.1f}%)"
                })
            else:
                recomendacoes_lista.append({
                    'Estratégia': 'Opção 1: Melhorar MTBF',
                    'Ação Necessária': "Não aplicável"
                })
            
            if gap_analise['mttr_viavel']:
                recomendacoes_lista.append({
                    'Estratégia': 'Opção 2: Melhorar MTTR',
                    'Ação Necessária': f"MTTR ≤ {gap_analise['MTTR_maximo']:.1f}h (-{gap_analise['melhoria_MTTR_percentual']:.1f}%)"
                })
            else:
                recomendacoes_lista.append({
                    'Estratégia': 'Opção 2: Melhorar MTTR',
                    'Ação Necessária': "Não aplicável"
                })
            
            if gap_analise['df_viavel']:
                recomendacoes_lista.append({
                    'Estratégia': 'Opção 3: Melhorar DF',
                    'Ação Necessária': f"DF ≥ {gap_analise['DF_necessario']*100:.1f}% (+{gap_analise['melhoria_DF_percentual']:.1f}%)"
                })
            else:
                recomendacoes_lista.append({
                    'Estratégia': 'Opção 3: Melhorar DF',
                    'Ação Necessária': "Não aplicável"
                })
            
            recomendacoes = pd.DataFrame(recomendacoes_lista)
            st.dataframe(recomendacoes, use_container_width=True, hide_index=True)
            
            # Interpretação detalhada
            interpretacao_texto = f"""**Interpretação:**\n\nPara atingir **Ao = {Ao_meta*100:.2f}%**, você pode:\n"""
            
            opcoes_texto = []
            
            if gap_analise['mtbf_viavel']:
                opcoes_texto.append(f"""
1. **Aumentar MTBF** de {MTBF_atual:.0f}h para {gap_analise['MTBF_necessario']:.0f}h
   - Melhoria necessária: **{gap_analise['melhoria_MTBF_percentual']:.1f}%**
   - Ações: Melhorar confiabilidade, manutenção preditiva, substituir componentes críticos""")
            
            if gap_analise['mttr_viavel']:
                opcoes_texto.append(f"""
2. **Reduzir MTTR** de {MTTR_atual:.1f}h para {gap_analise['MTTR_maximo']:.1f}h
   - Redução necessária: **{gap_analise['melhoria_MTTR_percentual']:.1f}%**
   - Ações: Treinamento, peças em estoque, procedimentos otimizados""")
            
            if gap_analise['df_viavel']:
                opcoes_texto.append(f"""
3. **Aumentar DF** de {DF_atual*100:.1f}% para {gap_analise['DF_necessario']*100:.1f}%
   - Melhoria necessária: **{gap_analise['melhoria_DF_percentual']:.1f}%**
   - Ações: Reduzir paradas programadas, otimizar setup, melhorar planejamento""")
            
            if opcoes_texto:
                interpretacao_texto += "\n".join(opcoes_texto)
                interpretacao_texto += "\n\n💡 **Recomendação:** Combine melhorias nos parâmetros viáveis para resultados sustentáveis."
            else:
                interpretacao_texto += "\n⚠️ **Atenção:** Nenhuma opção individual é viável. Considere combinar melhorias ou revisar a meta."
            
            st.info(interpretacao_texto)
        else:
            st.error("⚠️ Meta muito ambiciosa!")
            st.warning(f"""
            A meta de **Ao = {Ao_meta*100:.2f}%** não é atingível individualmente.
            
            **Ao máximo teórico** com DF={DF_meta*100:.0f}% e UF={UF_meta*100:.0f}%: **{(DF_meta * UF_meta)*100:.2f}%**
            
            **Sugestões:**
            1. Revisar metas de DF e UF (aumentar)
            2. Aceitar meta de Ao mais realista
            3. Combinar melhorias em MTBF, MTTR e DF simultaneamente
            """)
    
    st.divider()
    
    # ==================== GRÁFICOS ====================
    
    st.header("📊 VISUALIZAÇÕES")
    
    tab1, tab2, tab3 = st.tabs(["Comparativo", "Degradação e PM", "Matriz: MTBF × MTTR × DF"])
    
    with tab1:
        st.subheader("Comparação: Atual vs Meta")
        
        fig_comp = criar_grafico_comparativo(
            Ao_atual, Ao_meta, Ai_atual, Ai_meta,
            DF_atual, DF_meta, UF_atual, UF_meta
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
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
                step=1.0
            ) / 100
        
        with col_config2:
            beta_desgaste = st.slider(
                "Intensidade de Degradação (β)",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                step=0.1
            )
        
        resultado_pm = encontrar_intervalo_PM_otimo(
            MTBF=MTBF_atual,
            MTTR=MTTR_atual,
            DF=DF_meta,
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
            
            2. **Início da Degradação** ({resultado_pm['t_inicio_desgaste']:.0f}h):
               - Componentes começam a desgastar
               - Taxa de falha aumenta gradualmente
            
            3. **Ponto Ótimo de PM** ({resultado_pm['T_otimo']:.0f}h operadas):
               - Disponibilidade: {resultado_pm['disponibilidade']*100:.2f}%
               - Confiabilidade: {resultado_pm['confiabilidade']*100:.2f}%
               - **Momento ideal para intervenção preventiva**
            
            **Recomendação:**
            - Fazer PM a cada **{T_cal/24:.1f} dias** (calendário)
            - Ou a cada **{resultado_pm['T_otimo']:.0f} horas** de operação
            - Frequência: **{frequencia_pm_mes:.2f} PMs/mês** ou **{frequencia_pm_mes*12:.1f} PMs/ano**
            """)
    
    with tab3:
        st.subheader("Matriz de Disponibilidade: MTBF × MTTR × DF")
        
        st.markdown("""
        **Configure a matriz escolhendo qual parâmetro fixar.**
        
        Marcadores:
        - 🔴 **X Vermelho**: Posição ATUAL
        - 🔵 **Estrela Azul**: Meta melhorando MTBF
        - 🟡 **Diamante Amarelo**: Meta melhorando MTTR
        - 🟢 **Quadrado Verde**: Meta melhorando DF
        """)
        
        st.divider()
        
        col_config1, col_config2 = st.columns([1, 2])
        
        with col_config1:
            st.markdown("**Configuração da Matriz:**")
            
            parametro_fixo = st.selectbox(
                "Parâmetro Fixo:",
                ["DF", "MTBF", "MTTR"],
                help="Escolha qual parâmetro manter constante"
            )
            
            if parametro_fixo == "MTBF":
                valor_fixo = st.number_input(
                    "Valor do MTBF Fixo (h)",
                    min_value=10.0,
                    value=MTBF_atual,
                    step=10.0
                )
                
                st.markdown("**Ranges:**")
                mttr_min = st.number_input("MTTR Mín (h)", min_value=0.5, value=0.0, step=0.5)
                mttr_max = st.number_input("MTTR Máx (h)", min_value=0.5, value=20.0, step=0.5)
                df_min = st.number_input("DF Mín (%)", min_value=50.0, value=70.0, step=1.0) / 100
                df_max = st.number_input("DF Máx (%)", min_value=50.0, value=100.0, step=1.0) / 100
                
                ranges = {
                    'mttr_min': mttr_min,
                    'mttr_max': mttr_max,
                    'df_min': df_min,
                    'df_max': df_max
                }
                
            elif parametro_fixo == "MTTR":
                valor_fixo = st.number_input(
                    "Valor do MTTR Fixo (h)",
                    min_value=0.1,
                    value=MTTR_atual,
                    step=0.5
                )
                
                st.markdown("**Ranges:**")
                mtbf_min = st.number_input("MTBF Mín (h)", min_value=10.0, value=100.0, step=10.0)
                mtbf_max = st.number_input("MTBF Máx (h)", min_value=10.0, value=500.0, step=10.0)
                df_min = st.number_input("DF Mín (%)", min_value=50.0, value=70.0, step=1.0) / 100
                df_max = st.number_input("DF Máx (%)", min_value=50.0, value=100.0, step=1.0) / 100
                
                ranges = {
                    'mtbf_min': mtbf_min,
                    'mtbf_max': mtbf_max,
                    'df_min': df_min,
                    'df_max': df_max
                }
                
            else:  # DF fixo
                valor_fixo = st.slider(
                    "Valor do DF Fixo (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=DF_atual * 100,
                    step=0.5
                ) / 100
                
                st.markdown("**Ranges:**")
                mtbf_min = st.number_input("MTBF Mín (h)", min_value=10.0, value=100.0, step=10.0)
                mtbf_max = st.number_input("MTBF Máx (h)", min_value=10.0, value=500.0, step=10.0)
                mttr_min = st.number_input("MTTR Mín (h)", min_value=0.5, value=0.0, step=0.5)
                mttr_max = st.number_input("MTTR Máx (h)", min_value=0.5, value=20.0, step=0.5)
                
                ranges = {
                    'mtbf_min': mtbf_min,
                    'mtbf_max': mtbf_max,
                    'mttr_min': mttr_min,
                    'mttr_max': mttr_max
                }
            
            resolucao = st.slider(
                "Resolução",
                min_value=15,
                max_value=50,
                value=30,
                step=5
            )
        
        with col_config2:
            valores_meta = {
                'MTBF_necessario': gap_analise['MTBF_necessario'],
                'MTTR_maximo': gap_analise['MTTR_maximo'],
                'DF_necessario': gap_analise['DF_necessario']
            }
            
            fig_matriz = criar_matriz_disponibilidade(
                parametro_fixo=parametro_fixo,
                valor_fixo=valor_fixo,
                UF=UF_meta,
                MTBF_atual=MTBF_atual,
                MTTR_atual=MTTR_atual,
                DF_atual=DF_atual,
                valores_meta=valores_meta,
                ranges=ranges,
                n_pontos=resolucao
            )
            
            st.plotly_chart(fig_matriz, use_container_width=True)
            
            st.info(f"""
            **Posição Atual:**
            - MTBF = {MTBF_atual:.0f}h
            - MTTR = {MTTR_atual:.1f}h
            - DF = {DF_atual*100:.1f}%
            - Ao = {Ao_atual*100:.2f}%
            
            **Para meta (Ao = {Ao_meta*100:.2f}%):**
            - MTBF: {gap_analise['MTBF_necessario']:.0f}h (+{gap_analise['melhoria_MTBF_percentual']:.1f}%)" if gap_analise['mtbf_viavel'] else "- MTBF: Não aplicável
            - MTTR: {gap_analise['MTTR_maximo']:.1f}h (-{gap_analise['melhoria_MTTR_percentual']:.1f}%)" if gap_analise['mttr_viavel'] else "- MTTR: Não aplicável
            - DF: {gap_analise['DF_necessario']*100:.1f}% (+{gap_analise['melhoria_DF_percentual']:.1f}%)" if gap_analise['df_viavel'] else "- DF: Não aplicável
            """)
    
    # ==================== EXPORT ====================
    
    st.divider()
    st.header("💾 Exportar Resultados")
    
    df_export = pd.DataFrame({
        'Categoria': [
            'MTBF Atual', 'MTTR Atual', 'DF Atual', 'UF Atual',
            'Ai Atual', 'Aa Atual', 'Ao Atual', 'Horas Produção Atual',
            'DF Meta', 'UF Meta', 'Ao Meta', 'Horas Produção Meta',
            'Gap Ao (%)', 'Gap Horas',
            'MTBF Necessário', 'MTTR Máximo', 'DF Necessário',
            'Melhoria MTBF (%)', 'Melhoria MTTR (%)', 'Melhoria DF (%)',
            'Falhas Esperadas/Mês', 'Tempo Reparo Total/Mês',
            'Intervalo PM Ótimo (h)', 'Intervalo PM (dias)',
            'Frequência PM/Mês', 'Frequência PM/Ano'
        ],
        'Valor': [
            f"{MTBF_atual:.1f}h", f"{MTTR_atual:.1f}h", f"{DF_atual*100:.2f}%", f"{UF_atual*100:.2f}%",
            f"{Ai_atual*100:.2f}%", f"{Aa_atual*100:.2f}%", f"{Ao_atual*100:.2f}%", f"{horas_prod_atual:.0f}h",
            f"{DF_meta*100:.2f}%", f"{UF_meta*100:.2f}%", f"{Ao_meta*100:.2f}%", f"{horas_prod_meta:.0f}h",
            f"{gap*100:.2f}%", f"{delta_horas:.0f}h",
            f"{gap_analise['MTBF_necessario']:.0f}h" if gap_analise['mtbf_viavel'] else "N/A",
            f"{gap_analise['MTTR_maximo']:.1f}h" if gap_analise['mttr_viavel'] else "N/A",
            f"{gap_analise['DF_necessario']*100:.1f}%" if gap_analise['df_viavel'] else "N/A",
            f"{gap_analise['melhoria_MTBF_percentual']:.1f}%" if gap_analise['mtbf_viavel'] else "N/A",
            f"{gap_analise['melhoria_MTTR_percentual']:.1f}%" if gap_analise['mttr_viavel'] else "N/A",
            f"{gap_analise['melhoria_DF_percentual']:.1f}%" if gap_analise['df_viavel'] else "N/A",
            f"{num_falhas_atual:.2f}", f"{tempo_reparo_total_atual:.1f}h",
            f"{resultado_pm['T_otimo']:.0f}h", f"{T_cal/24:.1f} dias",
            f"{frequencia_pm_mes:.2f}", f"{frequencia_pm_mes*12:.1f}"
        ]
    })
    
    csv_export = df_export.to_csv(index=False, encoding='utf-8-sig')
    
    st.download_button(
        label="📥 Download Análise Completa (CSV)",
        data=csv_export,
        file_name="analise_disponibilidade_completa.csv",
        mime="text/csv"
    )
    
    # ==================== RODAPÉ ====================
    
    st.divider()
    st.markdown("""
    **Calculadora de Disponibilidade Operacional v5.0**
    
    Sistema completo para análise de disponibilidade e planejamento de manutenção preventiva.
    
    **Fórmulas:**

    
    $$A_i = \\frac{MTBF}{MTBF + MTTR}$$

    
    $$A_a = A_i \\times DF$$

    
    $$A_o = A_i \\times DF \\times UF$$
    
    **Referências:**
    - IEC 60300-3-1: Dependability management
    - MIL-HDBK-338B: Electronic Reliability Design Handbook
    - ISO 14224: Collection and exchange of reliability data
    """)

if __name__ == "__main__":
    main()
