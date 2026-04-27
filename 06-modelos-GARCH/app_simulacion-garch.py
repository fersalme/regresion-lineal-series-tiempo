import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from statsmodels.tsa.stattools import pacf

app_ui = ui.page_fluid(
    ui.panel_title("Simulador de Volatilidad GARCH(1,1)"),
    
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Parámetros del Modelo"),
            ui.input_slider("n", "Número de observaciones (n)", 100, 2000, 1000),
            ui.input_slider("omega", "Omega (Base)", 0.01, 1.0, 0.5, step=0.01),
            ui.input_slider("alpha", "Alpha (Reacción al shock)", 0.0, 0.9, 0.1, step=0.05),
            ui.input_slider("beta", "Beta (Persistencia)", 0.0, 0.9, 0.3, step=0.05),
            ui.hr(),
            ui.markdown(
                """
                **Nota:** Para estabilidad, se recomienda que:
                **α + β ≤ 1**
                """
            )
        ),
        ui.card(
            ui.card_header("Visualización de la Serie y Volatilidad"),
            output_widget("garch_plot"),
            full_screen=True
        ),
        ui.card(
            ui.card_header("PACF de Retornos al Cuadrado (Volatilidad)"),
            output_widget("pacf_plot")
        )
    )
)

def server(input, output, session):
    
    @reactive.calc
    def simulate_garch():
        # Obtenemos valores de los inputs
        n = input.n()
        omega = input.omega()
        alpha = input.alpha()
        beta = input.beta()
        
        # Inicialización
        series = [np.random.normal(), np.random.normal()]
        vols = [1.0, 1.0]
        
        # Simulación GARCH(1,1)
        for _ in range(n):
            # Formula: sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
            new_vol_sq = omega + alpha * (series[-1]**2) + beta * (vols[-1]**2)
            new_vol = np.sqrt(new_vol_sq)
            new_val = np.random.normal() * new_vol
            
            vols.append(new_vol)
            series.append(new_val)
        
        return pd.DataFrame({"Retornos": series, "Volatilidad": vols})

    @render_widget
    def garch_plot():
        df = simulate_garch()
        
        # Crear subplots (Serie, Volatilidad, Superposición)
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Retornos Simulados", "Volatilidad Estimada (Sigma)", "Superposición")
        )

        # 1. Retornos
        fig.add_trace(
            go.Scatter(y=df["Retornos"], name="Retornos", line=dict(color="#1f77b4", width=1)),
            row=1, col=1
        )

        # 2. Volatilidad
        fig.add_trace(
            go.Scatter(y=df["Volatilidad"], name="Volatilidad", line=dict(color="orange", width=2)),
            row=2, col=1
        )

        # 3. Superposición
        fig.add_trace(
            go.Scatter(y=df["Retornos"], name="Retornos", opacity=0.4, line=dict(color="#1f77b4")),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(y=df["Volatilidad"], name="Volatilidad", line=dict(color="orange", width=2)),
            row=3, col=1
        )

        fig.update_layout(height=800, showlegend=False, template="plotly_white")
        return fig

    @render_widget
    def pacf_plot():
        # Obtenemos los datos de la función reactiva que ya definimos
        df = simulate_garch()
        
        # Calculamos los retornos al cuadrado
        squared_returns = df["Retornos"]**2
        
        # Calculamos PACF (usualmente 20 a 40 rezagos)
        lags = 25
        pacf_values = pacf(squared_returns, nlags=lags)
        
        # Crear el gráfico con Plotly
        fig = go.Figure()

        # Añadir las barras del PACF
        fig.add_trace(go.Bar(
            x=list(range(lags + 1)), 
            y=pacf_values,
            name="PACF",
            marker_color="#004581"
        ))

        # Añadir intervalos de confianza (aprox. 1.96 / sqrt(n))
        conf_interval = 1.96 / np.sqrt(len(squared_returns))
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", opacity=0.5)

        fig.update_layout(
            title="Partial Autocorrelation (Squared Returns)",
            xaxis_title="Rezagos (Lags)",
            yaxis_title="Correlación Parcial",
            template="plotly_white",
            height=400
        )
        
        return fig

app = App(app_ui, server)
