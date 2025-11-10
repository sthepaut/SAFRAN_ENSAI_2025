import plotly.graph_objects as go
import numpy as np






# def plot_trajectory(trajectory):
#     """
#     trajectory : liste de listes [[v1, v2], [v1, v2], ...]
#     """
#     values = np.array(trajectory)
#     x = list(range(len(trajectory)))

#     fig = go.Figure()
    
#     fig.add_trace(go.Scatter(
#         x=x, y=values[:, 0],
#         mode='lines+markers',
#         name='Indicator 1',
#         line=dict(color='blue')
#     ))

#     fig.add_trace(go.Scatter(
#         x=x, y=values[:, 1],
#         mode='lines+markers',
#         name='Indicator 2',
#         line=dict(color='red')
#     ))

#     fig.update_layout(
#         title='Degradation Trajectory',
#         xaxis_title='Time Step',
#         yaxis_title='Indicator Value',
#         legend=dict(x=0.01, y=0.99),
#         template='plotly_white'
#     )

#     fig.show()


def plot_trajectory(trajectory):
    """
    Plot the degradation trajectory for compressor, turbine, and combustion chamber.

    Parameters
    ----------
    trajectory : list of lists
        Each element is [degrad_comp, degrad_turb, degrad_comb].
    """
    values = np.array(trajectory)
    x = list(range(len(trajectory)))

    fig = go.Figure()

    # Compressor degradation
    fig.add_trace(go.Scatter(
        x=x, y=values[:, 0],
        mode='lines+markers',
        name='Compressor (Comp)',
        line=dict(color='blue')
    ))

    # Turbine degradation
    fig.add_trace(go.Scatter(
        x=x, y=values[:, 1],
        mode='lines+markers',
        name='Turbine (Turb)',
        line=dict(color='red')
    ))

    # Combustion chamber degradation
    if values.shape[1] > 2:
        fig.add_trace(go.Scatter(
            x=x, y=values[:, 2],
            mode='lines+markers',
            name='Combustion (Comb)',
            line=dict(color='green')
        ))

    # Layout and styling
    fig.update_layout(
        
        title='Degradation Trajectory (Comp, Turb, Comb)',
        xaxis_title='Time Step',
        yaxis_title='Health Indicator Value',
        # legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        # Legend on the right side
        legend=dict(
            x=0.8,          # move to right
            y=0.99,           # vertical center
            traceorder='normal',
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
        ),
        margin=dict(r=120),  # add space on the right for legend

        template='plotly_white'
    )

    fig.show()

