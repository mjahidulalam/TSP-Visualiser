import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, State, Input
import dash_bootstrap_components as dbc
from solvers import Solver

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "Jahidul - TSP"
app.prevent_initial_callback=True
app.debug=False
app._favicon = ("connection.png")

server = app.server

style = {
    'NODE_MARKER_SIZE' : 5,
    'HOME_MARKER_SIZE' : 9,
    'NODE_MARKER_COLOUR' : '#E4E4E4',
    'HOME_MARKER_COLOUR' : '#FF0000',
    'LINE_COLOUR' : '#FFFFFF'
}

solver = Solver()
_ = solver.solve_it(2, "NN", "2OPT", [], **style)
_ = solver.solve_it(2, "NI", "2OPT", [], **style)
_ = solver.solve_it(2, "FI", "2OPT", [], **style)


fig = go.Figure(
    layout=go.Layout(
        plot_bgcolor='#444A4D',
        paper_bgcolor='#202324',
        width=750, height=750,
        margin=dict(t=0),
        xaxis=dict(range=[0, 110], autorange=False, visible=False, fixedrange=True),
        yaxis=dict(range=[0, 110], autorange=False, visible=False, fixedrange=True),
        showlegend=False,
    )
)

app.layout = html.Div(
    [dbc.Row(
        [dbc.Col([html.Div(html.H1("Travelling Saleman Problem")),
                html.Div(
                    dbc.Card(
                        dbc.CardBody(
                            html.Div(children=[
                            dbc.Label("Input number of nodes:"),
                            # dbc.Input(id='node_input', value='100', placeholder="Input goes here...", type="text"),
                            dbc.Input(id="node_input", type="number", placeholder="Input",min=5, max=1000, step=1),
                            dbc.FormText("Type an integer between 5 and 1000 in the box above", id='input_alert',  color='white'),
                            dbc.Alert(
                                            "Please enter a number between 5 and 1000",
                                            id="alert-auto",
                                            is_open=False,
                                            duration=4000,
                                            color='danger'
                                        ),
                            html.P(),
                            dbc.Label('Select a method:'),
                            dbc.RadioItems(
                                id="method_input",
                                options=[
                                    {"label": "Nearest Neighbor", "value": "NN"},
                                    {"label": "Nearest Insertion", "value": "NI"},
                                    {"label": "Farthest Insertion", "value": "FI"},
                                    {"label": "Christofides Algorithm", "value": "CA", "disabled":True}
                                ],
                                value='NN'
                            ),
                            html.P(),
                            dbc.Label('Select a method for local search:'),
                            dbc.Checklist(
                                id="LS-input",
                                options=[
                                    {"label": "2-opt", "value": "2OPT", "disabled": "False"},
                                ],
                                value=""
                            ),
                            html.P(),
                            dbc.Label('Would you like to animate the results?'),
                            dbc.Checklist(id='animate_input',
                                        options=[{'label': "Animate", 'value': "animate"}],
                                        value=['animate']
                                    ),
                            html.P(),
                            html.Div(id="slider_container", children=[
                                dbc.Label('Select a speed:'),
                                dcc.Slider(id='speed_input', min=0, max=300, step=None,
                                        marks={
                                                300: {'label': 'Slow', 'style':{'color':'#FFFFFF'}},
                                                200: {'label': '', 'style':{'color':'#FFFFFF'}},
                                                100: {'label': '', 'style':{'color':'#FFFFFF'}},
                                                0: {'label': 'Fast', 'style':{'color':'#FFFFFF'}}
                                            },
                                            value=200
                                        )], style={'display':'Block'}
                            ),
                            html.P(),
                            dcc.Loading(
                                        id="loading-1",
                                        children=[dbc.Button(id='submit-button',
                                                    n_clicks=None,
                                                    children="Submit",
                                                    color="primary", 
                                                    disabled=False,
                                        )],
                                        type="circle"),
                            ])
                        ), style={"padding":"0.5rem"},
                    ), 
                style={"padding":"2rem 2rem 0rem 2rem"}),                  
            ]),
            dbc.Col(
                html.Div(id="graph_container", 
                    children=[
                        dcc.Graph(
                            id='graph',
                            figure=fig,
                            config={'displayModeBar': False},
                            style={'padding':0}
                            ),
                    ],
                ),
            ),
        ]
    )],
    className='container',
    style={'padding':"2rem"}
)

@app.callback([Output('speed_input','value'),
                Output('slider_container','style'),
                Output('LS-input','options')],
                [Input('node_input', 'value'),
                Input('animate_input', 'value')])
def update_options(node_input, animate):
    speed_value = 200
    slider_style = {'display':'Block'}
    ls_disabled = [{"label": "2-opt", "value": "2OPT", "disabled": False}]

    if len(animate) == 0:
        slider_style = {'display':'None'}
         
    if node_input is None:
        print('something')
        return speed_value, slider_style, ls_disabled
    else:
        if node_input < 50:
            speed_value = 300
        elif node_input >= 50 and node_input < 100:
            speed_value = 200
        elif node_input >= 100 and node_input < 300:
            speed_value = 100
        elif node_input >= 300 and node_input < 500:
            speed_value = 0
        else:
            speed_value = 0
            ls_disabled = [{"label": "2-opt", "value": "2OPT", "disabled": True}]
    return speed_value, slider_style, ls_disabled

@app.callback([Output('graph', 'figure'),
                Output('submit-button','children')],
                Input('submit-button', 'n_clicks'),
                [State('speed_input', 'value'),
                State('method_input', 'value'),
                State('LS-input', 'value'),
                State('node_input', 'value'),
                State('animate_input', 'value')])
def update_output(s_clicks, speed_input, method_input, LS_input, node_input, animate_input):
    print(speed_input)

    # try:
    #     int(node_input)
    # except:
    #         return fig, 'red', "Submit"
    
    if node_input is None:
        return fig, "Submit"

    if s_clicks is not None:
        points, solution, data= solver.solve_it(node_input, method_input, LS_input, animate_input, **style)

        fig.data = []
        fig.layout.updatemenus = []

        fig.add_trace(
            go.Scatter(x=[points[0,0]],
                        y=[points[0,1]], 
                        mode='lines'))

        fig.add_trace(
            go.Scatter(x=[points[0,0]],
                        y=[points[0,1]], 
                        mode='markers',
                        marker=dict(color=style['HOME_MARKER_COLOUR'], size=style['HOME_MARKER_SIZE'])))

        fig.add_trace(
            go.Scatter(x=points[1:,0],
                        y=points[1:,1], 
                        mode='markers',
                        marker=dict(color=style['NODE_MARKER_COLOUR'], size=style['NODE_MARKER_SIZE'])))
        
        fig.add_trace(
            go.Scatter(x=[points[0,0]],
                        y=[points[0,1]], 
                        mode='lines',
                        line=dict(color=style['LINE_COLOUR'])))

        fig.add_trace(
            go.Scatter(x=[points[0,0]],
                        y=[points[0,1]], 
                        mode='lines',
                        line=dict(color=style['LINE_COLOUR'])))

        if len(animate_input) != 0:
            fig.layout.updatemenus = [dict(
                    type="buttons",
                    showactive=False,
                    font = dict(size=11, color='#000000'),
                    bgcolor = '#B4B4B4',
                    bordercolor = '#000000',
                    active=50,
                    buttons=[dict(label="Start",
                                method="animate",
                                args=[None, {"frame": {"duration": speed_input, 
                                                        "redraw": False},
                                                        "fromcurrent":False, 
                                                        "transition": {"duration": 10}}]),
                            dict(label="Stop",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, 
                                                        "redraw": False},
                                                        "mode":"immediate", 
                                                        "transition": {"duration": 0}}]),
                    ]
                )]
        else:
            fig.layout.updatemenus = [dict(
                    type="buttons",
                    font = dict(size=11, color='#000000'),
                    bgcolor = '#B4B4B4',
                    bordercolor = '#000000',
                    buttons=[dict(label="Show",
                                method="animate",
                                args=[None, {"frame": {"duration": speed_input, 
                                                        "redraw": False},
                                                        "fromcurrent":False, 
                                                        "transition": {"duration": 0}}]),
                    ]
                )]

        fig.frames = data

    return fig, "Submit"

if __name__ == '__main__':
    app.run_server(debug=True)