# -*- coding: utf-8 -*-
import dash
import dash_auth
import dash_table
import json
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import time

import pandas as pd
import numpy as np
import datetime

VALID_USERNAME_PASSWORD_PAIRS = {
    'caravel': 'assessment'
}

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

server = app.server

opportunity = pd.read_csv('data/days.csv', index_col=[0,1,2,3])
annual_operating = pd.read_csv('data/annual.csv', index_col=[0,1])
stats = pd.read_csv('data/scores.csv')
quantiles = np.arange(50,101,1)
quantiles = quantiles*.01
quantiles = np.round(quantiles, decimals=2)
lines = opportunity.index.get_level_values(1).unique()
asset_metrics = ['Yield', 'Rate', 'Uptime']
groupby = ['Line', 'Product group']
oee = pd.read_csv('data/oee.csv')
oee['From Date/Time'] = pd.to_datetime(oee["From Date/Time"])
oee['To Date/Time'] = pd.to_datetime(oee["To Date/Time"])
oee["Run Time"] = pd.to_timedelta(oee["Run Time"])
oee = oee.loc[oee['Rate'] < 2500]
res = oee.groupby(groupby)[asset_metrics].quantile(quantiles)

df = pd.read_csv('data/products.csv')
descriptors = df.columns[:8]
production_df = df
production_df['product'] = production_df[descriptors[2:]].agg('-'.join, axis=1)
production_df = production_df.sort_values(['Product Family', 'EBIT'],
                                          ascending=False)

stat_df = pd.read_csv('data/category_stats.csv')
old_products = df[descriptors].sum(axis=1).unique().shape[0]
weight_match = pd.read_csv('data/weight_match.csv')

def maximize_ebitda(families, descriptors):
    local_df = available_indicator_dropdown(families, descriptors)
    results_df = pd.DataFrame()
    for family in families:
        for index in local_df.index:
            new_df = production_df.loc[production_df['Product Family'] == family]
            new_df = new_df.loc[~(new_df[local_df.iloc[index]['descriptor']] ==\
                    local_df.iloc[index]['group'])]
            new_df = pd.concat([new_df, production_df.loc[~(production_df['Product Family'] == family)]]) # add back fams

            new_EBITDA = new_df['Adjusted EBITDA'].sum()
            EBITDA_percent = new_EBITDA / production_df['Adjusted EBITDA'].sum() * 100
            EBITDA_delta = new_EBITDA - production_df['Adjusted EBITDA'].sum()
            new_products = new_df[production_df.columns[:8]].sum(axis=1).unique().shape[0]
            product_percent_reduction = (new_products) / \
                old_products * 100
            new_kg = new_df['Sales Quantity in KG'].sum()
            old_kg = production_df['Sales Quantity in KG'].sum()
            kg_percent = new_kg / old_kg * 100

            results = [family, local_df.iloc[index]['descriptor'], local_df.iloc[index]['group'],\
                       new_EBITDA, EBITDA_delta, EBITDA_percent, new_products, product_percent_reduction, new_kg, kg_percent]
            results = pd.DataFrame(results).T
            results.columns = ['Family', 'Descriptor', 'Group', 'EBITDA', 'EBITDA Delta', '% EBITDA', 'Products',\
                               '% Products', 'Volume', '% Volume']
            results_df = pd.concat([results_df, results])
    results_df = results_df.loc[results_df['EBITDA Delta'] != 0]
    results_df = results_df.sort_values('% EBITDA', ascending=False).reset_index(drop=True)

    return results_df

def available_indicator_dropdown(families, descriptors):
    df = production_df.loc[production_df['Product Family'].isin(families)]
    local_df = stat_df.loc[stat_df['descriptor'].isin(descriptors)]
    sub_df = pd.DataFrame()
    for i in range(local_df.shape[0]):
        if df.loc[df[local_df.iloc[i]['descriptor']] == local_df.iloc[i]['group']].shape[0] > 0:
            sub_df = pd.concat([sub_df, pd.DataFrame(local_df.iloc[i]).T])
    sub_df = sub_df.reset_index(drop=True)
    return sub_df

def calculate_margin_opportunity(sort='Worst', select=[0,10], descriptors=None,
                                 families=None, results_df=None):
    if results_df is not None:
        new_df = production_df
        for index in results_df.index:
            new_df = new_df.loc[~((new_df['Product Family'] == results_df.iloc[index]['Family']) &
                        (new_df[results_df.iloc[index]['Descriptor']] == results_df.iloc[index]['Group']))]
    else:
        stat_df = available_indicator_dropdown(families, descriptors)
        if sort == 'Best':
            local_df = stat_df.sort_values('score', ascending=False)
            local_df = local_df.reset_index(drop=True)
        else:
            local_df = stat_df
        if descriptors != None:
            local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
        if sort == 'Best':
            if families != None:
                sub_family_df = production_df.loc[production_df['Product Family'].isin(families)]
            else:
                sub_family_df = production_df
            new_df = pd.DataFrame()
            for index in range(select[0],select[1]):
                x = sub_family_df.loc[(sub_family_df[local_df.iloc[index]['descriptor']] == \
                    local_df.iloc[index]['group'])]
                new_df = pd.concat([new_df, x])
            new_df = new_df.drop_duplicates()
        else:
            if families != None:
                new_df = production_df.loc[production_df['Product Family'].isin(families)]
            else:
                new_df = production_df
            for index in range(select[0],select[1]):
                new_df = new_df.loc[~(new_df[local_df.iloc[index]['descriptor']] ==\
                        local_df.iloc[index]['group'])]
            wait = new_df
        if families != None:
            new_df = pd.concat([new_df, production_df.loc[~(df['Product Family'].isin(families))]]) # add back fams

    new_EBITDA = new_df['Adjusted EBITDA'].sum()
    EBITDA_percent = new_EBITDA / production_df['Adjusted EBITDA'].sum() * 100

    new_products = new_df[production_df.columns[:8]].sum(axis=1).unique().shape[0]

    product_percent_reduction = (new_products) / \
        old_products * 100

    new_kg = new_df['Sales Quantity in KG'].sum()
    old_kg = production_df['Sales Quantity in KG'].sum()
    kg_percent = new_kg / old_kg * 100

    return "€{:.1f} M of €{:.1f} M ({:.1f}%)".format(new_EBITDA/1e6,
                production_df['Adjusted EBITDA'].sum()/1e6, EBITDA_percent), \
            "{} of {} Products ({:.1f}%)".format(new_products,old_products,
                product_percent_reduction),\
            "{:.1f} M of {:.1f} M kg ({:.1f}%)".format(new_kg/1e6, old_kg/1e6,
                kg_percent)

def make_violin_plot(sort='Worst', select=[0,10], descriptors=None, families=None):
    if families != None:
        local_df = available_indicator_dropdown(families, descriptors)
    else:
        local_df = stat_df
    if type(descriptors) == str:
        descriptors = [descriptors]
    if sort == 'Best':
        local_df = local_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = local_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    fig = go.Figure()
    for index in range(select[0],select[1]):
        x = df.loc[(df[local_df.iloc[index]['descriptor']] == \
            local_df.iloc[index]['group'])]['Adjusted EBITDA']
        y = local_df.iloc[index]['descriptor'] + ': ' + df.loc[(df[local_df\
            .iloc[index]['descriptor']] == local_df.iloc[index]['group'])]\
            [local_df.iloc[index]['descriptor']]
        name = '€ {:.0f}'.format(x.median())
        fig.add_trace(go.Violin(x=y,
                                y=x,
                                name=name,
                                box_visible=True,
                                meanline_visible=True))
    fig.update_layout({
                "plot_bgcolor": "#FFFFFF",
                "paper_bgcolor": "#FFFFFF",
                "title": 'Adjusted EBITDA by Product Descriptor (Median in Legend)',
                "yaxis.title": "EBITDA (€)",
                "height": 400,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4),
                })
    return fig

def make_sunburst_plot(clickData=None, toAdd=None, col=None, val=None):
    if clickData != None:
        col = clickData["points"][0]['x'].split(": ")[0]
        val = clickData["points"][0]['x'].split(": ")[1]
    elif col == None:
        col = 'Thickness Material A'
        val = '47'
    desc = list(descriptors[:-2])
    if col in desc:
        desc.remove(col)
    if toAdd != None:
        for item in toAdd:
            desc.append(item)
    test = production_df.loc[production_df[col] == val]
    fig = px.sunburst(test, path=desc[:], color='Adjusted EBITDA', title='{}: {}'.format(
        col, val),
        color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout({
                "plot_bgcolor": "#FFFFFF",
                "title": '{}: {}'.format(col,val),
                "paper_bgcolor": "#FFFFFF",
                "height": 400,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
   ),
                })
    return fig

def make_ebit_plot(production_df,
                   select=None,
                   sort='Worst',
                   descriptors=None,
                   family=None,
                   results_df=None):
    production_df = production_df.loc[production_df['Net Sales Quantity in KG'] > 0]
    if results_df is not None:
        production_df = production_df.loc[production_df['Product Family'].isin(results_df['Family'].unique())]
    elif family != None:
        production_df = production_df.loc[production_df['Product Family'].isin(family)]
    families = production_df['Product Family'].unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',\
              '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    colors_cycle = cycle(colors)
    grey = ['#7f7f7f']
    color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(families,
                                                                 colors)}
    grey_dic =  {'{}'.format(i): '{}'.format('#7f7f7f') for i in families}
    fig = go.Figure()


    if select == None:
        for data in px.scatter(
                production_df,
                x='product',
                y='Adjusted EBITDA',
                size='Net Sales Quantity in KG',
                color='Product Family',
                color_discrete_map=color_dic,
                opacity=0.6).data:
            fig.add_trace(
                data
            ),
    if results_df is not None:
        ### the new way w/ family discernment
        new_df = pd.DataFrame()
        for index in results_df.index:
            x = production_df.loc[(production_df['Product Family'] == results_df.iloc[index]['Family']) &
                        (production_df[results_df.iloc[index]['Descriptor']] == results_df.iloc[index]['Group'])]
            x['color'] = next(colors_cycle) # for line shapes
            new_df = pd.concat([new_df, x])
            new_df = new_df.reset_index(drop=True)
        shapes=[]

        for index, i in enumerate(new_df['product']):
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': -4e5,
                           'x1': i,
                           'y1': 4e5,
                           'line':dict(
                               dash="dot",
                               color=new_df['color'][index],)})
        fig.update_layout(shapes=shapes)

    elif select != None:
        color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(select,
                                                                     colors)}
        for data in px.scatter(
                production_df,
                x='product',
                y='Adjusted EBITDA',
                color='Product Family',
                size='Net Sales Quantity in KG',
                color_discrete_map=color_dic,
                opacity=0.6).data:
            fig.add_trace(
                data,
            )


        local_df = available_indicator_dropdown(families, descriptors)
        if sort == 'Best':
            local_df = local_df.sort_values('score', ascending=False)
        elif sort == 'Worst':
            local_df = local_df


            ### the old way w/o family discernment

        new_df = pd.DataFrame()
        if descriptors != None:
            local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
        for index in select:
            x = production_df.loc[(production_df[local_df.iloc[index]\
                ['descriptor']] == local_df.iloc[index]['group'])]
            x['color'] = next(colors_cycle) # for line shapes
            new_df = pd.concat([new_df, x])
            new_df = new_df.reset_index(drop=True)
        shapes=[]

        for index, i in enumerate(new_df['product']):
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': -4e5,
                           'x1': i,
                           'y1': 4e5,
                           'line':dict(
                               dash="dot",
                               color=new_df['color'][index],)})
        fig.update_layout(shapes=shapes)
    fig.update_layout({
            "plot_bgcolor": "#FFFFFF",
            "paper_bgcolor": "#FFFFFF",
            "title": 'Adjusted EBITDA by Product Family',
            "yaxis.title": "EBITDA (€)",
            "height": 500,
            "margin": dict(
                   l=0,
                   r=0,
                   b=0,
                   t=30,
                   pad=4
),
            "xaxis.tickfont.size": 8,
            })
    return fig
search_bar = dbc.Row(
    [
        dbc.Col(html.Img(src='assets/mfg_logo.png', height="30px")),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

NAVBAR = dbc.Navbar(
    [
            dbc.Row(
                [
                    dbc.Col(html.Img(src='assets/caravel_logo.png', height="30px")),
                    # dbc.Col(dbc.NavbarBrand("Product Margin Analysis")),
                ],
                align="center",
                no_gutters=True,
            ),
        dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
    ],
    color="light",
    dark=False,
    sticky='top',
)

app.layout = html.Div([NAVBAR,
html.Div(className='pretty_container', children=[
# html.Div([
#     html.Div([
#         html.H1('Caravel Tier One Assessment Demo'),
#         ], className='nine columns',
#         ),
#     ], className='row flex-display',
#     ),
# html.Div([
#     html.Div([
#         html.H3(["Product Margin Optimization"]),
#         ], className='nine columns',
#         ),
#     ], className='row flex-display',
#     ),
html.Div([
        html.Div([
            html.H6(id='margin-new-rev'), html.P('Adjusted EBITDA')
        ], className='mini_container',
           id='margin-rev',

        ),
        html.Div([
            html.H6(id='margin-new-rev-percent'), html.P('Unique Products')
        ], className='mini_container',
           id='margin-rev-percent',
        ),
        html.Div([
            html.H6(id='margin-new-products'), html.P('Volume')
        ], className='mini_container',
           id='margin-products',
        ),
    ], className='row container-display',
    ),
html.Div([
    html.Div([
    dcc.Tabs(id='tabs-control', value='tab-1', children=[
        dcc.Tab(label='Visualization', value='tab-1', children=[
            html.P('Presets',
                style={'margin-bottom': '20px',
                   'margin-top': '20px'},),
            dcc.RadioItems(id='preset_view',
                            options=[{'label': 'INTERACTIVE  ', 'value': 'INTERACTIVE'},
                                    {'label': 'VIEW 1  ', 'value': 'VIEW 1'},
                                    {'label': 'VIEW 2  ', 'value': 'VIEW 2'},
                                    {'label': 'VIEW 3  ', 'value': 'VIEW 3'}],
                            value='INTERACTIVE',
                            labelStyle={'display': 'inline-block'},
                            style={'margin-bottom': '20px',
                                   'margin-top': '20px'},
                            inputStyle={"margin-right": "5px",
                                   "margin-left": "20px"},
                            className='dcc_control'),
            html.P('Families'),
            dcc.Dropdown(id='family_dropdown',
                         options=[{'label': i, 'value': i} for i in
                                    production_df['Product Family'].unique()],
                         value=production_df['Product Family'].unique(),
                         multi=True,
                         className="dcc_control"),
            html.P('Descriptors'),
            dcc.Dropdown(id='descriptor_dropdown',
                         options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                 {'label': 'Width', 'value': 'Width Material Attri'},
                                 {'label': 'Base Type', 'value': 'Base Type'},
                                 {'label': 'Additional Treatment', 'value': 'Additional Treatment'},
                                 {'label': 'Color', 'value': 'Color Group'},
                                 {'label': 'Product Group', 'value': 'Product Group'},
                                 {'label': 'Base Polymer', 'value': 'Base Polymer'},
                                 {'label': 'Product Family', 'value': 'Product Family'}],
                         value=['Thickness Material A',
                                'Width Material Attri', 'Base Type',
                                'Additional Treatment', 'Color Group',
                                'Product Group',
                                'Base Polymer', 'Product Family'],
                         multi=True,
                         className="dcc_control"),
            html.P('Number of Descriptors:', id='descriptor-number'),
            dcc.RangeSlider(
                        id='select',
                        min=0,
                        max=53,
                        step=1,
                        value=[0,10],
            ),
            html.P('Sort by'),
            dcc.RadioItems(
                        id='sort',
                        options=[{'label': i, 'value': j} for i, j in \
                                [['Low EBITDA', 'Worst'],
                                ['High EBITDA', 'Best']]],
                        value='Best',
                        labelStyle={'display': 'inline-block'},
                        style={"margin-bottom": "10px"},
                        inputStyle={"margin-right": "5px",
                               "margin-left": "20px"},),
            html.P('Toggle Violin/Descriptor Data onto EBITDA by Product Family'),
            daq.BooleanSwitch(
              id='daq-violin',
              on=False,
              style={"margin-bottom": "10px", "margin-left": "0px",
              'display': 'inline-block'}),
              ]),
    dcc.Tab(label='Analytics', value='tab-2', children=[
        html.Div([
        html.P('Families',
        style={"margin-top": "20px"}),
        dcc.Dropdown(id='family_dropdown_analytics',
                     options=[{'label': i, 'value': i} for i in
                                production_df['Product Family'].unique()],
                     # value=production_df['Product Family'].unique(),
                     value=['Shrink Sleeve', 'Cards Core'],
                     multi=True,
                     className="dcc_control"),
        html.P('Descriptors'),
        dcc.Dropdown(id='descriptor_dropdown_analytics',
                     options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                             {'label': 'Width', 'value': 'Width Material Attri'},
                             {'label': 'Base Type', 'value': 'Base Type'},
                             {'label': 'Additional Treatment', 'value': 'Additional Treatment'},
                             {'label': 'Color', 'value': 'Color Group'},
                             {'label': 'Product Group', 'value': 'Product Group'},
                             {'label': 'Base Polymer', 'value': 'Base Polymer'},
                             {'label': 'Product Family', 'value': 'Product Family'}],
                     # value=['Thickness Material A',
                     #        'Width Material Attri', 'Base Type',
                     #        'Additional Treatment', 'Color Group',
                     #        'Product Group',
                     #        'Base Polymer', 'Product Family'],
                     value=['Base Polymer', 'Base Type'],
                     multi=True,
                     className="dcc_control",
                     style={'margin-bottom': '10px'}),
        html.Button('Find opportunity',
                    id='opportunity-button',
                    style={'textAlign': 'center',
                           'margin-bottom': '10px'}),
            ]),
            ]),
        ]),
        ], className='mini_container',
           id='descriptorBlock',
        ),
    html.Div([
        dcc.Graph(id='ebit_plot',
                  figure=make_ebit_plot(production_df)),
        ], className='mini_container',
           id='ebit-family-block'
        ),
], className='row container-display',
),
    html.Div([
        html.Div([
            dcc.Graph(
                        id='violin_plot',
                        figure=make_violin_plot()),
            html.Div([
            dcc.Loading(
                id="loading-1",
                type="default",
                children=dash_table.DataTable(id='opportunity-table',
                                 row_selectable='multi',),),
                    ],
                    id='opportunity-table-block',
                    style={'margin-left': '2%',
                           'margin-right': '2%',
                           'max-height': '700px',
                           'max-width': '600px',
                           'overflow': 'scroll',
                           'display': 'none'}),
            ], className='mini_container',
               id='violin',
               style={'display': 'block'},
                ),
        html.Div([
            dcc.Dropdown(id='length_width_dropdown',
                        options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                 {'label': 'Width', 'value': 'Width Material Attri'}],
                        value=['Width Material Attri'],
                        multi=True,
                        placeholder="Include in sunburst chart...",
                        className="dcc_control"),
            dcc.Graph(
                        id='sunburst_plot',
                        figure=make_sunburst_plot()),
                ], className='mini_container',
                   id='sunburst',
                ),
            ], className='row container-display',
               style={'margin-bottom': '10px'},
            ),
    ], #className='pretty container'
    ),
],
)
app.config.suppress_callback_exceptions = True

@app.callback(
    [Output('opportunity-table', 'data'),
    Output('opportunity-table', 'columns'),],
    [Input('descriptor_dropdown_analytics', 'value'),
     Input('family_dropdown_analytics', 'value'),
     Input('opportunity-button', 'n_clicks')]
)
def display_opportunity_results(descriptors, families, button):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'opportunity-button.n_clicks':
        results = maximize_ebitda(families, descriptors)
        results[results.columns[3:]] = np.round(results[results.columns[3:]].astype(float))
        columns=[{"name": i, "id": i} for i in results.columns]
        return results.to_dict('rows'), columns
    else:
        return None

@app.callback(
    [Output('descriptor_dropdown', 'value'),
     Output('daq-violin', 'on'),
     Output('family_dropdown', 'value'),
     Output('sort', 'value'),],
    [Input('preset_view', 'value'),
     Input('descriptor_dropdown', 'options')]
)
def update_preset_view(value, options):
    if value == 'VIEW 1':
        return ['{}'.format(options[6]['value']),], True,\
            ['Shrink Sleeve'], 'Worst'
    elif value == 'VIEW 2':
        return ['Base Type'], True,\
            ['Shrink Sleeve'], 'Worst'
    elif value == 'VIEW 3':
        return ['Base Type'], True,\
            ['Cards Core'], 'Worst'
    else:
        return descriptors, False, production_df['Product Family'].unique(),\
            'Best'

@app.callback(
    Output('sunburst_plot', 'figure'),
    [Input('violin_plot', 'clickData'),
     Input('length_width_dropdown', 'value'),
     Input('sort', 'value'),
     Input('select', 'value'),
     Input('descriptor_dropdown', 'value'),
     Input('family_dropdown', 'value'),
     Input('opportunity-table', 'derived_viewport_selected_rows'),
     Input('opportunity-table', 'data'),
     Input('tabs-control', 'value')])
def display_sunburst_plot(clickData, toAdd, sort, select, descriptors, families,
                            rows, data, tab):
    if tab == 'tab-1':
        stat_df = available_indicator_dropdown(families, descriptors)
        if sort == 'Best':
            local_df = stat_df.sort_values('score', ascending=False)
            local_df = local_df.reset_index(drop=True)
        else:
            local_df = stat_df
        if descriptors != None:
            local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
        local_df = local_df.reset_index(drop=True)
        col = local_df['descriptor'][select[0]]
        val = local_df['group'][select[0]]
        return make_sunburst_plot(clickData, toAdd, col, val)
    elif tab == 'tab-2':
        results_df = pd.DataFrame(data)
        results_df = results_df.iloc[rows].reset_index(drop=True)
        col=results_df.iloc[-1]['Descriptor']
        val=results_df.iloc[-1]['Group']
        return make_sunburst_plot(col=col, val=val)

@app.callback(
    Output('descriptor-number', 'children'),
    [Input('select', 'value')]
)
def display_descriptor_number(select):
    return "Number of Descriptors: {}".format(select[1]-select[0])

@app.callback(
    [Output('violin_plot', 'figure'),
     Output('violin_plot', 'style'),
     Output('opportunity-table-block', 'style')],
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('family_dropdown', 'value'),
    Input('tabs-control', 'value')]
)
def display_violin_plot(sort, select, descriptors, families, tab):

    if tab == 'tab-1':
        return make_violin_plot(sort, select, descriptors, families),\
            {'display': 'block'}, {'display': 'none'}
    else:
        return make_violin_plot(sort, select, descriptors, families),\
            {'display': 'none'}, \
            {'max-height': '500px',
               'overflow': 'scroll',
               'display': 'block',
               'padding': '0px 20px 20px 20px'}

@app.callback(
    Output('ebit_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('daq-violin', 'on'),
    Input('family_dropdown', 'value'),
    Input('opportunity-table', 'derived_viewport_selected_rows'),
    Input('opportunity-table', 'data'),
    Input('tabs-control', 'value')]
)
def display_ebit_plot(sort, select, descriptors, switch, families, rows, data,
                      tab):
    if tab == 'tab-1':
        if switch == True:
            select = list(np.arange(select[0],select[1]))
            return make_ebit_plot(production_df, select, sort=sort,
                descriptors=descriptors, family=families)
        else:
            return make_ebit_plot(production_df, family=families)
    elif tab == 'tab-2':
        results_df = pd.DataFrame(data)
        results_df = results_df.iloc[rows].reset_index(drop=True)
        return make_ebit_plot(production_df, results_df=results_df)

@app.callback(
    [Output('margin-new-rev', 'children'),
     Output('margin-new-rev-percent', 'children'),
     Output('margin-new-products', 'children')],
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('family_dropdown', 'value'),
    Input('opportunity-table', 'derived_viewport_selected_rows'),
    Input('opportunity-table', 'data'),
    Input('tabs-control', 'value')]
)
def display_opportunity(sort, select, descriptors, families, rows, data, tab):
    if tab == 'tab-1':
        return calculate_margin_opportunity(sort, select, descriptors, families)
    elif tab == 'tab-2':
        results_df = pd.DataFrame(data)
        results_df = results_df.iloc[rows].reset_index(drop=True)
        return calculate_margin_opportunity(results_df=results_df)

@app.callback(
    [Output('select', 'max'),
    Output('select', 'value'),],
    [Input('descriptor_dropdown', 'value'),
     Input('preset_view', 'value'),
     Input('family_dropdown', 'value'),]
)
def update_descriptor_choices(descriptors, preset_view_status, families):
    min_val = 0
    max_value = 53
    ctx = dash.callback_context
    if preset_view_status == 'VIEW 1':
        value = 1
        max_value = 2
    elif preset_view_status == 'VIEW 2':
        value = 2
        max_value = 2
    elif preset_view_status == 'VIEW 3':
        value = 3
        max_value = 4
        min_val = 0
    elif ctx.triggered[0]['value'] == 'INTERACTIVE':
        max_value = 53
        value = 10
    elif (len(families) < 14) | (len(descriptors) < 8):
        max_value = available_indicator_dropdown(families, descriptors).shape[0]
        value = min(10, max_value)


    else:
        # stat_df = available_indicator_dropdown(families, descriptors)

        max_value = available_indicator_dropdown(families, descriptors).shape[0]
        value = min(10, max_value)
    return max_value, [min_val, value]

if __name__ == "__main__":
    app.run_server(debug=True)
