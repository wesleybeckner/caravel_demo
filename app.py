# -*- coding: utf-8 -*-
import base64
import io

import dash
import dash_auth
import dash_table
import json
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import time

import pandas as pd
import numpy as np
import datetime
from utils import *

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

########## KP Films
# production_df = pd.read_csv('data/films.csv')
# metric = margin_column = 'EBITDA (€)'
# volume_column = 'Net Sales Quantity in KG'
# descriptors = list(production_df.columns[:8])
# stat_df = pd.read_csv('data/category_stats.csv')

##########

########## Kraton
production_df = pd.read_csv('data/polyamides.csv')
margin_column = metric ='CM Rate ($/hr)'
volume_column = 'Sales Vol'
descriptors = ['U14INT', 'U18', 'TOFA1INT', 'D60', 'M35', 'EDA',
       'H2PO4', 'HMDA', 'Polyetheramine', 'Piperazine', 'Azelaic Acid',
       'Sebacic Acid', 'Dimer Acid', 'Sales Vol']
production_df[descriptors] = np.round(production_df[descriptors].astype(float),1)
stat_df = pd.read_csv('data/kraton_stats.csv')
##########

production_df[descriptors] = production_df[descriptors].astype(str)
production_df = production_df.sort_values(['Product Family', metric],
                                          ascending=False)
production_json = production_df.to_json()
stat_json = stat_df.to_json()




def maximize_ebitda(production_df, stat_df, families, descriptors, volume_column, margin_column):

    local_df = available_indicator_dropdown(production_df, stat_df, families, descriptors)
    old_products = production_df['Product'].unique().shape[0]
    results_df = pd.DataFrame()
    for family in families:
        for index in local_df.index:
            new_df = production_df.loc[production_df['Product Family'] == family]
            new_df = new_df.loc[~(new_df[local_df.iloc[index]['descriptor']] ==\
                    local_df.iloc[index]['group'])]
            new_df = pd.concat([new_df, production_df.loc[~(production_df['Product Family'] == family)]]) # add back fams

            new_EBITDA = new_df[margin_column].sum()
            EBITDA_percent = new_EBITDA / production_df[margin_column].sum() * 100
            EBITDA_delta = new_EBITDA - production_df[margin_column].sum()
            new_products = new_df['Product'].unique().shape[0]
            product_percent_reduction = (new_products) / \
                old_products * 100
            new_kg = new_df[volume_column].sum()
            old_kg = production_df[volume_column].sum()
            kg_percent = new_kg / old_kg * 100

            results = [family, local_df.iloc[index]['descriptor'], local_df.iloc[index]['group'],\
                       new_EBITDA, EBITDA_delta, EBITDA_percent, new_products, product_percent_reduction, new_kg, kg_percent]
            results = pd.DataFrame(results).T
            results.columns = ['Family', 'Descriptor', 'Group', 'EBITDA', 'EBITDA Delta', '% EBITDA', 'Products',\
                               '% Products', 'Volume', '% Volume']
            results_df = pd.concat([results_df, results])
    results_df = results_df.loc[np.abs(results_df['EBITDA Delta']) > 1]
    results_df = results_df.sort_values('% EBITDA', ascending=False).reset_index(drop=True)

    return results_df

def available_indicator_dropdown(production_df, stat_df, families, descriptors):
    df = production_df.loc[production_df['Product Family'].isin(families)]
    local_df = stat_df.loc[stat_df['descriptor'].isin(descriptors)]
    sub_df = pd.DataFrame()
    for i in range(local_df.shape[0]):
        if df.loc[df[local_df.iloc[i]['descriptor']] == local_df.iloc[i]['group']].shape[0] > 0:
            sub_df = pd.concat([sub_df, pd.DataFrame(local_df.iloc[i]).T])
    sub_df = sub_df.reset_index(drop=True)
    return sub_df

def calculate_margin_opportunity(production_df, stat_df, volume_column, margin_column, sort='Worst', select=[0,10], descriptors=None,
                                 families=None, results_df=None):
    old_products = production_df['Product'].unique().shape[0]
    if results_df is not None:
        new_df = production_df
        for index in results_df.index:
            new_df = new_df.loc[~((new_df['Product Family'] == results_df.iloc[index]['Family']) &
                        (new_df[results_df.iloc[index]['Descriptor']] == results_df.iloc[index]['Group']))]
    else:
        local_df = available_indicator_dropdown(production_df, stat_df, families, descriptors)
        if sort == 'Best':
            local_df = local_df.sort_values('score', ascending=False)
            local_df = local_df.reset_index(drop=True)
        else:
            local_df = local_df.sort_values('score', ascending=True)
            local_df = local_df.reset_index(drop=True)
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
            new_df = pd.concat([new_df, production_df.loc[~(production_df['Product Family'].isin(families))]]) # add back fams

    new_EBITDA = new_df[margin_column].sum()
    EBITDA_percent = new_EBITDA / production_df[margin_column].sum() * 100

    new_products = new_df['Product'].unique().shape[0]

    product_percent_reduction = (new_products) / \
        old_products * 100

    new_kg = new_df[volume_column].sum()
    old_kg = production_df[volume_column].sum()
    kg_percent = new_kg / old_kg * 100

    return "€{:.1f} M of €{:.1f} M ({:.1f}%)".format(new_EBITDA/1e6,
                production_df[margin_column].sum()/1e6, EBITDA_percent), \
            "{} of {} Products ({:.1f}%)".format(new_products,old_products,
                product_percent_reduction),\
            "{:.1f} M of {:.1f} M kg ({:.1f}%)".format(new_kg/1e6, old_kg/1e6,
                kg_percent)

def make_violin_plot(production_df, stat_df, margin_column, sort='Worst',
                     select=[0,10], descriptors=None, families=None):
    production_df = production_df.sort_values(
        ['Product Family', margin_column],
        ascending=False).reset_index(drop=True)
    if families != None:
        local_df = available_indicator_dropdown(production_df, stat_df, families, descriptors)
    else:
        local_df = stat_df
    if type(descriptors) == str:
        descriptors = [descriptors]
    if sort == 'Best':
        local_df = local_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = local_df.sort_values('score', ascending=True)
        local_df = local_df.reset_index(drop=True)
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    fig = go.Figure()
    for index in range(select[0],select[1]):
        x = production_df.loc[(production_df[local_df.iloc[index]['descriptor']] == \
            local_df.iloc[index]['group'])][margin_column]
        y = local_df.iloc[index]['descriptor'] + ': ' + production_df.loc[(production_df[local_df\
            .iloc[index]['descriptor']] == local_df.iloc[index]['group'])]\
            [local_df.iloc[index]['descriptor']].astype(str)
        name = '€ {:.0f}'.format(x.median())
        fig.add_trace(go.Violin(x=y,
                                y=x,
                                name=name,
                                box_visible=True,
                                meanline_visible=True))
    fig.update_layout({
                "plot_bgcolor": "#FFFFFF",
                "paper_bgcolor": "#FFFFFF",
                "title": '{} by Product Descriptor (Median in Legend)'.format(margin_column),
                "yaxis.title": "{}".format(margin_column),
                # "height": 400,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4),
                })
    return fig

def make_sunburst_plot(production_df, margin_column, descriptors, clickData=None, toAdd=None, col=None, val=None):
    # production_df[descriptors] = production_df[descriptors].astype(str)
    if clickData != None:
        col = clickData["points"][0]['x'].split(": ")[0]
        val = clickData["points"][0]['x'].split(": ")[1]
        production_df[descriptors] = production_df[descriptors].astype(str)
    elif col == None:
        col = descriptors[0]
        val = production_df[descriptors[0]][0]
    desc = []
    if toAdd != None:
        for item in toAdd:
            if item not in desc:
                desc.append(item)
    if col in desc:
        desc.remove(col)

    test = production_df.loc[production_df[col] == val]

    test[descriptors] = test[descriptors].astype(str)
    fig = px.sunburst(test, path=desc, color=margin_column, title='{}: {}'.format(
        col, val), hover_data=desc,
        color_continuous_scale=px.colors.sequential.Viridis,
         )
    fig.update_layout({
                "plot_bgcolor": "#FFFFFF",
                "title": '{}: {}'.format(col,val),
                "paper_bgcolor": "#FFFFFF",
                # "height": 400,
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
                   stat_df,
                   margin_column,
                   volume_column,
                   select=None,
                   sort='Worst',
                   descriptors=None,
                   family=None,
                   results_df=None):
    local_df = stat_df

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
    production_df[volume_column] = production_df[volume_column].astype(float)


    if select == None:
        for data in px.scatter(
                production_df,
                x='Product',
                y=margin_column,
                size=volume_column,
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

        for index, i in enumerate(new_df['Product']):
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': new_df[margin_column][index],
                           'x1': i,
                           'y1': max(production_df[margin_column]),
                           'line':dict(
                               dash="dot",
                               color=new_df['color'][index],)})
        fig.update_layout(shapes=shapes)

    elif select != None:
        color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(select,
                                                                     colors)}
        for data in px.scatter(
                production_df,
                x='Product',
                y=margin_column,
                color='Product Family',
                size=volume_column,
                color_discrete_map=color_dic,
                opacity=0.6).data:
            fig.add_trace(
                data,
            )



        if sort == 'Best':
            local_df = local_df.sort_values('score', ascending=False)
            local_df = local_df.reset_index(drop=True)
        else:
            local_df = local_df.sort_values('score', ascending=True)
            local_df = local_df.reset_index(drop=True)


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

        for index, i in enumerate(new_df['Product']):
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': new_df[margin_column][index],
                           'x1': i,
                           'y1': max(production_df[margin_column]),
                           'line':dict(
                               dash="dot",
                               color=new_df['color'][index],)})
        fig.update_layout(shapes=shapes)
    fig.update_layout({
            "plot_bgcolor": "#FFFFFF",
            "paper_bgcolor": "#FFFFFF",
            "title": '{} by Product Family'.format(margin_column),
            "yaxis.title": "{}".format(margin_column),
            "height": 600,
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

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df
    # html.Div([
    #     html.H5(filename),
    #     html.H6(datetime.datetime.fromtimestamp(date)),
    #
    #     dash_table.DataTable(
    #         data=df.head(10).to_dict('records'),
    #         columns=[{'name': i, 'id': i} for i in df.columns]
    #     ),
    #
    #     html.Hr(),  # horizontal line
    #
    #     # For debugging, display the raw contents provided by the web browser
    #     html.Div('Raw Content'),
    #     html.Pre(contents[0:200] + '...', style={
    #         'whiteSpace': 'pre-wrap',
    #         'wordBreak': 'break-all'
    #     })
    # ])

UPLOAD = html.Div([
    html.Div([
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '95%',
                'height': '60px',
                # 'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'vertical-align': 'middle',
                'margin': '10px',

                'padding': '5px',
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),],className='four columns',
            style={
            'margin-left': '40px',
            },
        id='up-option-1',),
        html.Div([
        html.P(' - or - ',
        style={
               'textAlign': 'center',
               # 'margin-left': '30px',
               # 'justify-content': 'center',
               # 'align-items': 'center',
               # 'vertical-align': 'middle',
               'margin-top': '30px'}
               ),],className='four columns',
                   id='up-option-2',
                             ),
        html.Div([
        dcc.Dropdown(id='preset-files',
                     multi=False,
                     options=[{'label': i, 'value': i} for i in ['polyamides', 'films']],
                     # placeholder="Select Cloud Dataset",
                     className='dcc_control',
                     style={
                            'textAlign': 'center',
                            'width': '95%',
                            'margin': '10px',
                            # 'height': '60px',
                            # 'margin-right': '40px'
                            }
                            ),],className='four columns',
                            id='up-option-3',
                                                        style={
                                                        # 'textAlign': 'right',
                                                        #        # 'justify-content': 'center',
                                                        #        # 'align-items': 'center',
                                                        #        # 'width': '300px',
                                                        #        # 'height': '60px',
                                                        #        'vertical-align': 'middle',
                                                        # 'width': '95%',
                                                        'margin-right': '40px',

                                                               }
                                                               ),
        ], className='row flex-display',
        ),
    # html.Div([
    html.P('Margin Column'),
    dcc.Dropdown(id='upload-margin',
                 multi=False,
                 options=[],
                 className="dcc_control",
                 style={'textAlign': 'center',
                        'margin-bottom': '10px'}),
    html.P('Volume Column'),
    dcc.Dropdown(id='upload-volume',
                 multi=False,
                 options=[],
                 className="dcc_control",
                 style={'textAlign': 'center',
                        'margin-bottom': '10px'}),
    html.P('Descriptor-Attribute Columns'),
    dcc.Dropdown(id='upload-descriptors',
                 multi=True,
                 options=[],
                 className="dcc_control",
                 style={'textAlign': 'left',
                        'margin-bottom': '10px'}),
    html.P('p-Value Limit for Median Test', id='pvalue-number'),
    dcc.Slider(id='p-value-slider',
               min=0.01,
               max=1,
               step=0.01,
               value=0.5),
    html.Button('Proccess data file',
                id='datafile-button',
                style={'textAlign': 'center',
                       'margin-bottom': '10px'}),
    html.Div(id='production-df-upload',
             style={'display': 'none'},
             children=production_json),
    html.Div(id='stat-df-upload',
             style={'display': 'none'},
             children=stat_json),
    html.Div(id='descriptors-upload',
             style={'display': 'none'},
             children=descriptors),
    html.Div(id='metric-upload',
             style={'display': 'none'},
             children=margin_column),
    html.Div(id='volume-upload',
             style={'display': 'none'},
             children=volume_column),
    html.Div(id='production-df-holding',
             style={'display': 'none'},
             children=None),
],)

ABOUT = html.Div([dcc.Markdown('''

###### This analysis correlates key product attributes (descriptors) with product margin (EBITDA) ######

**KPIs:**

Calculate EBITDA (€)/products/volume from either: 1) eliminating products from 2019 production if ‘Sort by’ = ‘Low EBITDA’, or 2) adding products to an empty production schedule if ‘Sort by’ = ‘High EBITDA’

**Charts:**

EBITDA by Product Family Bubble Chart: Products are grouped by family then rank ordered according to EBITDA. Bubble size is by order volume (kg)

Product Descriptor Violin Chart: Distributions for selected descriptors are shown in a violin plot. Read more about violin plots [here](https://en.wikipedia.org/wiki/Violin_plot)

Descriptor Sunburst Chart: Displays product breakdown for a given descriptor selected in the violin plot. Read from inner to outer rings, the ‘pie’ slices in each ring depict the volume break down for that level in terms of order numbers; classes of product become more specific moving from the inner to outer rings. Color code indicates EBITDA for products described by that level in the pie. Outer rings can toggle width/thickness attributes on and off

**Controls:**

Families & Descriptors: select families/descriptors upon which to perform analysis/visualization

Visualization Tab:

Presets: Interactive allows user selection of various settings, Opportunities 1 & 2 show margin opportunities in the Shrink Sleeve and Cards Core families, respectively

Sort by: if set to ‘High EBITDA’, range bar is sorted from positive to negative correlation with EBITDA; if set to ‘Low EBITDA’, range bar is sorted form negative to positive correlation with EBITDA

Number of Descriptors: range bar selects sorted descriptors. Selection updates plots and KPIs with products described by those descriptors.

Toggle Violin: overlays selected products onto the EBITDA by Product Family chart

Analytics Tab:

Find Opportunity: Algorithm calculates % EBITDA increase by eliminating products according to family/descriptor selection. Table is returned that is sorted from high to low % EBITDA increase.

''')],style={'margin-top': '20px',
             'max-height': '500px',
             'overflow': 'scroll'})

search_bar = dbc.Row(
    [
        dbc.Col(html.Img(src='assets/mfg_logo.png', height="40px")),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

NAVBAR = dbc.Navbar(
    [
            dbc.Row(
                [
                    dbc.Col(html.Img(src='assets/caravel_logo.png', height="40px")),
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
html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H4(id='margin-new-rev'), html.H6(margin_column, id='margin-label')
            ], id='kpi1', className='six columns', style={'margin': '10px'}
            ),
            html.Div([
                html.Img(src='assets/money_icon_1.png', width='80px'),
            ], id='icon1', className='five columns',
                style={
                    'textAlign': 'right',
                    'margin-top': '20px',
                    'margin-right': '20px',
                    'vertical-align': 'text-bottom',
                }),
            ], className='row flex-display',
            ),
        ], className='mini_container',
           id='margin-rev',),
    html.Div([
        html.Div([
            html.Div([
                html.H4(id='margin-new-rev-percent'), html.H6('Unique Products', id='margin-label2')
            ], className='six columns', style={'margin': '10px'}, id='kpi2',
            ),
            html.Div([
                html.Img(src='assets/product_icon_3.png', width='80px'),
            ], className='five columns',
                style={
                    'textAlign': 'right',
                    'margin-top': '20px',
                    'margin-right': '20px',
                    'vertical-align': 'text-bottom',
                }),
            ], className='row flex-display',
            ),
        ], className='mini_container',
           id='margin-rev-percent',),
    html.Div([
        html.Div([
            html.Div([
                html.H4(id='margin-new-products'), html.H6('Volume', id='margin-label3')
            ], className='six columns', style={'margin': '10px'}, id='kpi3',
            ),
            html.Div([
                html.Img(src='assets/volume_icon_3.png', width='80px'),
            ], className='five columns',
                style={
                    'textAlign': 'right',
                    'margin-top': '20px',
                    'margin-right': '20px',
                    'vertical-align': 'text-bottom',
                }),
            ], className='row flex-display',
            ),
        ], className='mini_container',
           id='margin-products',),
        # html.Div([
        #     html.Div([
        #         html.H4(id='margin-new-rev-percent'), html.H6('Unique Products')
        #
        #     html.Div([
        #         html.Img(src='assets/money_icon_1.png', height='80px'),
        #     ], id='icon1', className='five columns',
        #         style={
        #             'textAlign': 'right',
        #             'margin': '20px',
        #             'vertical-align': 'text-bottom',
        #         }),
        #     ], className='row flex-display',
        #     ),
        # ], className='mini_container',
        #    id='margin-rev-percent',
        # ),

        # html.Div([
        #     html.H4(id='margin-new-products'), html.H6('Volume')
        # ], className='mini_container',
        #    id='margin-products',
        # ),
    ], className='row container-display',
    ),
html.Div([
    html.Div([
    dcc.Tabs(id='tabs-control', value='tab-4', children=[
        dcc.Tab(label='About', value='tab-3', children=[ABOUT]),
        dcc.Tab(label='Upload', value='tab-4', children=[UPLOAD]),
        dcc.Tab(label='Visualization', value='tab-1', children=[
            html.Div([
            # html.P('Presets',
            #     style={'margin-bottom': '20px',
            #        'margin-top': '20px'},),
            # dcc.RadioItems(id='preset_view',
            #                 options=[{'label': 'INTERACTIVE  ', 'value': 'INTERACTIVE'},
            #                         {'label': 'OPPORTUNITY 1  ', 'value': 'OPPORTUNITY 1'},
            #                         {'label': 'OPPORTUNITY 2  ', 'value': 'OPPORTUNITY 2'}],
            #                 value='INTERACTIVE',
            #                 labelStyle={'display': 'inline-block'},
            #                 style={'margin-bottom': '20px',
            #                        'margin-top': '20px'},
            #                 inputStyle={"margin-right": "5px",
            #                        "margin-left": "20px"},
            #                 className='dcc_control'),
            html.P('Families'),
            dcc.Dropdown(id='family_dropdown',
                         options=[{'label': i, 'value': i} for i in
                                    production_df['Product Family'].unique()],
                         value=production_df['Product Family'].unique(),
                         multi=True,
                         className="dcc_control"),
            html.P('Descriptors'),
            dcc.Dropdown(id='descriptor_dropdown',
                         # options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                         #         {'label': 'Width', 'value': 'Width Material Attri'},
                         #         {'label': 'Base Type', 'value': 'Base Type'},
                         #         {'label': 'Additional Treatment', 'value': 'Additional Treatment'},
                         #         {'label': 'Color', 'value': 'Color Group'},
                         #         {'label': 'Product Group', 'value': 'Product Group'},
                         #         {'label': 'Base Polymer', 'value': 'Base Polymer'},
                         #         {'label': 'Product Family', 'value': 'Product Family'}],
                         options=[{'label': i, 'value': i} for i in
                                   descriptors],
                         value=descriptors,
                         multi=True,
                         className="dcc_control"),
            html.P('Sort by'),
            dcc.RadioItems(
                        id='sort',
                        options=[{'label': i, 'value': j} for i, j in \
                                [['Low EBITDA', 'Worst'],
                                ['High EBITDA', 'Best']]],
                        value='Worst',
                        labelStyle={'display': 'inline-block'},
                        style={"margin-bottom": "10px"},
                        inputStyle={"margin-right": "5px",
                               "margin-left": "20px"},),
            html.P('Number of Descriptors:', id='descriptor-number'),
            dcc.RangeSlider(
                        id='select',
                        min=0,
                        max=53,
                        step=1,
                        value=[0,10],
            ),
            html.P('Toggle Violin/Descriptor Data onto EBITDA by Product Family'),
            daq.BooleanSwitch(
              id='daq-violin',
              on=False,
              style={"margin-bottom": "10px", "margin-left": "0px",
              'display': 'inline-block'}),
              ],style={'max-height': '500px',
                       'overflow': 'scroll',
                       'margin-top': '20px'}),
              ],),
        dcc.Tab(label='Analytics', value='tab-2', children=[
        html.Div([
        html.P('Families',
        style={"margin-top": "20px"}),
        dcc.Dropdown(id='family_dropdown_analytics',
                     options=[{'label': i, 'value': i} for i in
                                production_df['Product Family'].unique()],
                     value=production_df['Product Family'].unique(),
                     multi=True,
                     className="dcc_control"),
        html.P('Descriptors'),
        dcc.Dropdown(id='descriptor_dropdown_analytics',
                     options=[{'label': i, 'value': i} for i in
                               descriptors],
                     value=descriptors[:2],
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
                  figure=make_ebit_plot(production_df, stat_df, margin_column, volume_column)),
        ], className='mini_container',
           id='ebit-family-block',
           style={'display': 'block'},
        ),
], className='row container-display',
),
    html.Div([
        html.Div([
            dcc.Graph(className='inside_container',
                        id='violin_plot',
                        figure=make_violin_plot(production_df, stat_df, margin_column)),
            html.Div([
            dcc.Loading(
                id="loading-1",
                type="default",
                children=dash_table.DataTable(id='opportunity-table',
                                 row_selectable='multi',),),
                    ],
                    id='opportunity-table-block',
                    style={'overflow': 'scroll',
                           'display': 'none'}),
            ], className='mini_container',
               id='violin',
               style={'display': 'block'},
                ),
        # html.Div([
        #     dcc.Loading(id='violin-load'),
        #
        #     html.Div([
        #     dcc.Graph(id='violin_plot',
        #                 figure=make_violin_plot(production_df, stat_df)),
        #     dcc.Loading(
        #         id="loading-1",
        #         type="default",
        #         children=dash_table.DataTable(id='opportunity-table',
        #                          row_selectable='multi',),),
        #             ],
        #             id='opportunity-table-block',
        #             style={'overflow': 'scroll',
        #                    'display': 'block',
        #                    'max-width': '500px'}),
        #     html.Div([
        #     dcc.Loading(
        #         id="loading-2",
        #         type="default",
        #         children=dash_table.DataTable(id='upload-table',),),
        #             ],
        #             id='upload-table-block',
        #             style={'overflow': 'scroll',
        #                    'display': 'none',
        #                    'max-width': '500px'}),
        #     ], className='mini_container',
        #        id='violin',
        #        style={'display': 'block',
        #               'overflow': 'scroll',
        #               'padding': '0px 20px 20px 20px',
        #               'max-height': '500px'},
        #         ),
        html.Div([
            dcc.Dropdown(id='length_width_dropdown',
                        options=[{'label': i, 'value': i} for i in
                                   descriptors],
                        value=descriptors,
                        multi=True,
                        placeholder="Include in sunburst chart...",
                        className="dcc_control"),
            dcc.Graph(
                        id='sunburst_plot',
                        figure=make_sunburst_plot(production_df, margin_column, descriptors, toAdd=descriptors)),
                ], className='mini_container',
                   id='sunburst',
                ),
            ], className='row container-display',
               style={'margin-bottom': '10px'},
            ),
    ],
    ),
],
)
app.config.suppress_callback_exceptions = True

@app.callback(
    [Output('opportunity-table', 'data'),
    Output('opportunity-table', 'columns'),],
    [Input('production-df-upload', 'children'),
    Input('stat-df-upload', 'children'),
    Input('descriptor_dropdown_analytics', 'value'),
     Input('family_dropdown_analytics', 'value'),
     Input('opportunity-button', 'n_clicks'),
     Input('volume-upload', 'children'),
     Input('metric-upload', 'children')]
)
def display_opportunity_results(production_df, stat_df, descriptors, families, button, volume_column, margin_column):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'opportunity-button.n_clicks':
        production_df = pd.read_json(production_df)
        stat_df = pd.read_json(stat_df)
        results = maximize_ebitda(production_df, stat_df, families, descriptors, volume_column, margin_column)
        results[results.columns[3:]] = np.round(results[results.columns[3:]].astype(float))
        columns=[{"name": i, "id": i} for i in results.columns]
        return results.to_dict('rows'), columns

@app.callback(
    [Output('violin_plot', 'figure'),
     Output('violin_plot', 'style'),
     Output('opportunity-table-block', 'style'),],
    [Input('production-df-upload', 'children'),
    Input('stat-df-upload', 'children'),
    Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('family_dropdown', 'value'),
    Input('tabs-control', 'value'),
    Input('metric-upload', 'children')]
)
def display_violin_plot(production_df, stat_df, sort, select, descriptors, families, tab, margin_column):
    production_df = pd.read_json(production_df)
    stat_df = pd.read_json(stat_df)
    if (tab == 'tab-1') | (tab == 'tab-3') | (tab == 'tab-4'):
        return make_violin_plot(production_df, stat_df, margin_column, sort, select, descriptors, families),\
            {'display': 'block',
             'margin': '10px',
             'padding': '15px',
             'position': 'relative',
             'border-radius': '5px',
             'width': '95%'}, {'display': 'none'}
    elif tab == 'tab-2':
        return make_violin_plot(production_df, stat_df, margin_column, sort, select, descriptors, families),\
            {'display': 'none'}, \
            {'max-height': '500px',
               'overflow': 'scroll',
               'display': 'block',
               'padding': '0px 20px 20px 20px'}

# @app.callback(
#     Output('violin-load', 'children'),
#     [Input('sort', 'value'),
#     Input('select', 'value'),
#     Input('descriptor_dropdown', 'value'),
#     Input('family_dropdown', 'value'),
#     Input('tabs-control', 'value'),
#     Input('production-df-upload', 'children'),
#     Input('stat-df-upload', 'children'),
#     Input('opportunity-button', 'n_clicks'),
#     Input('descriptor_dropdown_analytics', 'value'),
#     Input('family_dropdown_analytics', 'value'),
#     Input('production-df-holding', 'children'),
#     # Input('opportunity-table', 'derived_viewport_selected_rows'),
#     # Input('opportunity-table', 'data'),
#     ]
# )
# def display_violin_plot(sort, select, descriptors, families, tab,
#                         production_df, stat_df, button, descriptors2, families2,
#                         holding_df):
#     ctx = dash.callback_context
#
#     production_df = pd.read_json(production_df)
#     stat_df = pd.read_json(stat_df)
#     if tab == 'tab-4':
#         if holding_df is not None:
#             production_df = pd.read_json(holding_df)
#         columns=[{"name": i, "id": i} for i in production_df.columns]
#         return dash_table.DataTable(id='upload-table',
#             data=production_df.to_dict('rows'),
#             columns=columns)
#     elif (tab == 'tab-2'):
#         # if ctx.triggered[0]['prop_id'] == 'opportunity-button.n_clicks':
#         results = maximize_ebitda(production_df, stat_df, families2, descriptors2)
#         results[results.columns[3:]] = np.round(results[results.columns[3:]].astype(float))
#         columns=[{"name": i, "id": i} for i in results.columns]
#         return dash_table.DataTable(id='opportunity-table',
#                                     row_selectable='multi',
#                                     data=results.to_dict('rows'),
#                                     columns=columns)
#     else:
#         plot = make_violin_plot(production_df, stat_df, sort, select, descriptors, families)
#         return dcc.Graph(figure=plot, id='violin_plot')
#
# @app.callback(
#      Output('violin', 'style'),
#     [Input('tabs-control', 'value'),]
# )
# def display_violin_plot(tab):
#     if (tab == 'tab-4') or (tab == 'tab-2'):
#         return {'display': 'block',
#                'overflow': 'scroll',
#                'padding': '20px 20px 20px 20px',
#                'max-height': '550px'}
#     else:
#         return {'display': 'block',
#                'padding': '20px 20px 20px 20px'}

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
    Input('tabs-control', 'value'),
    Input('production-df-upload', 'children'),
    Input('stat-df-upload', 'children'),
    Input('volume-upload', 'children'),
    Input('metric-upload', 'children')]
)
def display_opportunity(sort, select, descriptors, families, rows, data, tab,
                        production_df, stat_df, volume_column, margin_column):

    if (tab == 'tab-1') or (tab == 'tab-3') or (tab == 'tab-4'):
        production_df = pd.read_json(production_df)
        production_df = production_df.sort_values(['Product Family', margin_column], ascending=False).reset_index(drop=True)
        stat_df = pd.read_json(stat_df)
        return calculate_margin_opportunity(production_df, stat_df, volume_column, margin_column, sort, select, descriptors, families)
    elif (tab == 'tab-2'):
        production_df = pd.read_json(production_df)
        production_df = production_df.sort_values(['Product Family', margin_column], ascending=False).reset_index(drop=True)
        stat_df = pd.read_json(stat_df)
        results_df = pd.DataFrame(data)
        results_df = results_df.iloc[rows].reset_index(drop=True)
        return calculate_margin_opportunity(production_df, stat_df, volume_column, margin_column, results_df=results_df)

@app.callback(
    [Output('descriptor_dropdown', 'options'),
    Output('descriptor_dropdown', 'value'),
    Output('family_dropdown', 'options'),
     Output('family_dropdown', 'value'),
     Output('descriptor_dropdown_analytics', 'options'),
     Output('descriptor_dropdown_analytics', 'value'),
     Output('family_dropdown_analytics', 'options'),
      Output('family_dropdown_analytics', 'value'),
      Output('length_width_dropdown', 'options'),
      Output('length_width_dropdown', 'value'),],
    [Input('descriptors-upload', 'children'),
     Input('production-df-upload', 'children')]
)
def update_dropdowns(descriptors, production_df):
    production_df = pd.read_json(production_df)
    families = list(production_df['Product Family'].unique())
    descriptor_options = [{'label': i, 'value': i} for i in descriptors]
    family_options = columns = [{'label': i, 'value': i} for i in families]
    return descriptor_options, descriptors, family_options, families, descriptor_options, \
        descriptors, family_options, families, descriptor_options, descriptors

# @app.callback(
#     [Output('descriptor_dropdown', 'value'),
#      Output('daq-violin', 'on'),
#      Output('family_dropdown', 'value'),
#      Output('sort', 'value'),],
#     [Input('preset_view', 'value'),
#      Input('descriptor_dropdown', 'options')]
# )
# def update_preset_view(value, options):
#     if value == 'OPPORTUNITY 1':
#         return ['Base Polymer'], True,\
#             ['Shrink Sleeve'], 'Worst'
#     elif value == 'OPPORTUNITY 2':
#         return ['Base Type'], True,\
#             ['Cards Core'], 'Worst'
#     else:
#         return descriptors, False, production_df['Product Family'].unique(),\
#             'Worst'

@app.callback(
    [Output('select', 'max'),
    Output('select', 'value'),],
    [Input('descriptor_dropdown', 'value'),
     # Input('preset_view', 'value'),
     Input('family_dropdown', 'value'),
     Input('production-df-upload', 'children'),
     Input('stat-df-upload', 'children'),
     Input('metric-upload', 'children'),]
)
def update_descriptor_choices(descriptors,families, production_df, stat_df, margin_column):
    production_df = pd.read_json(production_df)
    production_df = production_df.sort_values(['Product Family', margin_column], ascending=False).reset_index(drop=True)
    stat_df = pd.read_json(stat_df)
    min_val = 0
    max_value = 53
    ctx = dash.callback_context
    # if ctx.triggered[0]['prop_id'] == 'preset_view.value':
    #     if preset_view_status == 'OPPORTUNITY 1':
    #         value = 1
    #         max_value = 2
    #     elif preset_view_status == 'OPPORTUNITY 2':
    #         value = 3
    #         max_value = 4
    #         min_val = 0
    #     elif ctx.triggered[0]['value'] == 'INTERACTIVE':
    #         max_value = 53
    #         value = 10
    #     elif (len(families) < 14) | (len(descriptors) < 8):
    #         max_value = available_indicator_dropdown(production_df, stat_df, families, descriptors).shape[0]
    #         value = min(10, max_value)
    # else:
    max_value = available_indicator_dropdown(production_df, stat_df, families, descriptors).shape[0]
    value = min(10, max_value)
    return max_value, [min_val, value]

@app.callback(
    Output('descriptor-number', 'children'),
    [Input('select', 'value')]
)
def display_descriptor_number(select):
    return "Number of Descriptors: {}".format(select[1]-select[0])

@app.callback(
    Output('pvalue-number', 'children'),
    [Input('p-value-slider', 'value')]
)
def display_descriptor_number(select):
    return "p-Value Limit for Median Test: {}".format(select)

@app.callback(
    Output('margin-label', 'children'),
    [Input('metric-upload', 'children')]
)
def display_descriptor_number(select):
    return select


### FIGURES ###
@app.callback(
    Output('ebit_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('daq-violin', 'on'),
    Input('family_dropdown', 'value'),
    Input('opportunity-table', 'derived_viewport_selected_rows'),
    Input('opportunity-table', 'data'),
    Input('tabs-control', 'value'),
    Input('production-df-upload', 'children'),
    Input('stat-df-upload', 'children'),
    Input('metric-upload', 'children'),
    Input('volume-upload', 'children')]
)
def display_ebit_plot(sort, select, descriptors, switch, families, rows, data,
                      tab, production_df, stat_df, margin_column, volume_column):

    production_df = pd.read_json(production_df)
    production_df = production_df.sort_values(['Product Family', margin_column], ascending=False).reset_index(drop=True)
    stat_df = pd.read_json(stat_df)
    stat_df = available_indicator_dropdown(production_df, stat_df, families, descriptors)
    if (tab == 'tab-1') | (tab == 'tab-3') | (tab == 'tab-4'):
        if switch == True:
            select = list(np.arange(select[0],select[1]))
            return make_ebit_plot(production_df, stat_df, margin_column, volume_column, select, sort=sort,
                descriptors=descriptors, family=families)
        else:
            return make_ebit_plot(production_df, stat_df, margin_column, volume_column, family=families)
    elif tab == 'tab-2':
        results_df = pd.DataFrame(data)
        results_df = results_df.iloc[rows].reset_index(drop=True)

        return make_ebit_plot(production_df, stat_df, margin_column, volume_column, results_df=results_df)

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
     Input('tabs-control', 'value'),
     Input('production-df-upload', 'children'),
     Input('stat-df-upload', 'children'),
     Input('metric-upload', 'children')])
def display_sunburst_plot(clickData, toAdd, sort, select, descriptors, families,
                            rows, data, tab, production_df, stat_df, margin_column):


    if (tab == 'tab-1') | (tab == 'tab-3') | (tab == 'tab-4'):
        production_df = pd.read_json(production_df)
        stat_df = pd.read_json(stat_df)
        local_df = available_indicator_dropdown(production_df, stat_df, families, descriptors)
        if sort == 'Best':
            local_df = local_df.sort_values('score', ascending=False)
            local_df = local_df.reset_index(drop=True)
        else:
            local_df = local_df.sort_values('score', ascending=True)
            local_df = local_df.reset_index(drop=True)
        if descriptors != None:
            local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
        local_df = local_df.reset_index(drop=True)
        col = local_df['descriptor'][select[0]]
        val = local_df['group'][select[0]]
    elif tab == 'tab-2':
        production_df = pd.read_json(production_df)
        stat_df = pd.read_json(stat_df)
        results_df = pd.DataFrame(data)
        results_df = results_df.iloc[rows].reset_index(drop=True)
        col=results_df.iloc[-1]['Descriptor']
        val=results_df.iloc[-1]['Group']

    return make_sunburst_plot(production_df, margin_column, descriptors, clickData=clickData, toAdd=toAdd, col=col, val=val)

### UPLOAD TOOL ###
@app.callback(
    [Output('upload-margin', 'options'),
   Output('upload-descriptors', 'options'),
   Output('production-df-holding', 'children'),
   Output('upload-volume', 'options')],
  [Input('upload-data', 'contents'),
   Input('preset-files', 'value')],
  [State('upload-data', 'filename'),
   State('upload-data', 'last_modified')])
def update_production_df_and_table(list_of_contents, preset_file, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df = [parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        df = df[0]
        columns = [{'label': i, 'value': i} for i in df.columns]
        columns_table = [{"name": i, "id": i} for i in df.columns]
        return columns, columns, df.to_json(), columns
    elif preset_file is not None:
        df = pd.read_csv('data/{}.csv'.format(preset_file))
        columns = [{'label': i, 'value': i} for i in df.columns]
        columns_table = [{"name": i, "id": i} for i in df.columns]
        return columns, columns, df.to_json(), columns

@app.callback(
    [Output('production-df-upload', 'children'),
    Output('stat-df-upload', 'children'),
    Output('descriptors-upload', 'children'),
    Output('metric-upload', 'children'),
    Output('volume-upload', 'children'),],
   [Input('production-df-holding', 'children'),
    Input('upload-margin', 'value'),
    Input('upload-descriptors', 'value'),
    Input('datafile-button', 'n_clicks'),
    Input('upload-volume', 'value'),
    Input('p-value-slider', 'value')]
)
def update_main_dataframe(holding_df, margin, descriptors, button, volume, pvalue):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'datafile-button.n_clicks':
        production_df = pd.read_json(holding_df)
        for desc in descriptors: #9 is arbitrary should be a fraction of total datapoints or something
            if (len(production_df[desc].unique()) > 9) and (production_df[desc].dtype == float):
                production_df[desc] = np.round(production_df[desc].astype(float),1)
        stat_df = my_median_test(production_df,
                   metric=margin,
                   descriptors=descriptors,
                   stat_cut_off=pvalue,
                   continuous=False)
        production_df[descriptors] = production_df[descriptors].astype(str)
        production_df = production_df.sort_values(['Product Family', margin],
                                                  ascending=False)
        return production_df.to_json(), stat_df.to_json(), descriptors, margin,\
            volume
# @app.callback(
#     [Output('opportunity-table', 'data'),
#     Output('opportunity-table', 'columns'),],
#     [Input('descriptor_dropdown_analytics', 'value'),
#      Input('family_dropdown_analytics', 'value'),
#      Input('opportunity-button', 'n_clicks'),
#      Input('production-df-upload', 'children'),
#      Input('stat-df-upload', 'children')]
# )
# def display_opportunity_results(descriptors, families,
#                                 button, production_df, stat_df):
#     ctx = dash.callback_context
#     if ctx.triggered[0]['prop_id'] == 'opportunity-button.n_clicks':
#         production_df = pd.read_json(production_df)
#         production_df = production_df.sort_values(['Product Family', margin_column],
#             ascending=False).reset_index(drop=True)
#         stat_df = pd.read_json(stat_df)
#         results = maximize_ebitda(production_df, stat_df, families, descriptors)
#         results[results.columns[3:]] = np.round(results[results.columns[3:]].astype(float))
#         columns=[{"name": i, "id": i} for i in results.columns]
#         return results.to_dict('rows'), columns
# @app.callback(
#     [Output('upload-table', 'data'),
#     Output('upload-table', 'columns'),],
#     [Input('production-df-holding', 'children'),
#      Input('production-df-upload', 'children'),]
# )
# def store_upload_results(df_holding, df_upload):
#     if df_holding is not None:
#
#         production_df = pd.read_json(df_holding)
#     else:
#         production_df = pd.read_json(df_upload)
#     # production_df = production_df.sort_values(['Product Family', margin_column], ascending=False).reset_index(drop=True)
#     columns=[{"name": i, "id": i} for i in production_df.columns]
#     return production_df.to_dict('rows'), columns
if __name__ == "__main__":
    app.run_server(debug=True)
