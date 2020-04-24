# -*- coding: utf-8 -*-
import dash
import dash_auth
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

import pandas as pd
import numpy as np
import datetime

VALID_USERNAME_PASSWORD_PAIRS = {
    'caravel': 'assessment'
}

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
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

def bubble_chart_kpi(x='EBITDA per Hr Rank', y='Adjusted EBITDA', color='Line',
                      size='Net Sales Quantity in KG'):
    if x == 'EBITDA per Hr Rank':
        lowx = weight_match.groupby(color)[x].mean().sort_values().index[-1]
        highx = weight_match.groupby(color)[x].mean().sort_values().index[0]
    else:
        lowx = weight_match.groupby(color)[x].mean().sort_values().index[0]
        highx = weight_match.groupby(color)[x].mean().sort_values().index[-1]
    lowy = weight_match.groupby(color)[y].mean().sort_values().index[0]
    highy = weight_match.groupby(color)[y].mean().sort_values().index[-1]

    return "{}".format(highy), \
            "top {} {}".format(y, color), \
            "{}".format(highx), \
            "top {} {}".format(x, color), \
            "{}".format(lowy), \
            "bottom {} {}".format(y, color), \
            "{}".format(lowx), \
            "bottom {} {}".format(x, color)


def make_bubble_chart(x='EBITDA per Hr Rank', y='Adjusted EBITDA', color='Line',
                      size='Net Sales Quantity in KG'):

    fig = px.scatter(weight_match, x=x, y=y, color=color, size=size)
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                # "title": 'EBIT by Product Descriptor',
                })

    return fig

def calculate_margin_opportunity(sort='Worst', select=[0,10], descriptors=None,
                                 families=None):
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    if sort == 'Best':
        if families != None:
            sub_family_df = df.loc[df['Product Family'].isin(families)]
        else:
            sub_family_df = df
        new_df = pd.DataFrame()
        for index in range(select[0],select[1]):
            x = sub_family_df.loc[(sub_family_df[local_df.iloc[index]['descriptor']] == \
                local_df.iloc[index]['group'])]
            new_df = pd.concat([new_df, x])
        new_df = new_df.drop_duplicates()
    else:
        if families != None:
            new_df = df.loc[df['Product Family'].isin(families)]
        else:
            new_df = df
        for index in range(select[0],select[1]):
            new_df = new_df.loc[~(new_df[local_df.iloc[index]['descriptor']] ==\
                    local_df.iloc[index]['group'])]
        wait = new_df
    if families != None:
        new_df = pd.concat([new_df, df.loc[~(df['Product Family'].isin(families))]]) # add back fams

    new_EBITDA = new_df['Adjusted EBITDA'].sum()
    EBITDA_percent = new_EBITDA / df['Adjusted EBITDA'].sum() * 100

    new_products = new_df[df.columns[:8]].sum(axis=1).unique().shape[0]

    product_percent_reduction = (new_products) / \
        old_products * 100

    new_kg = new_df['Sales Quantity in KG'].sum()
    old_kg = df['Sales Quantity in KG'].sum()
    kg_percent = new_kg / old_kg * 100

    return "€{:.1f} M of €{:.1f} M ({:.1f}%)".format(new_EBITDA/1e6,
                df['Adjusted EBITDA'].sum()/1e6, EBITDA_percent), \
            "{} of {} Products ({:.1f}%)".format(new_products,old_products,
                product_percent_reduction),\
            "{:.1f} M of {:.1f} M kg ({:.1f}%)".format(new_kg/1e6, old_kg/1e6,
                kg_percent)

def make_violin_plot(sort='Worst', select=[0,10], descriptors=None):
    if type(descriptors) == str:
        descriptors = [descriptors]
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
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
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": 'Adjusted EBITDA by Product Descriptor (Median in Legend)',
                "yaxis.title": "EBITDA (€)",
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
                "plot_bgcolor": "#F9F9F9",
                "title": '(Select in Violin) {}: {}'.format(col,val),
                "paper_bgcolor": "#F9F9F9",
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
                   family=None):
    production_df = production_df.loc[production_df['Net Sales Quantity in KG'] > 0]
    if family != None:
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
                opacity=1).data:
            fig.add_trace(
                data
            ),


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

        if sort == 'Best':
            local_df = stat_df.sort_values('score', ascending=False)
        elif sort == 'Worst':
            local_df = stat_df


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
            "plot_bgcolor": "#F9F9F9",
            "paper_bgcolor": "#F9F9F9",
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

def calculate_overlap(lines=['E27', 'E26']):
    path=['Product group', 'Polymer', 'Base Type', 'Additional Treatment']

    line1 = oee.loc[oee['Line'].isin([lines[0]])].groupby(path)\
                    ['Quantity Good'].sum()
    line2 = oee.loc[oee['Line'].isin([lines[1]])].groupby(path)\
                    ['Quantity Good'].sum()

    set1 = set(line1.index)
    set2 = set(line2.index)

    both = set1.intersection(set2)
    unique = set1.union(set2) - both

    kg_overlap = (line1.loc[list(both)].sum() + line2.loc[list(both)].sum()) /\
    (line1.sum() + line2.sum())
    return kg_overlap*100

def make_product_sunburst(lines=['E27', 'E26']):
    fig = px.sunburst(oee.loc[oee['Line'].isin(lines)],
        path=['Product group', 'Polymer', 'Base Type', 'Additional Treatment',\
                'Line'],
        color='Line')
    overlap = calculate_overlap(lines)
    fig.update_layout({
                 "plot_bgcolor": "#F9F9F9",
                 "paper_bgcolor": "#F9F9F9",
                 "height": 500,
                 "margin": dict(
                        l=0,
                        r=0,
                        b=0,
                        t=30,
                        pad=4
    ),
                 "title": "Product Overlap {:.1f}%: {}, {}".format(overlap,
                                                            lines[0], lines[1]),
     })
    return fig

def compute_distribution_results(line='K40', pareto='Product', toggle='Yield'):

    if line != 'All Lines':
        plot = oee.loc[oee['Line'] == line]
    else:
        plot = oee
    plot = plot.sort_values('Thickness Material A')
    plot['Thickness Material A'] = pd.to_numeric(plot['Thickness Material A'])
    cut=3
    plot = plot.groupby(pareto).filter(lambda x : (x[pareto].count()>=cut).any())

    low_mean = plot.groupby(pareto)[toggle].mean().sort_values().reset_index().iloc[0][0]
    high_mean = plot.groupby(pareto)[toggle].mean().sort_values().reset_index().iloc[-1][0]
    low_std = plot.groupby(pareto)[toggle].std().sort_values().reset_index().iloc[0][0]
    high_std = plot.groupby(pareto)[toggle].std().sort_values().reset_index().iloc[-1][0]

    low_mean_val = plot.groupby(pareto)[toggle].mean().sort_values().reset_index().iloc[0][1]
    high_mean_val = plot.groupby(pareto)[toggle].mean().sort_values().reset_index().iloc[-1][1]
    low_std_val = plot.groupby(pareto)[toggle].std().sort_values().reset_index().iloc[0][1]
    high_std_val = plot.groupby(pareto)[toggle].std().sort_values().reset_index().iloc[-1][1]

    if toggle == 'Rate':
        units = ' kg/hr'
    elif toggle == 'Yield':
        units = ''
    if pareto == 'Thickness Material A':
        pareto = 'Thickness'

    return "{}".format(high_mean), \
           "Highest Avg {} {} ({:.1f}{})".format(toggle, pareto, high_mean_val, units), \
           "{}".format(low_mean), \
           "Lowest Avg {} {} ({:.1f}{})".format(toggle, pareto, low_mean_val, units), \
           "{}".format(low_std), \
           "Highest Variability {} {} ({:.1f}{})".format(toggle, pareto, high_std_val, units), \
           "{}".format(high_std), \
           "Lowest Variability {} {} ({:.1f}{})".format(toggle, pareto, low_std_val, units)

def make_metric_plot(line='K40', pareto='Product', marginal='histogram',
                     toggle='Yield'):
    if line != 'All Lines':
        plot = oee.loc[oee['Line'] == line]
    else:
        plot = oee
    plot = plot.sort_values('Thickness Material A')
    plot['Thickness Material A'] = pd.to_numeric(plot['Thickness Material A'])
    mean_of_the_std = plot.groupby(pareto)[toggle].std().mean()
    if marginal == 'none':
        fig = px.density_contour(plot, x='Rate', y='Yield',
                     color=pareto)
    else:
        fig = px.density_contour(plot, x='Rate', y='Yield',
                 color=pareto, marginal_x=marginal, marginal_y=marginal)
    fig.update_layout({
                 "plot_bgcolor": "#F9F9F9",
                 "paper_bgcolor": "#F9F9F9",
                 "height": 750,
                 "title": "{}, {:.2f} Average {} Variance By {}".\
                    format(line, mean_of_the_std, toggle, pareto),
     })
    return fig

def make_utilization_plot():
    downdays = pd.DataFrame(oee.groupby('Line')['Uptime'].sum().sort_values()/24)
    downdays.columns = ['Unutilized Days, 2019']
    fig = px.bar(downdays, y=downdays.index, x='Unutilized Days, 2019',
           orientation='h', color=downdays.index)
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": "Utilization, All Lines (Note: data did not "\
                        "distinguish between downtime and utilization)",
                "height": 400,
    })
    return fig

def find_quantile(to_remove_line='E26', to_add_line='E27',
                  metrics=['Rate', 'Yield', 'Uptime'],
                  uptime=None):
    if type(to_add_line) == str:
        to_add_line = [to_add_line]
    to_remove_kg = annual_operating.loc[to_remove_line]['Quantity Good']
    to_remove_rates = res.loc[to_remove_line].unstack()['Rate']
    to_remove_yields = res.loc[to_remove_line].unstack()['Yield']
    target_days_needed = pd.DataFrame(to_remove_kg).values / to_remove_yields\
                            / to_remove_rates / 24
    target_days_needed = target_days_needed.T
    target_days_needed['Total'] = target_days_needed.sum(axis=1)

    target_data = opportunity.loc['Additional Days'].loc[to_add_line].unstack()\
                    [metrics].sum(axis=1)
    target = pd.DataFrame(target_data).unstack().T.loc[0]

    if uptime != None:
        target[to_add_line] = target[to_add_line] + uptime

    final = pd.merge(target_days_needed, target, left_index=True, right_index=True)
    quantile = (abs(final[to_add_line].sum(axis=1) - final['Total'])).idxmin()
    return quantile, final.iloc[:-1]

def make_consolidate_plot(remove='E26', add='E27',
                          metrics=['Rate', 'Yield', 'Uptime'],
                          uptime=None):
    quantile, final = find_quantile(remove, add, metrics, uptime)
    fig = go.Figure(data=[
    go.Bar(name='Days Available', x=final.index, y=final[add]),
    go.Bar(name='Days Needed', x=final.index, y=final['Total'])
    ])
    if uptime != None:
        title = "Quantile-Performance Target: {} + {} Uptime Days"\
            .format(quantile, uptime)
    else:
        title = "Quantile-Performance Target: {}".format(quantile)
    # Change the bar mode
    fig.update_layout(barmode='group',
                  yaxis=dict(title="Days"),
                   xaxis=dict(title="Quantile"))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": title,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
   ),
    })
    return fig

def pareto_product_family(quantile=0.9, clickData=None):
    if clickData != None:
        line = clickData["points"][0]["y"]
    else:
        line = 'K40'
    data = opportunity.reorder_levels([0,2,1,3]).sort_index().\
            loc['Additional Days', quantile, line]
    total = data.sum().sum()
    cols = data.columns
    bar_fig = []
    for col in cols:
        bar_fig.append(
        go.Bar(
        name=col,
        orientation="h",
        y=[str(i) for i in data.index],
        x=data[col],
        customdata=[col],
        )
        )

    figure = go.Figure(
        data=bar_fig,
        layout=dict(
            barmode="group",
            yaxis_type="category",
            yaxis=dict(title="Product Group"),
            xaxis=dict(title="Days"),
            title="{}: {:.1f} days of opportunity".format(line,total),
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9"
        )
    )
    figure.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return figure

def make_days_plot(quantile=0.9):
    data = opportunity.reorder_levels([0,2,1,3]).sort_index()\
                .loc['Additional Days', quantile].groupby('Line').sum()
    cols = ['Rate', 'Yield', 'Uptime']
    data['Total'] = data.sum(axis=1)
    data = data.sort_values(by='Total')
    bar_fig = []
    for col in cols:
        bar_fig.append(
        go.Bar(
        name=col,
        orientation="h",
        y=[str(i) for i in data.index],
        x=data[col],
        customdata=[col]
        )
    )

    figure = go.Figure(
        data=bar_fig,
        layout=dict(
            barmode="stack",
            yaxis_type="category",
            yaxis=dict(title="Line"),
            xaxis=dict(title="Days"),
            title="Annualized Opportunity",
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9"
        )
    )
    figure.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return figure

def make_culprits():
    fig = px.bar(stats, x='group', y='score', color='metric',
        barmode='group')
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "xaxis.title": "Contingency Table Score",
    })
    return fig

def pie_line(clickData=None):
    if clickData != None:
        line = clickData["points"][0]["y"]
    else:
        line = 'K40'
    data = annual_operating.loc[line]
    total = data['Net Quantity Produced'].sum()/1e6
    fig = px.pie(data, values='Net Quantity Produced', names=data.index,
                title='Production distribution 2019 ({:.1f}M kg)'.format(total))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return fig

def calculate_opportunity(quantile=0.9):
    data = opportunity.reorder_levels([0,2,1,3]).sort_index()\
                .loc['Additional Days', quantile].groupby('Line').sum()
    data['Total'] = data.sum(axis=1)
    return "{:.1f}".format(data.sum()[3]), \
            "{:.1f}".format(data.sum()[0]), \
            "{:.1f}".format(data.sum()[1]), \
            "{:.1f}".format(data.sum()[2])
# Describe the layout/ UI of the app

app.layout = html.Div([
html.Div([
    html.Div([
        html.H1('Caravel Tier One Assessment Demo'),
        ], className='nine columns',
        ),
    ], className='row flex-display',
    ),
html.Div([
    html.Div([
        # html.H2('Company Name'),
        ], className='nine columns',
        ),
    ], className='row flex-display',
    ),
html.Div([
    html.Div([
        html.H3(["Product Margin Optimization"]),
        ], className='nine columns',
        ),
], className='row flex-display',
),
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
            dcc.RadioItems(id='view1',
                            options=[{'label': 'INTERACTIVE', 'value': 'INTERACTIVE'},
                                    {'label': 'VIEW 1', 'value': 'VIEW 1'},
                                    {'label': 'VIEW 2', 'value': 'VIEW 2'},
                                    {'label': 'VIEW 3', 'value': 'VIEW 3'}],
                            value='INTERACTIVE',
                            labelStyle={'display': 'inline-block'},
                        style={'margin-bottom': '20px'}),
            html.P(' '),
            html.P(' '),
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
                        max=stat_df.shape[0],
                        step=1,
                        value=[0,10],
            ),
            html.P('Sort by:'),
            dcc.RadioItems(
                        id='sort',
                        options=[{'label': i, 'value': j} for i, j in \
                                [['Low EBITDA', 'Worst'],
                                ['High EBITDA', 'Best']]],
                        value='Best',
                        labelStyle={'display': 'inline-block'},
                        style={"margin-bottom": "10px"},),
            html.P('Toggle Violin/Descriptor Data onto EBITDA by Product Family:'),
            daq.BooleanSwitch(
              id='daq-violin',
              on=False,
              style={"margin-bottom": "10px", "margin-left": "0px",
              'display': 'inline-block'}),
                ], className='mini_container',
                    id='descriptorBlock',
                ),
            html.Div([
                dcc.Graph(
                            id='ebit_plot',
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
                ], className='mini_container',
                   id='violin',
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
               style={'margin-bottom': '50px'},
            ),
    ], className='pretty container'
    )

app.config.suppress_callback_exceptions = False

@app.callback(
    [Output('descriptor_dropdown', 'value'),
     Output('daq-violin', 'on'),
     Output('family_dropdown', 'value'),
     Output('sort', 'value'),],
    [Input('view1', 'value'),
     Input('descriptor_dropdown', 'options')]
)
def update_view1(value, options):
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
    [Output('select', 'max'),
    Output('select', 'value'),],
    [Input('descriptor_dropdown', 'value'),
     Input('view1', 'value'),]
)
def update_descriptor_choices(descriptors, view1_status):
    min_val = 0
    if view1_status == 'VIEW 1':
        value = 1
        max_value = 2
    elif view1_status == 'VIEW 2':
        value = 2
        max_value = 16
    elif view1_status == 'VIEW 3':
        value = 6
        max_value = 16
        min_val = 2
    else:
        max_value = stat_df.loc[stat_df['descriptor'].isin(descriptors)].shape[0]
        max_value = 53
        value = min(10, max_value)
    return max_value, [min_val, value]

@app.callback(
    Output('sunburst_plot', 'figure'),
    [Input('violin_plot', 'clickData'),
     Input('length_width_dropdown', 'value'),
     Input('sort', 'value'),
     Input('select', 'value'),
     Input('descriptor_dropdown', 'value')])
def display_sunburst_plot(clickData, toAdd, sort, select, descriptors):
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

@app.callback(
    Output('descriptor-number', 'children'),
    [Input('select', 'value')]
)
def display_descriptor_number(select):
    return "Number of Descriptors: {}".format(select[1]-select[0])

@app.callback(
    Output('violin_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value')]
)
def display_violin_plot(sort, select, descriptors):
    return make_violin_plot(sort, select, descriptors)

@app.callback(
    Output('ebit_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('daq-violin', 'on'),
    Input('family_dropdown', 'value'),]
)
def display_ebit_plot(sort, select, descriptors, switch, families):
    if switch == True:
        select = list(np.arange(select[0],select[1]))
        return make_ebit_plot(production_df, select, sort=sort,
            descriptors=descriptors, family=families)
    else:
        return make_ebit_plot(production_df, family=families)

@app.callback(
    [Output('margin-new-rev', 'children'),
     Output('margin-new-rev-percent', 'children'),
     Output('margin-new-products', 'children')],
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('family_dropdown', 'value'),]
)
def display_opportunity(sort, select, descriptors, families):
    return calculate_margin_opportunity(sort, select, descriptors, families)

if __name__ == "__main__":
    app.run_server(debug=True)
