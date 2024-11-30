import dash
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from collections import Counter

from scipy.stats import shapiro, kstest, normaltest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from layouts import (
    data_cleaning_layout,
    outlier_detection_layout,
    data_transformation_layout,
    normality_test_layout,
    dimensionality_reduction_layout,
    numerical_viz_layout,
    categorical_viz_layout,
    statistical_analysis_layout,
    landing_page_layout
)
###get data
url = 'https://media.githubusercontent.com/media/KaurHarleen1930/apartment_rent/refs/heads/feature/information_visualization/apartments_for_rent_classified_100K.csv'
df = pd.read_csv(url, sep=";", encoding='cp1252',low_memory=False)


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    ]
)


SIDEBAR_ITEMS = [
    {"icon": "fas fa-home", "id": "landing-page", "label": "Project Analysis"},
    {"icon": "fas fa-database", "id": "data-cleaning", "label": "Data Cleaning"},
    {"icon": "fas fa-filter", "id": "outlier-detection", "label": "Outlier Detection"},
    {"icon": "fas fa-server", "id": "data-transformation", "label": "Data Transformation"},
    {"icon": "fas fa-chart-area", "id": "normality-test", "label": "Normality Tests"},
    {"icon": "fas fa-compress-arrows-alt", "id": "dimensionality", "label": "Dimensionality Reduction"},
    {"icon": "fas fa-chart-line", "id": "numerical-viz", "label": "Numerical Features"},
    {"icon": "fas fa-chart-pie", "id": "categorical-viz", "label": "Categorical Features"},
    {"icon": "fas fa-chart-bar", "id": "statistical-ana", "label": "Statistical Analysis"}

]


sidebar = html.Div(
    [

        html.Div([
            html.I(className="fas fa-chart-bar fa-lg"),
            html.Span("Harleen Kaur", className="student-name")
        ],

            className="sidebar-brand",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.I(className=item["icon"] + " fa-lg"),
                        html.Span(item["label"], className="nav-label")
                    ],
                    id=f"sidebar-{item['id']}",
                    className="sidebar-item",
                )
                for item in SIDEBAR_ITEMS
            ],
            className="sidebar-nav"
        )
    ],
    className="sidebar"
)


content = html.Div(id="page-content", className="content")


app.layout = html.Div([
    dcc.Store(id='current-data', storage_type='memory'),
    sidebar,
    content
])


@app.callback(
    Output("page-content", "children"),
    [Input(f"sidebar-{item['id']}", "n_clicks") for item in SIDEBAR_ITEMS],
    [State("current-data", "data")]
)
def render_page_content(*args):
    n_clicks_list = args[:-1]  # All arguments except the last one (State)
    ctx = dash.callback_context

    if not ctx.triggered:
        # Default view
        return landing_page_layout()

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "sidebar-data-cleaning":
        return data_cleaning_layout()
    elif button_id == "sidebar-outlier-detection":
        return outlier_detection_layout()
    elif button_id == "sidebar-dimensionality":
        return dimensionality_reduction_layout()
    elif button_id == "sidebar-numerical-viz":
        return numerical_viz_layout()
    elif button_id == "sidebar-categorical-viz":
        return categorical_viz_layout()
    elif button_id == "sidebar-statistical-ana":
        return statistical_analysis_layout()
    elif button_id == "sidebar-data-transformation":
        return data_transformation_layout()
    elif button_id == "sidebar-normality-test":
        return normality_test_layout()

    # Default fallback
    return landing_page_layout()





def get_apartment_rent_data(df):
    df = df[[
        'id', 'amenities', 'bathrooms', 'bedrooms', 'fee', 'has_photo',
        'pets_allowed', 'price', 'square_feet', 'cityname', 'state',
        'latitude', 'longitude', 'source', 'time'
    ]]

    df = df.astype({
        'amenities': 'string',
        'bathrooms': 'Float32',
        'bedrooms': 'Float32',
        'fee': 'string',
        'has_photo': 'string',
        'pets_allowed': 'string',
        'price': 'Float64',
        'square_feet': 'Int64',
        'latitude': 'Float64',
        'longitude': 'Float64',
        'source': 'string',
        'time': 'Int64'
    })
    apts = pd.DataFrame(df)
    apts['amenities'] = apts["amenities"].fillna("no amenities available")
    apts["pets_allowed"] = apts["pets_allowed"].fillna("None")
    apts['time'] = pd.to_datetime(apts['time'], errors='coerce')
    return apts



@app.callback(
    [Output('original-stats', 'children'),
     Output('original-data', 'data')],
    [Input('feature-type', 'value')]
    )
def update_original_stats(feature_type):
    apts = get_apartment_rent_data(df)


    numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet', 'latitude', 'longitude']
    categorical_cols = ['cityname', 'state', 'amenities', 'fee', 'has_photo', 'pets_allowed', 'source']

    numerical_df = apts[numerical_cols]
    categorical_df = apts[categorical_cols]


    data_store = {
        'numerical': numerical_df.to_dict('records'),
        'categorical': categorical_df.to_dict('records')
    }


    if feature_type == "numerical":
        working_df = numerical_df
    else:
        working_df = categorical_df

    missing_vals = working_df.isnull().sum()
    original_table = html.Table(
        [html.Tr([html.Th("Column"), html.Th("Original Missing Values")])] +
        [html.Tr([html.Td(col), html.Td(str(missing_vals[col]))])
         for col in working_df.columns],
        className="table-auto w-full"
    )
    return original_table, data_store

@app.callback(
    [Output("cleaning-results", "children"),
     Output("missing-values-table", "children"),
     Output("method-warning", "children"),
     Output("cleaned-state", "data")],
    Input("apply-cleaning", "n_clicks"),
    [ State("feature-type", "value"),
     State("cleaning-method", "value"),
    State('original-data', 'data'),
     State('cleaned-state', 'data')]
)
def update_cleaning_results(n_clicks, feature_type, cleaning_method, original_data, cleaned_state):
    if n_clicks is None or original_data is None:
        raise PreventUpdate


    if feature_type == "categorical":
        working_df = pd.DataFrame(original_data['categorical'])
    else:
        working_df= pd.DataFrame(original_data['numerical'])



    try:
        if cleaning_method == "display":
            if cleaned_state is None:
                working_df = pd.DataFrame(original_data)
            else:
                working_df = pd.DataFrame(cleaned_state)

                # Apply cleaning method
        elif cleaning_method == "remove":
            working_df = working_df.dropna()
        elif cleaning_method == "duplicate":
            working_df = working_df.drop_duplicates()
        elif cleaning_method == "mean" and feature_type == "numerical":
            working_df = working_df.fillna(working_df.mean())
        elif cleaning_method == "median" and feature_type == "numerical":
            working_df = working_df.fillna(working_df.median())
        elif cleaning_method == "mode":
            working_df = working_df.fillna(working_df.mode().iloc[0])
        elif cleaning_method == "ffill":
            working_df = working_df.fillna(method='ffill')
        elif cleaning_method == "bfill":
            working_df = working_df.fillna(method='bfill')

        if cleaning_method in ['mean', 'median'] and feature_type == "categorical":
            return dash.no_update, dash.no_update, "Cannot apply numerical methods to categorical data", None

    except Exception as e:
        return dash.no_update, dash.no_update, f"Error applying cleaning method: {str(e)}", None

    # Create results table
    missing_vals = working_df.isnull().sum()
    results_table = html.Table(
        [html.Tr([html.Th("Column"), html.Th("Missing Values")])] +
        [html.Tr([html.Td(col), html.Td(str(missing_vals[col]))])
         for col in working_df.columns],
        className="table-auto w-full"
    )



    new_cleaned_state = cleaned_state if cleaned_state is not None else {}
    new_cleaned_state[feature_type] = working_df.to_dict('records')

    summary = html.Div([
        html.P(f"Selected Feature Type: {feature_type}"),
        html.P(f"Applied Cleaning Method: {cleaning_method}"),
        html.P(f"Total rows: {len(working_df)}"),
        html.P(f"Total missing values: {missing_vals.sum()}")
    ])
    print(f"Selected Feature Type: {feature_type}"),
    print(f"Applied Cleaning Method: {cleaning_method}"),
    print(f"Total rows: {len(working_df)}"),
    print(f"Total missing values: {missing_vals.sum()}")

    return results_table, summary, "", cleaned_state

######Outlier Detection########
#######################################################################################################################


def perform_data_cleaning(apts):
    apts = apts[apts['bathrooms'].notna()]
    apts = apts[apts['bedrooms'].notna()]
    apts = apts[apts['price'].notna()]
    apts = apts[apts['latitude'].notna()]
    apts = apts[apts['longitude'].notna()]
    apts = apts[apts['cityname'].notna()]
    apts = apts[apts['state'].notna()]
    apts = apts[apts.state != 0]

    apts['state'] = apts['state'].astype("string")
    apts['cityname'] = apts['cityname'].astype("string")
    apts['time'] = pd.to_datetime(apts['time'], errors='coerce')
    return apts


def createBoxPlot(apts, features):
    fig = px.box(data_frame=apts, y = features, template='plotly_dark',
                 title = f'Box plot of feature: {features}',
                 points = 'outliers')
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    return fig


def createScatterPlot(apts, features):
    fig = px.scatter(data_frame=apts, y = features, template='plotly_dark',
                     title = f'Scatter plot of feature: {features}',
                     )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    return fig


def createKdePlot(apts, features):
    fig = px.histogram(data_frame=apts, x = features, template='plotly_dark',
                       title = f'KDE plot of feature: {features}',
                       marginal='violin',
                       nbins=100,)
    fig.update_layout(
        height=400,
        margin = dict(l=0,r=0,t=40,b=0),
        showlegend = False
    )
    return fig


def createViolinPlot(apts, features):
    fig = px.violin(data_frame=apts, y = features, template='plotly_dark',
              title = f'Violin plot of feature: {features}',
              box=True,
              points = 'outliers'
              )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend = False
    )
    return fig
#########callbacks############
def remove_outlier_method(apts, method, feature):
    if method=="iqr":
        Q1 = apts[feature].quantile(0.25)
        Q3 = apts[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_quantile_range = Q1 - 1.5 * IQR
        upper_quantile_range = Q3 + 1.5 * IQR

        outlier_mask = (apts[feature] >= lower_quantile_range) & (apts[feature] <= upper_quantile_range)
        outliers_count = (~outlier_mask).sum()
        clean_data = apts[outlier_mask].copy()

        statics = {
            'total_records': len(apts),
            'outliers_removed': outliers_count,
            'percentage_outliers': (outliers_count / len(apts)) * 100,
            'lower_bound': lower_quantile_range,
            'upper_bound': upper_quantile_range
        }
        return clean_data, statics
    elif method=="zscore":
        zscore_stat = stats.zscore(apts[feature])
        z_score = abs(zscore_stat)
        outlier_mask = z_score<3
        outliers_count = (~outlier_mask).sum()
        clean_data = apts[outlier_mask].copy()
        statics = {
            'total_records': len(apts),
            'outliers_removed': outliers_count,
            'percentage_outliers': (outliers_count / len(apts)) * 100,
        }
        return clean_data, statics



@app.callback(
    [Output("outlier-summary", "children"),
     Output("current-data-state", "data")],
    Input("detect-outliers","n_clicks"),
    [State("outlier-method","value"),
     State("numerical-feature-dropdown","value"),]
)
def detect_outliers(n_clicks, method, feature):
    if n_clicks is None or not feature:
        raise PreventUpdate
    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)
    try:
        clean_data, statics = remove_outlier_method(apts, method, feature)

        stats_display = html.Div([
            html.H4("Outlier Detection Statistics"),
            html.P(f"Total records: {statics['total_records']}"),
            html.P(f"Outliers removed: {statics['outliers_removed']}"),
            html.P(f"Percentage of outliers: {statics['percentage_outliers']:.2f}%"),
            html.P(f"Method used: {method.upper()}")
        ])

        if method == 'iqr':
            stats_display.children.extend([
                html.P(f"Lower bound: {statics['lower_bound']:.2f}"),
                html.P(f"Upper bound: {statics['upper_bound']:.2f}")
            ])

        return stats_display,clean_data.to_dict('records')

    except Exception as e:
        print(f"Error in Outlier Detection: {str(e)}")
        return html.Div([
            html.H4("Error in Outlier Detection"),
            html.P(f"Error: {str(e)}")
        ]), None

@app.callback(
    Output("original-plots", "children"),
    Input("show-outlier-plots", "n_clicks"),
    [State("numerical-feature-dropdown","value"),
    State("select-plot-type","value"),
     State("current-data-state","data")
    ]
)
def show_outlier_plot(n_clicks, features, plot_types, current_data):
    if n_clicks is None or not features or not plot_types:
        raise PreventUpdate

    try:
        if current_data is None:
            apts = get_apartment_rent_data(df)
            apts = perform_data_cleaning(apts)
        else:
            apts = pd.DataFrame.from_dict(current_data)
        plots = []
        for plot_type in plot_types:
            if plot_type == 'boxplot':
                fig_box = createBoxPlot(apts, features)
                plots.append(dcc.Graph(figure=fig_box, className = "mb-4"))
            elif plot_type == 'scatterplot':
                fig_scatter = createScatterPlot(apts, features)
                plots.append(dcc.Graph(figure=fig_scatter, className = "mb-4"))
            elif plot_type == 'kdeplot':
                fig_kde = createKdePlot(apts, features)
                plots.append(dcc.Graph(figure=fig_kde, className = "mb-4"))
            elif plot_type == 'violinplot':
                violin_plot = createViolinPlot(apts, features)
                plots.append(dcc.Graph(figure=violin_plot, className = "mb-4"))

        return html.Div(plots)
    except Exception as e:
        return html.Div([
            html.H4("Error in Plot Generation"),
            html.P(f"Error: {str(e)}")
        ])
############################################################################################
#####################Data Transformation
############################################################################################

@app.callback(
    [Output("transform-distributions", "figure"),
     Output("transform-stats", "children")],
    [Input("apply-transforms-btn", "n_clicks")],
    [State("transform-feature-select", "value"),
     State("transform-methods", "value")]
)
def update_transformations(n_clicks, feature, methods):
    if n_clicks is None:
        raise PreventUpdate


    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)


    feature_data = apts[feature].dropna()

    n_plots = 1 + len(methods)
    fig = make_subplots(rows=n_plots, cols=1,
                        subplot_titles=['Original Data'] + methods,
                        vertical_spacing=0.05)


    all_data = {'Original': feature_data}


    fig.add_trace(
        go.Histogram(x=feature_data, name='Original',
                     nbinsx=50, histnorm='probability'),
        row=1, col=1
    )


    current_row = 2
    for method in methods:
        try:
            if method == 'standardization':
                scaler = StandardScaler()
                transformed = scaler.fit_transform(feature_data.values.reshape(-1, 1)).flatten()
                all_data['Standardized'] = transformed

            elif method == 'normalization':
                scaler = MinMaxScaler()
                transformed = scaler.fit_transform(feature_data.values.reshape(-1, 1)).flatten()
                all_data['Normalized'] = transformed

            elif method == 'log':
#log1p for zeros
                transformed = np.log1p(feature_data)
                all_data['Log'] = transformed

            fig.add_trace(
                go.Histogram(x=transformed, name=f'Graph for method: {method}',
                             nbinsx=50, histnorm='probability',
                             ),
                row=current_row, col=1
            )

            current_row += 1

        except Exception as e:
            print(f"Error in {method} transformation: {str(e)}")

    fig.update_layout(
        height=400 * n_plots,
        showlegend=False,
        template='plotly_dark',
        title_text=f"Distribution Comparison for {feature}"
    )


    stats_dict = {}
    for name, data in all_data.items():
        stats_dict[name] = {
            'Mean': np.mean(data),
            'Std': np.std(data),
            'Min': np.min(data),
            'Q1': np.percentile(data, 25),
            'Median': np.percentile(data, 50),
            'Q3': np.percentile(data, 75),
            'Max': np.max(data),
            'Skewness': pd.Series(data).skew(),
            'Kurtosis': pd.Series(data).kurtosis()
        }


    stats_df = pd.DataFrame(stats_dict).round(3)


    stats_table = html.Table([

        html.Thead(
            html.Tr([html.Th("Metric")] + [
                html.Th(col) for col in stats_df.columns
            ])
        ),

        html.Tbody([
            html.Tr([
                html.Td(index),
                *[html.Td(f"{stats_df.loc[index, col]:.3f}" if not pd.isna(stats_df.loc[index, col]) else "N/A")
                  for col in stats_df.columns]
            ]) for index in stats_df.index
        ])
    ], className="transform-stats-table")

    return fig, stats_table
############################################################################################
#####################Normality Tests
############################################################################################

@app.callback(
[Output("normality-plots", "figure"),
         Output("normality-results", "children")],
        [Input("run-normality-tests-btn", "n_clicks")],
        [State("normality-feature-select", "value"),
         State("normality-tests", "value")]
)
def run_normality_tests(n_clicks, feature, tests):
    if n_clicks is None:
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    data = apts[feature].dropna()

    z_scores = (data - data.mean()) / data.std()

    # Create subplots
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Q-Q Plot", "Distribution with Normal Curve"],
                        specs=[[{"type": "scatter"}], [{"type": "histogram"}]])

    # Create Q-Q plot with standardized data
    sorted_z_scores = np.sort(z_scores)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_z_scores)))

    # Add Q-Q plot
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_z_scores,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(
                color='lightblue',
                size=3,
            )
        ),
        row=1, col=1
    )

    # Add reference line
    min_val = min(theoretical_quantiles)
    max_val = max(theoretical_quantiles)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Reference Line',
            line=dict(
                color='magenta',
                dash='dash',
                width=2
            )
        ),
        row=1, col=1
    )

    # Update Q-Q plot layout
    fig.update_xaxes(
        title_text="Theoretical Quantiles",
        range=[-3, 3],  # Set fixed range for x-axis
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Sample Quantiles",
        range=[-3, 3],  # Set fixed range for y-axis
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=z_scores,
            name='Data Distribution',
            nbinsx=50,
            histnorm='probability density',
        ),
        row=2, col=1
    )
    fig.update_traces(marker=dict(opacity=0.8, color='cyan'))

    # Add normal curve
    x_range = np.linspace(-3, 3, 100)
    normal_curve = stats.norm.pdf(x_range, 0, 1)

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name='Normal Curve',
            line=dict(color='magenta', width=3),

        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"Normality Analysis for {feature}",
        height=600,
        width=None,
        margin=dict(
            l=50,
            r=50,
            t=50,
            b=50
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        # plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
    )

    # Update subplot properties
    fig.update_xaxes(showgrid=True, gridwidth=1,)
    fig.update_yaxes(showgrid=True, gridwidth=1)


    test_results = []
    alpha = 0.01

    for test in tests:
        if test == 'shapiro':
            stats_val, p_val = shapiro(data)
            result = {
                'test': 'Shapiro-Wilk Test',
                'statistic': stats_val,
                'p_value': p_val,
                'is_normal': p_val > alpha
            }
            test_results.append(result)

        elif test == 'ks':

            normal_dist = np.random.normal(data.mean(), data.std(), len(data))
            stats_val, p_val = kstest(data, normal_dist)
            result = {
                'test': 'Kolmogorov-Smirnov Test',
                'statistic': stats_val,
                'p_value': p_val,
                'is_normal': p_val > alpha
            }
            test_results.append(result)

        elif test == 'dagostino':
            stats_val, p_val = normaltest(data)
            result = {
                'test': "D'Agostino-Pearson Test",
                'statistic': stats_val,
                'p_value': p_val,
                'is_normal': p_val > alpha
            }
            test_results.append(result)


    results_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Test"),
            html.Th("Statistic"),
            html.Th("P-Value"),
            html.Th("Result")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(result['test']),
                html.Td(f"{result['statistic']:.4f}"),
                html.Td(f"{result['p_value']:.4f}"),
                html.Td(
                    "Normal" if result['is_normal'] else "Not Normal",
                    style={'color': '#22c55e' if result['is_normal'] else '#ef4444'}
                )
            ]) for result in test_results
        ])
    ], className="normality-test-table")

    return fig, results_table

############################################################################################
#####################Numerical Visualization
############################################################################################

def remove_outliers(apts):
    numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']
    Q1 = apts[numerical_cols].quantile(0.25)
    Q3 = apts[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_quantile_range = Q1 - 1.5 * IQR
    upper_quantile_range = Q3 + 1.5 * IQR

    apts = apts[~((apts[numerical_cols] < lower_quantile_range) | (apts[numerical_cols] > upper_quantile_range)).any(axis=1)]
    return apts


def createHistogramPlot(apts, features):
    fig = px.histogram(
        df, x=features,
        template="plotly_dark",
        title=f"Histogram of {features}",
        marginal="box"  # adds box plot to the histogram
    )
    return fig


# def cal_kde_plot(apts, feature):
#     apts= apts[feature].tolist()
#     fig = ff.create_distplot(
#         [apts],
#         [feature],
#         show_hist=False,
#         show_rug=True
#     )
#     fig.update_layout(
#         template="plotly_dark",
#         title=f"KDE Plot of {feature}"
#     )


@app.callback(
    [Output("univariate-plots","children"),
    Output("correlation-stats-uni","children")],
    Input("uni-plot-button", "n_clicks"),
    [State("uni-feature-selector","value"),
     State("uni-plot-selector","value")]
)
def generate_uni_numerical_feature_plot(n_clicks, features, plot_type):
    if n_clicks is None or not features or not plot_type:
        raise PreventUpdate
    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)
    apts = remove_outliers(apts)
    plots=[]
    for plot_type in plot_type:
        if plot_type == 'box':
            fig = createBoxPlot(apts, features)

        elif plot_type == 'histogram':
            fig = createHistogramPlot(apts, features)
        elif plot_type == 'kde':
            fig = createKdePlot(apts, features)
        elif plot_type == 'violin':
            fig = createViolinPlot(apts, features)

        fig.update_layout(
        height =400
        )
        plots.append(dcc.Graph(figure=fig, className="mb-4"))

    stats_div = html.Div([
        html.H4("Descriptive Statistics", className="mt-4"),
        html.P(f"Mean: {df[features].mean():.2f}"),
        html.P(f"Median: {df[features].median():.2f}"),
        html.P(f"Std Dev: {df[features].std():.2f}"),
        html.P(f"Skewness: {df[features].skew():.2f}")
    ])

    return html.Div(plots),stats_div

@app.callback(
   [ Output("bivariate-plots","children"),
    Output("correlation-stats-bi","children")],
    Input("bi-plot-button", "n_clicks"),
    [State("bi-feature-selector-1","value"),
    State("bi-feature-selector-2","value"),
     State("bi-plot-selector","value")]
)
def generate_bivariate_feature_plot(n_clicks, feature1,feature2, plot_type):
    if n_clicks is None or not feature1 or not feature2 or not plot_type:
        raise PreventUpdate
    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)
    apts = remove_outliers(apts)
    plots = []
    for plot_type in plot_type:
        if plot_type == "scatter":
            fig = px.scatter(
                df, x=feature1, y=feature2,
                template="plotly_dark",
                title=f"Scatter Plot: {feature1} vs {feature2}",
                trendline="ols"  # adds trend line
            )

        elif plot_type == "hexbin":
            fig = px.density_heatmap(
                df, x=feature1, y=feature2,
                template="plotly_dark",
                title=f"Hexbin Plot: {feature1} vs {feature2}",
                nbinsx=30, nbinsy=30,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                plot_bgcolor="rgba(17, 17, 17, 0.9)",  # Plot background
                paper_bgcolor="rgba(17, 17, 17, 0.9)",  # Paper background
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                coloraxis_colorbar=dict(
                    title="Count",
                    tickformat=",.0f"
                )
            )
            fig.update_traces(
                colorbar=dict(
                    tickfont=dict(color='white'),
                    titlefont=dict(color='white')
                )
            )

        elif plot_type == "density":
            fig = px.density_contour(
                df, x=feature1, y=feature2,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.G10,
                title=f"Density Contour: {feature1} vs {feature2}"
            )

        elif plot_type == "histogram2d":
            fig = px.density_heatmap(
                df, x=feature1, y=feature2,
                template="plotly_dark",
                title=f"2D Histogram: {feature1} vs {feature2}",
                marginal_x="histogram",
                marginal_y="histogram"
            )

        elif plot_type == "cluster":
            n_clusters = 5
            coords = apts[[feature1, feature2]]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            apts['cluster'] = kmeans.fit_predict(coords)

            # Create the scatter map
            fig = px.scatter_mapbox(
                apts,
                lat='latitude',
                lon='longitude',
                color='cluster',
                color_continuous_scale='Viridis',
                zoom=3,
                title="Apartment Locations by Cluster",
                hover_data=['price', 'bedrooms', 'bathrooms'],
                category_orders={'cluster': sorted(apts['cluster'].unique())},
                labels={'cluster': 'Cluster'}
            )


            fig.update_layout(
                mapbox_style="carto-darkmatter",
                height=600,
                margin=dict(l=0, r=0, t=30, b=0)
            )

        fig.update_layout(height=400)
        plots.append(dcc.Graph(figure=fig, className="mb-4"))

    pearson_corr = df[feature1].corr(df[feature2], method='pearson')
    spearman_corr = df[feature1].corr(df[feature2], method='spearman')

    correlation_stats = html.Div([
        html.H4("Correlation Analysis"),
        html.P(f"Pearson Correlation: {pearson_corr:.3f}"),
        html.P(f"Spearman Correlation: {spearman_corr:.3f}")
    ])

    return html.Div(plots), correlation_stats


@app.callback(
    Output("cat-uni-plots", "children"),
    [Input("cat-uni-button", "n_clicks")],
    [State("cat-uni-feature", "value"),
     State("cat-uni-plot-type", "value")]
)
def update_univariate_plots(n_clicks, feature, plot_types):
    if n_clicks is None or not feature or not plot_types:
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)
    if feature=='amenities':


        # Split the amenities column and count each unique amenity
        amenities_list = apts['amenities'].str.split(',').sum()
        amenities_count = Counter(amenities_list).most_common(15)
        dfa  = pd.DataFrame(amenities_count)
        x = dfa[0]
        y=dfa[1]
    else:
        value_counts = apts[feature].value_counts()
        x = value_counts.index
        y=value_counts.values
    plots = []


    for plot_type in plot_types:
        try:
            if plot_type == "bar":
                fig = px.bar(
                    x=x,
                    y=y,
                    title=f"Distribution of {feature}",
                    labels={'x': feature, 'y': 'Count'},
                    template="plotly_dark"
                )

            elif plot_type == "pie":
                fig = px.pie(
                    values=y[:15],
                    names=x[:15],
                    title=f"Distribution of {feature}",
                    template="plotly_dark"
                )

            elif plot_type == "treemap":
                fig = px.treemap(
                    names=x,
                    parents=[""] * len(y),
                    values=y,
                    title=f"Distribution of {feature}",
                    template="plotly_dark"
                )

            elif plot_type == "table":
                freq_df = pd.DataFrame({
                    'Category': x,
                    'Count': y,
                    'Percentage': (y / len(apts) * 100).round(2)
                })

                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=list(freq_df.columns),
                        fill_color='#1e293b',
                        align='left',
                        font=dict(color='white')
                    ),
                    cells=dict(
                        values=[freq_df[col] for col in freq_df.columns],
                        fill_color='#0f172a',
                        align='left',
                        font=dict(color='white'),
                        format=[None, ",", ".2f"]
                    )
                )])
                fig.update_layout(title=f"Frequency Table for {feature}")

            if plot_type != "table":
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

            plots.append(dcc.Graph(figure=fig, className="mb-4"))

        except Exception as e:
            plots.append(html.Div([
                html.H4(f"Error in {plot_type} plot", className="text-danger"),
                html.P(f"Error: {str(e)}")
            ]))

    return html.Div(plots)


# @app.callback(
#     Output("cat-bi-plots", "children"),
#     Input("cat-bi-button", "n_clicks"),
#     [State("cat-bi-feature1", "value"),
#      State("cat-bi-feature2", "value"),
#      State("cat-bi-plot-type", "value")]
# )
# def update_bivariate_plots(n_clicks, feature1, feature2, plot_types):
#     if n_clicks is None or not feature1 or not feature2 or not plot_types:
#         raise PreventUpdate
#
#     apts = get_apartment_rent_data(df)
#     apts = perform_data_cleaning(apts)
#
#
#     # Create contingency table
#     cont_table = pd.crosstab(apts[feature1], apts[feature2], margins=True)
#     plots = []
#     custom_colors = [
#         '#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC',
#         '#99CCFF', '#FFB366', '#99FF99', '#FF99FF', '#66FFB2',
#         '#FF8533', '#99FFCC', '#FF99B2', '#99FFE6', '#FFB299',
#         '#B2FF99', '#FF99E6', '#99FFFF', '#FFE699', '#B299FF'
#     ]
#     for plot_type in plot_types:
#         try:
#             top_20_values = apts[feature2].value_counts().nlargest(20).index
#             fileterd = apts[apts[feature2].isin(top_20_values)]
#             if plot_type == "stacked_bar":
#                 fig = px.bar(
#                     apts,
#                     x=feature1,
#                     color=feature2,
#                     title=f"Stacked Bar Chart: {feature1} vs {feature2}",
#                     template="plotly_dark",
#                     color_discrete_sequence=px.colors.qualitative.G10,
#                 )
#
#             elif plot_type == "grouped_bar":
#                 fig = px.bar(
#                     apts,
#                     x=feature1,
#                     color=feature2,
#                     barmode="group",
#                     title=f"Grouped Bar Chart: {feature1} vs {feature2}",
#                     template="plotly_dark",
#                     color_discrete_sequence=px.colors.qualitative.G10,
#                 )
#
#             elif plot_type == "heatmap":
#                 heat_data = cont_table.iloc[:-1, :-1]
#                 fig = px.imshow(
#                     heat_data,
#                     title=f"Heatmap: {feature1} vs {feature2}",
#                     template="plotly_dark",
#                     color_continuous_scale="Viridis"
#                 )
#
#             elif plot_type == "sunburst":
#                 fig = px.sunburst(
#                     apts,
#                     path=[feature1, feature2],
#                     title=f"Sunburst Chart: {feature1} and {feature2}",
#                     template="plotly_dark"
#                 )
#
#             elif plot_type == "table":
#                 fig = go.Figure(data=[go.Table(
#                     header=dict(
#                         values=[feature1] + list(cont_table.columns),
#                         fill_color='#1e293b',
#                         align='left',
#                         font=dict(color='white')
#                     ),
#                     cells=dict(
#                         values=[cont_table.index] + [cont_table[col] for col in cont_table.columns],
#                         fill_color='#0f172a',
#                         align='left',
#                         font=dict(color='white'),
#                         format=[None] + [","] * len(cont_table.columns)
#                     )
#                 )])
#                 fig.update_layout(title=f"Contingency Table: {feature1} vs {feature2}")
#
#             if plot_type != "table":
#                 fig.update_layout(
#                     height=500,
#                     margin=dict(l=20, r=20, t=40, b=20)
#                 )
#
#             plots.append(dcc.Graph(figure=fig, className="mb-4"))
#
#         except Exception as e:
#             plots.append(html.Div([
#                 html.H4(f"Error in {plot_type} plot", className="text-danger"),
#                 html.P(f"Error: {str(e)}")
#             ]))
#     return html.Div(plots)

###############Statistical ANalysis################

@app.callback(
    Output("state-price-count-plot", "figure"),
    Input("analysis-tabs", "active_tab")
)
def update_state_plot(active_tab):
    if active_tab != "state-tab":
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    state_means = apts.groupby('state')['price'].mean().reset_index()
    state_means.sort_values(by=['price'], ascending=False, inplace=True)
    state_counts = apts.groupby('state').size().reset_index(name='count')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            name="Average Price",
            x=state_means['state'],
            y=state_means['price'],
            marker_color='#4A90E2',
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            name="Apartment Count",
            x=state_counts['state'],
            y=state_counts['count'],
            mode='markers',
            marker=dict(size=10, color='#FF9F1C')
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="State-wise Price Analysis and Apartment Count",
        template="plotly_dark",
        height=600
    )

    return fig

#####state city analysis

@app.callback(
    Output('state-selector', 'options'),
    Input('analysis-tabs', 'active_tab')
)
def populate_states(active_tab):
    if active_tab != "city-state-tab":
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    states = apts['state'].unique()
    return [{'label': state, 'value': state} for state in states]


@app.callback(
    [Output('city-plots', 'children'),
     Output('city-stats', 'children')],
    [Input('state-selector', 'value'),
     Input('plot-type-selector', 'value'),
     Input('analysis-tabs', 'active_tab')]
)
def update_city_analysis(state, plot_types, active_tab):
    if active_tab != "city-state-tab" or not state or not plot_types:
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)
    state_df = apts[apts['state'] == state]
    plots = []

    # Calculate city statistics
    city_stats = state_df.groupby('cityname').agg({
        'price': ['mean', 'median', 'std', 'count'],
        'square_feet': 'mean',
        'bedrooms': 'mean',
        'bathrooms': 'mean'
    }).round(2)

    # Reset index for sorting
    city_stats.columns = ['avg_price', 'median_price', 'price_std', 'listings', 'avg_sqft', 'avg_beds', 'avg_baths']
    city_stats = city_stats.reset_index()

    for plot_type in plot_types:
        if plot_type == 'price_dist':
            # Box plot of prices by city
            fig = go.Figure()
            for city in city_stats.nlargest(10, 'avg_price')['cityname']:
                city_data = state_df[state_df['cityname'] == city]['price']
                fig.add_trace(go.Box(
                    y=city_data,
                    name=city,
                    boxpoints='outliers'
                ))

            fig.update_layout(
                title=f"Price Distribution for Top 10 Most Expensive Cities in {state}",
                template="plotly_dark",
                height=500,
                yaxis_title="Price ($)",
                showlegend=True
            )
            plots.append(dcc.Graph(figure=fig))

        elif plot_type == 'rankings':
            # Bar chart of average prices
            # fig = go.Figure()

            sorted_cities = city_stats.sort_values('avg_price', ascending=True)
            costliest_cities = sorted_cities.nlargest(10, 'avg_price')
            cheapest_cities = sorted_cities.nsmallest(10, 'avg_price')
            display_cities = pd.concat([costliest_cities, cheapest_cities]).reset_index()
            display_cities['Category'] = ['Most Expensive'] * 10 + ['Cheapest'] * 10
            fig = px.bar(
                display_cities,
                x=display_cities['avg_price'],
                y=display_cities['cityname'],
                color='Category',
                orientation='h',
                color_discrete_sequence=px.colors.qualitative.Bold,
            )

            fig.update_layout(
                title=f"City Rankings by Average Price in {state}",
                template="plotly_dark",
                height=max(100, len(city_stats) * 8),
                xaxis_title="Average Price ($)",
                yaxis_title="City"
            )
            plots.append(dcc.Graph(figure=fig))

        elif plot_type == 'listings':
            # Scatter plot of average price vs number of listings
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=city_stats['listings'],
                y=city_stats['avg_price'],
                mode='markers',
                text=city_stats['cityname'],
                textposition="top center",
                marker=dict(
                    size=city_stats['listings'] / city_stats['listings'].max() * 50,
                    color=city_stats['avg_price'],
                    colorscale='plotly3',
                    showscale=True
                )
            ))

            fig.update_layout(
                title=f"Price vs Number of Listings in {state}",
                template="plotly_dark",
                height=600,
                xaxis_title="Number of Listings",
                yaxis_title="Average Price ($)"
            )
            plots.append(dcc.Graph(figure=fig))

    # Create statistics table
    stats_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("City", className="col-city"),
                html.Th("Avg Price", className="col-price"),
                html.Th("Median", className="col-price"),
                html.Th("#", className="col-listings"),
                html.Th("Sqft", className="col-metrics"),
                html.Th("Beds", className="col-metrics"),
                html.Th("Baths", className="col-metrics"),
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row['cityname']),
                html.Td(f"${row['avg_price']:,.0f}"),
                html.Td(f"${row['median_price']:,.0f}"),
                html.Td(f"{row['listings']}"),
                html.Td(f"{row['avg_sqft']:,.0f}"),
                html.Td(f"{row['avg_beds']:.1f}"),
                html.Td(f"{row['avg_baths']:.1f}")
            ]) for _, row in city_stats.iterrows()
        ])
    ], className="city-stats-table")

    stats_summary = html.Div([
        html.H5(f"Summary for {state}:", className="mb-3"),
        html.Div([
            html.P([
                "Total Cities: ",
                html.Span(f"{len(city_stats)}", className="highlight-value")
            ]),
            html.P([
                "Total Listings: ",
                html.Span(f"{city_stats['listings'].sum():,}", className="highlight-value")
            ]),
            html.P([
                "Most Expensive: ",
                html.Span(
                    f"{city_stats.iloc[city_stats['avg_price'].argmax()]['cityname']} "
                    f"(${city_stats['avg_price'].max():,.0f})",
                    className="highlight-value"
                )
            ]),
            html.P([
                "Least Expensive: ",
                html.Span(
                    f"{city_stats.iloc[city_stats['avg_price'].argmin()]['cityname']} "
                    f"(${city_stats['avg_price'].min():,.0f})",
                    className="highlight-value"
                )
            ])
        ], className="city-stats-summary"),
        html.H5("Detailed City Statistics:", className="mb-3"),
        html.Div([
            stats_table
        ], className="stats-card")
    ])
    return html.Div(plots), stats_summary



@app.callback(
    [Output("bedroom-bathroom-plot", "figure"),
     Output("bedbath-ratio-plot", "figure")],
    Input("analysis-tabs", "active_tab")
)
def update_bedroom_plots(active_tab):
    if active_tab != "bedroom-tab":
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    bed_bath = apts.groupby('state').agg({
        'bedrooms': 'mean',
        'bathrooms': 'mean',
        'price': 'mean'
    }).reset_index()
    bed_bath.sort_values(by=['price'], ascending=False, inplace=True)
    # First plot
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    fig1.add_trace(
        go.Bar(
            name="Bedrooms",
            x=bed_bath['state'],
            y=bed_bath['bedrooms'],
            marker_color='rgb(158,202,225)'
        ),
        secondary_y=False
    )

    fig1.add_trace(
        go.Scatter(
            name="Bathrooms",
            x=bed_bath['state'],
            y=bed_bath['bathrooms'],
            mode='markers',
            marker=dict(size=10, color='rgb(255,107,107)')
        ),
        secondary_y=False
    )

    fig1.add_trace(
        go.Scatter(
            name="Price",
            x=bed_bath['state'],
            y=bed_bath['price'],
            mode='lines',
            line=dict(color='rgb(50,205,50)')
        ),
        secondary_y=True
    )

    fig1.update_layout(
        title="State-wise Bedroom, Bathroom, and Price Analysis",
        template="plotly_dark",
        height=600
    )

    # Second plot
    bed_bath['ratio'] = bed_bath['bedrooms'] / bed_bath['bathrooms']
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    bed_bath.sort_values(by=['ratio'], ascending=False, inplace=True)
    fig2.add_trace(
        go.Bar(
            name="Bed/Bath Ratio",
            x=bed_bath['state'],
            y=bed_bath['ratio'],
            marker_color='rgb(158,202,225)'
        ),
        secondary_y=False
    )

    fig2.add_trace(
        go.Scatter(
            name="Price",
            x=bed_bath['state'],
            y=bed_bath['price'],
            mode='markers',
            marker=dict(size=10, color='rgb(255,107,107)')
        ),
        secondary_y=True
    )

    fig2.update_layout(
        title="Bedroom-Bathroom Ratio Analysis",
        template="plotly_dark",
        height=600
    )

    return fig1, fig2


# Square Feet Analysis Callback
@app.callback(
    Output("sqft-price-plot", "figure"),
    Input("analysis-tabs", "active_tab")
)
def update_sqft_plot(active_tab):
    if active_tab != "sqft-tab":
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    sqft = apts.groupby('state').agg({
        'square_feet': 'mean',
        'price': 'mean'
    }).reset_index()

    sqft['price_per_sqft'] = sqft['price'] / sqft['square_feet']
    sqft.sort_values(by=['price_per_sqft'], ascending=False, inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            name="Price per Sqft",
            x=sqft['state'],
            y=sqft['price_per_sqft'],
            marker_color='rgb(158,202,225)'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            name="Square Feet",
            x=sqft['state'],
            y=sqft['square_feet'],
            mode='markers',
            marker=dict(size=10, color='rgb(255,107,107)')
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Price per Square Foot Analysis",
        template="plotly_dark",
        height=600
    )

    return fig
#City Analysis

@app.callback(
    Output("city-tab-plot", "figure"),
    Input("analysis-tabs", "active_tab")
)
def update_city_plot(active_tab):
    if active_tab != "city-tab":
        raise PreventUpdate
    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)
    city_prices = apts.groupby('cityname')['price'].mean()
    most_expensive = city_prices.nlargest(10)
    least_expensive = city_prices.nsmallest(10)
    top_cities = pd.concat([most_expensive, least_expensive]).reset_index()
    top_cities['Category'] = ['Most Expensive'] * 10 + ['Cheapest'] * 10
    top_cities = top_cities.sort_values(by='price', ascending=False)
    fig = px.bar(top_cities, x='price', y='cityname', color='Category',
                 title=f"Top 10 Most Expensive and Cheapest Cities by Avg Rental Price",
                 template="plotly_dark",
                 height=600
                 )
    fig.update_xaxes(title_text="Avg Rental Price")
    fig.update_yaxes(title_text="City")
    return fig




# Correlation Analysis Callback
@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("analysis-tabs", "active_tab")
)
def update_correlation_plot(active_tab):
    if active_tab != "correlation-tab":
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']
    correlation = apts[numerical_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation,
        x=numerical_cols,
        y=numerical_cols,
        text=np.around(correlation, decimals=2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='RdBu'
    ))

    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_dark",
        height=600
    )

    return fig

#############################
###Dimensionality Reduction
############################

@app.callback(
    [Output("scree-plot", "figure"),
     Output("cumulative-variance-plot", "figure"),
     Output("pca-plot", "figure"),
     Output("loading-plot", "figure")],
    [Input("apply-pca-btn", "n_clicks")],
    [State("feature-selector", "value"),
     State("color-feature", "value")]
)
def update_pca_analysis(n_clicks, features, color_by):
    if n_clicks is None or not features:
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    # Prepare data for PCA
    X = apts[features].copy()
    X = X.dropna()
    apts = apts.loc[X.index]
    # Add categorical features for coloring
    if color_by == 'price_range':
        apts['price_range'] = pd.qcut(apts['price'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    elif color_by == 'size_category':
        apts['size_category'] = pd.qcut(apts['square_feet'], q=3, labels=['Small', 'Medium', 'Large'])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)

    # Create Scree Plot
    exp_var_ratio = pca.explained_variance_ratio_
    scree_fig = go.Figure()
    scree_fig.add_trace(go.Bar(
        x=[f'PC{i + 1}' for i in range(len(exp_var_ratio))],
        y=exp_var_ratio,
        name='Explained Variance'
    ))
    scree_fig.update_layout(
        template='plotly_dark',
        title='Explained Variance Ratio by Component',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio'
    )

    # Create Cumulative Variance Plot
    cum_var_ratio = np.cumsum(exp_var_ratio)
    cum_var_fig = go.Figure()
    cum_var_fig.add_trace(go.Scatter(
        x=[f'PC{i + 1}' for i in range(len(cum_var_ratio))],
        y=cum_var_ratio,
        mode='lines+markers',
        name='Cumulative Variance'
    ))
    cum_var_fig.update_layout(
        template='plotly_dark',
        title='Cumulative Explained Variance Ratio',
        xaxis_title='Number of Components',
        yaxis_title='Cumulative Explained Variance'
    )

    # Create PCA Plot
    pca_df = pd.DataFrame(
        data=pca_result[:, :2],
        columns=['PC1', 'PC2']
    )

    # Handle color feature
    if color_by == 'price_range':
        df['price_range'] = pd.qcut(
            df['price'].fillna(df['price'].median()),
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        color_column = 'price_range'
    elif color_by == 'size_category':
        df['size_category'] = pd.qcut(
            df['square_feet'].fillna(df['square_feet'].median()),
            q=3,
            labels=['Small', 'Medium', 'Large']
        )
        color_column = 'size_category'
    else:
        color_column = color_by

    # Add color column to PCA dataframe
    pca_df['color'] = df[color_column]

    # Create scatter plot with proper color handling
    pca_fig = go.Figure()

    # Add traces for each category
    for category in pca_df['color'].unique():
        mask = pca_df['color'] == category
        pca_fig.add_trace(go.Scatter(
            x=pca_df.loc[mask, 'PC1'],
            y=pca_df.loc[mask, 'PC2'],
            mode='markers',
            name=str(category),
            marker=dict(size=8),
            showlegend=True
        ))

    pca_fig.update_layout(
        template='plotly_dark',
        title='First Two Principal Components',
        height=600,
        xaxis_title=f'PC1 ({exp_var_ratio[0]:.2%} explained var.)',
        yaxis_title=f'PC2 ({exp_var_ratio[1]:.2%} explained var.)',
        showlegend=True,
        legend_title=color_by
    )

    # Create Loading Plot
    loading_matrix = pca.components_[:2, :]
    loading_fig = go.Figure()

    # Add arrows for loadings
    for i, feature in enumerate(features):
        loading_fig.add_trace(go.Scatter(
            x=[0, loading_matrix[0, i]],
            y=[0, loading_matrix[1, i]],
            mode='lines+text',
            name=feature,
            text=[None, feature],
            textposition='top center',
            line=dict(color='white')
        ))

    # Add unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    loading_fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='gray', dash='dot'),
        showlegend=False
    ))

    loading_fig.update_layout(
        template='plotly_dark',
        title='PCA Loading Plot',
        xaxis_title='PC1',
        yaxis_title='PC2',
        xaxis=dict(range=[-1.1, 1.1], showgrid=True),
        yaxis=dict(range=[-1.1, 1.1], showgrid=True),
        height=500,
        showlegend=True
    )

    return scree_fig, cum_var_fig, pca_fig, loading_fig

app.run_server(
    port = 8090,
    host = '0.0.0.0'
)