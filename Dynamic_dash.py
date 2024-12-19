import tempfile

import dash
import numpy as np
from dash import html, dcc, no_update
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
import plotly.io as pio
from scipy.stats import shapiro, kstest, normaltest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

###get data

url = 'https://github.com/KaurHarleen1930/apartment_rent-private-repo/raw/refs/heads/main/apartments_for_rent_classified_100K.csv'
#######please uncomment line 26 and comment line 24 if you want to read data locally
#url = 'apartments_for_rent_classified_100K.csv'
df = pd.read_csv(url, sep=";", encoding='cp1252',low_memory=False)
original_df = df.copy()

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    ]
)

app.title = "Data Visualization Apartment Rent Data"
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

from dash import html, dcc
import dash_bootstrap_components as dbc

####landing page
def landing_page_layout():
    return html.Div([
        # Row selection controls remain the same

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Let's Analyze apartment rent Data", className="mb-3"),
                        html.Header([
                            html.Div([
                                html.Figure([
                                    html.Img(src="/assets/img.png", alt="Apartment plot", className="mb-3",
                                             style={"width": "100%", "display": "inline-block"}),
                                    html.Figcaption("Data Visualization Apartment Data"),

                                ]
                                )
                            ])
                        ]),


                    ],
                    className="mb-3",)
                ], className="mb-4")
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Select the number of data observations", className="mb-3"),
                        dcc.RangeSlider(
                            id='row-range-slider',
                            min=0,
                            step=1000,
                            value=[0, len(df)],
                            marks={
                                0:'0',
                                5000: '5K',
                                10000: '10K',
                                25000: '25K',
                                50000: '50K',
                                75000: '75K',
                                90000: '90K'
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(id='final-row-count', className="mt-3")
                    ])
                ], className="mb-4")
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Set Text Style", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Slider(
                                    id='heading-size-slider',
                                    min=1,
                                    max=6,
                                    step=1,
                                    value=1,
                                    marks={i: f'H{i}' for i in range(1, 7)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=8),
                            dbc.Col([
                                dcc.Textarea(
                                    id='heading-size-input',
                                    placeholder='Enter H1-H6',
                                    value='H1',
                                    className="mb-2"
                                )
                            ], width=4)
                        ])
                    ])
                ], className="mb-4")
            ], width=12)
        ]),

        html.Div(id='content-container', children=[
            html.Div(id='dynamic-title', children=[
                html.H1("CS5764: Information Visualization Project", className="header-title")
            ]),
            html.H2("Project: Apartment Rent Data", className="header-subtitle"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Dataset Overview", className="content-title"),
                            html.P(
                                "This dataset comprises detailed information on apartment rentals, ideal for various "
                                "machine learning tasks including clustering, classification, and regression. It features "
                                "a comprehensive set of attributes that capture essential aspects of rental listings.",
                                className="content-description mb-4",
                                id="overview-text"
                            ),

                            html.Div([
                                html.H4("Key Features:", className="mb-3"),
                                html.Div([
                                    html.H5("Identifiers & Location:", className="feature-title"),
                                    html.P(
                                        "Includes unique identifiers (id), geographic details (address, cityname, state, latitude, longitude), and the source of the classified listing.",
                                        id="identifiers-text"
                                    ),

                                    html.H5("Property Details:", className="feature-title mt-3"),
                                    html.P(
                                        "Provides information on the apartment's category, title, body, amenities, number of bathrooms, bedrooms, and square_feet (size of the apartment).",
                                        id="property-text"
                                    ),

                                    html.H5("Pricing Information:", className="feature-title mt-3"),
                                    html.P(
                                        "Contains multiple features related to pricing, including price (rental price), price_display (displayed price), price_type (price in USD), and fee.",
                                        id="pricing-text"
                                    ),

                                    html.H5("Additional Features:", className="feature-title mt-3"),
                                    html.P(
                                        "Indicates whether the apartment has a photo (has_photo), whether pets are allowed (pets_allowed), and other relevant details such as currency and time of listing creation.",
                                        id="additional-text"
                                    )
                                ], className="feature-section")
                            ])
                        ])
                    ], className="overview-card")
                ], width=12)
            ])
        ])
    ], className="landing-content")


def data_cleaning_layout():
    return html.Div([
        html.H1("Data Cleaning"),
        html.Br(),
        html.H2("Select type of Feature", className="text-xl mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.Select(
                    options = [
                        {"label": "Categorical Features", "value": "categorical"},
                        {"label": "Numerical Features", "value": "numerical"},
                        ],
                        value='numerical',
                        id="feature-type",
                        className="mb-3",
                    ),
                    html.Div([
                        html.H3("Original Data Statistics", className="mb-3"),
                        html.Div(id="original-stats"),
                    ], className="mb-4"),

                    # Cleaned Data Display
                    html.Div([
                        html.H3("Cleaning Results", className="mb-3"),
                        html.Div(id="cleaning-results"),
                        html.Div(id="missing-values-table")
                    ], className="p-4 border rounded-lg"),

                    html.Div(id="method-warning", className="text-red-500 mb-3"),

                    # Store components
                    dcc.Store(id='cleaned-state'),
                    dcc.Store(id='original-data'),
                    dcc.Store(id='last-button-click')
                ], body=True)
            ], width=4),

            # Middle column - Methods selection
            dbc.Col([
                dbc.Card([
                    html.H3("Select Cleaning Method", className="mb-3"),
                    dcc.RadioItems(
                        options=[
                            {"label": "Display all null values", "value": "display"},
                            {"label": "Removal of missing row", "value": "remove"},
                            {"label": "Mean Method", "value": "mean"},
                            {"label": "Forward Fill", "value": "ffill"},
                            {"label": "Backward Fill", "value": "bfill"},
                            {"label": "Mode Method", "value": "mode"},
                            {"label": "Median Method", "value": "median"},
                            {"label":"Removal of Duplicates", "value": "duplicate"},
                        ],
                        value="display",
                        id="cleaning-method",
                        className="cleaning-methods",
                    ),
                    dbc.Button(
                        "Update Data",
                        id="apply-cleaning",
                        color="primary",
                        className="mt-3"
                    )
                ], body=True)
            ], width=4),

            # Right column - Results
            dbc.Col([
                dbc.Card([
                    html.Div(id="cleaning-results")
                ], body=True)
            ], width=4)
        ])
    ])


def outlier_detection_layout():
    return html.Div([
        html.H1("Outlier Detection"),
        dcc.Store(id='current-data-state'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H3("Select Features", className="mb-3"),
                    dcc.Dropdown(
                        id = "numerical-feature-dropdown",
                        options=[
                            {"label": "Bathrooms", "value": "bathrooms"},
                            {"label": "Bedrooms", "value": "bedrooms"},
                            {"label": "Price", "value": "price"},
                            {"label": "Square Feet", "value": "square_feet"},
                        ],
                        className="mb-4",
                        style={'color': 'black'},
                        placeholder="Select features...",
                    ),
                    html.H3("Select Plot Type", className="mb-3"),
                    dcc.Dropdown(
                        id = "select-plot-type",
                        options=[
                            {"label": "Box Plot", "value": "boxplot"},
                            {"label": "Violin Plot", "value": "violinplot"},
                            {"label": "Scatter Plot", "value": "scatterplot"},
                            {"label": "KDE Plot", "value": "kdeplot"},
                        ],
                        style={'color': 'black'},
                        multi = True,
                        placeholder="Select Plot Type..",
                        className="mb-4"
                    ),
                    dbc.Button(
                        "Show Plot",
                        id="show-outlier-plots",
                        color="primary",
                        className="w-100 mb-3"
                    ),
                    html.H3("Select Detection Method"),
                    dcc.Dropdown(
                        id="outlier-method",
                        options=[
                            {"label": "Z-Score", "value": "zscore"},
                            {"label": "IQR", "value": "iqr"},
                        ],
                        value="iqr",
                        style={'color': 'black'},
                        className="mb-4"
                    ),


                    dbc.Button(
                        "Detect Outliers",
                        id="detect-outliers",
                        color="primary",
                        className="w-100 mb-3"
                    ),
                    html.Div(id="outlier-summary", className="mt-4")
                ], body=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.H3("Original Data Distribution", className="mb-3"),
                    dbc.Spinner(html.Div(id="original-plots")),

                    # Outlier Detection Results
                    # html.H3("Outlier Detection Results", className="mb-3 mt-4"),
                    # dbc.Spinner(html.Div(id="outlier-plots"))
                ], body=True)
            ], width=8)
        ])
    ])
####data transformation
def data_transformation_layout():
    return html.Div([
        html.H1("Data Transformation Analysis"),

        dbc.Row([
            # Control Panel
            dbc.Col([
                dbc.Card([
                    html.H4("Select Feature", className="mb-3"),
                    dcc.Dropdown(
                        id='transform-feature-select',
                        options=[
                            {'label': 'Price', 'value': 'price'},
                            {'label': 'Square Feet', 'value': 'square_feet'},
                            {'label': 'Bedrooms', 'value': 'bedrooms'},
                            {'label': 'Bathrooms', 'value': 'bathrooms'}
                        ],
                        value='price',
                        style={'color': 'black'},
                        className="mb-3"
                    ),

                    html.H4("Select Transformations", className="mb-3"),
                    dcc.Checklist(
                        id='transform-methods',
                        options=[
                            {'label': 'Standardization (Z-Score)', 'value': 'standardization'},
                            {'label': 'Min-Max Normalization', 'value': 'normalization'},
                            {'label': 'Log Transformation', 'value': 'log'},
                        ],
                        value=['standardization', 'normalization', 'log'],
                        className="mb-3"
                    ),

                    dbc.Button(
                        "Apply Transformations",
                        id="apply-transforms-btn",
                        color="primary",
                        className="mt-3"
                    )
                ], body=True)
            ], width=3),

            # Results Panel
            dbc.Col([
                dbc.Card([
                    # Distribution Plots
                    html.H4("Distribution Comparison", className="mb-3"),
                    dcc.Graph(id="transform-distributions"),

                    # Statistics Table
                    html.Br(),
                    html.H4("Statistical Summary", className="mb-3"),
                    html.Div(id="transform-stats")
                ], body=True)
            ], width=9)
        ])
    ])


def normality_test_layout():
    return html.Div([
        html.H1("Normality Test Analysis"),

        dbc.Row([
            # Control Panel
            dbc.Col([
                dbc.Card([
                    html.H4("Select Feature", className="mb-3"),
                    dcc.Dropdown(
                        id='normality-feature-select',
                        options=[
                            {'label': 'Price', 'value': 'price'},
                            {'label': 'Square Feet', 'value': 'square_feet'},
                            {'label': 'Bedrooms', 'value': 'bedrooms'},
                            {'label': 'Bathrooms', 'value': 'bathrooms'}
                        ],
                        value='price',
                        style={'color': 'black'},
                        className="mb-3"
                    ),

                    html.H4("Select Tests", className="mb-3"),
                    dcc.Checklist(
                        id='normality-tests',
                        options=[
                            {'label': 'Shapiro-Wilk Test', 'value': 'shapiro'},
                            {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'},
                            #{'label': "D'Agostino-Pearson Test", 'value': 'dagostino'}
                        ],
                        value=['shapiro', 'ks'],
                        className="mb-3"
                    ),

                    dbc.Button(
                        "Run Tests",
                        id="run-normality-tests-btn",
                        color="primary",
                        className="mt-3"
                    )
                ], body=True)
            ], width=3),

            # Results Panel
            # dbc.Col([
            #     dbc.Card([
            #         # QQ Plot and Histogram
            #         html.H4("Distribution Analysis", className="mb-3"),
            #         dcc.Graph(id="normality-plots"),
            #
            #         # Test Results
            #         html.H4("Test Results", className="mb-3"),
            #         html.Div(id="normality-results", className="test-results")
            #     ], body=True)
            # ], width=9)
            dbc.Col([
                # Distribution Analysis Card
                dbc.Card([
                    dbc.CardHeader(
                        html.H3("Distribution Analysis", className="mb-0")
                    ),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id="normality-plots",
                                config={
                                    'displayModeBar': True,
                                    'responsive': True
                                },
                                style={
                                    'height': '600px',
                                    'width': '100%'
                                }
                            )
                        )
                    ], className="p-0")
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardHeader(
                        html.H3("Test Results", className="mb-0")
                    ),
                    dbc.CardBody(
                        id="normality-results"
                    )
                ])
            ], width=9)
        ])
    ])


def dimensionality_reduction_layout():
    return html.Div([
        html.H1("PCA Dimensionality Reduction"),

        dbc.Row([
            # Control Panel
            dbc.Col([
                dbc.Card([
                    html.H4("Select Features", className="mb-3"),
                    dcc.Checklist(
                        id='feature-selector',
                        options=[
                            {'label': ' Price', 'value': 'price'},
                            {'label': ' Square Feet', 'value': 'square_feet'},
                            {'label': ' Bedrooms', 'value': 'bedrooms'},
                            {'label': ' Bathrooms', 'value': 'bathrooms'},
                            {'label': ' Latitude', 'value': 'latitude'},
                            {'label': ' Longitude', 'value': 'longitude'}
                        ],
                        value=['price', 'square_feet', 'bedrooms', 'bathrooms'],
                        className="mb-4"
                    ),

                    html.H4("Color By", className="mb-3"),
                    dcc.Dropdown(
                        id='color-feature',
                        options=[
                            {'label': 'State', 'value': 'state'},
                            {'label': 'Price Range', 'value': 'price_range'},
                            {'label': 'Size Category', 'value': 'size_category'}
                        ],
                        value='state',
                        style={'color': 'black'},
                        className="mb-4"
                    ),

                    dbc.Button(
                        "Apply PCA",
                        id="apply-pca-btn",
                        color="primary",
                        className="w-100 mb-3"
                    )
                ], body=True)
            ], width=3),

            # Results Panel
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        # Scree Plot
                        dbc.Col([
                            html.H4("Explained Variance Ratio", className="mb-3"),
                            dcc.Graph(id="scree-plot")
                        ], width=6),

                        # Cumulative Variance
                        dbc.Col([
                            html.H4("Cumulative Explained Variance", className="mb-3"),
                            dcc.Graph(id="cumulative-variance-plot")
                        ], width=6)
                    ]),

                    # PCA Plot
                    html.H4("PCA Components Visualization", className="mb-3 mt-4"),
                    dcc.Graph(id="pca-plot"),

                    # Loading Plot
                    html.H4("Feature Loadings", className="mb-3"),
                    dcc.Graph(id="loading-plot")
                ], body=True)
            ], width=9)
        ])
    ])


def numerical_viz_layout():
    return html.Div([
        html.H1("Numerical Features Visualization"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H3("Univariate Analysis", className="mb-3"),
                    dcc.Dropdown(
                        id="uni-feature-selector",
                        options=[
                            {"label": "Bathrooms", "value": "bathrooms"},
                            {"label": "Bedrooms", "value": "bedrooms"},
                            {"label": "Price", "value": "price"},
                            {"label": "Square Feet", "value": "square_feet"},
                            {"label": "Latitude", "value": "latitude"},
                            {"label": "Longitude", "value": "longitude"}
                        ],
                        placeholder="Select a feature...",
                        className="mb-3",
                        style={'color': 'black'}
                    ),

                    dcc.Dropdown(
                        id="uni-plot-selector",
                        options=[
                            {"label": "Histogram", "value": "histogram"},
                            {"label": "Box Plot", "value": "box"},
                            {"label": "Violin Plot", "value": "violin"},
                            {"label": "KDE Plot", "value": "kde"},
                            # {"label": "KDE Density Plot", "value": "density"},
                            {"label": "Regression Plot", "value": "reg"},
                            # {"label": "Distribution Plot with Rug", "value": "dist_rug"},
                            {"label":"Area Plot", "value": "area"},
                        ],
                        multi=True,
                        placeholder="Select plot types...",
                        className="mb-3",
                        style={'color': 'black'}
                    ),

                    dbc.Button(
                        "Generate Univariate Plots",
                        id="uni-plot-button",
                        color="primary",
                        className="w-100 mb-4"
                    ),
                    html.H3("Bivariate Analysis", className="mb-3"),
                    dcc.Dropdown(
                        id="bi-feature-selector-1",
                        options=[
                            {"label": "Bathrooms", "value": "bathrooms"},
                            {"label": "Bedrooms", "value": "bedrooms"},
                            {"label": "Price", "value": "price"},
                            {"label": "Square Feet", "value": "square_feet"},
                            {"label": "Latitude", "value": "latitude"},
                            {"label": "Longitude", "value": "longitude"}
                        ],
                        placeholder="Select first feature...",
                        className="mb-3",
                        style={'color': 'black'}
                    ),

                    dcc.Dropdown(
                        id="bi-feature-selector-2",
                        options=[
                            {"label": "Bathrooms", "value": "bathrooms"},
                            {"label": "Bedrooms", "value": "bedrooms"},
                            {"label": "Price", "value": "price"},
                            {"label": "Square Feet", "value": "square_feet"},
                            {"label": "Latitude", "value": "latitude"},
                            {"label": "Longitude", "value": "longitude"}
                        ],
                        placeholder="Select second feature...",
                        className="mb-3",
                        style={'color': 'black'}
                    ),

                    dcc.Dropdown(
                        id="bi-plot-selector",
                        options=[
                            {"label": "Scatter Plot", "value": "scatter"},
                            {"label": "Hexbin Plot", "value": "hexbin"},
                            {"label": "Density Contour", "value": "density"},
                            {"label": "2D Histogram", "value": "histogram2d"},
                            {"label": "Clustering", "value": "cluster"},
                            {"label": "Joint Plot", "value": "joint"},
                        ],
                        multi=True,
                        placeholder="Select plot types...",
                        className="mb-3",
                        style={'color': 'black'}
                    ),

                    dbc.Button(
                        "Generate Bivariate Plots",
                        id="bi-plot-button",
                        color="primary",
                        className="w-100 mb-3"
                    ),
                    html.Div(id="correlation-stats-uni", className="mt-4"),
                    html.Div(id="correlation-stats-bi", className="mt-4")
                ], body=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.H3("Univariate Analysis Results", className="mb-3"),
                    dbc.Spinner(html.Div(id="univariate-plots")),

                    # Bivariate Plots Section
                    html.H3("Bivariate Analysis Results", className="mb-3 mt-4"),
                    dbc.Spinner(html.Div(id="bivariate-plots"))

                ], body=True)
            ], width=8)
        ])
    ])


def categorical_viz_layout():
    return html.Div([
        html.H1("Categorical Features Analysis"),

        dbc.Row([
            # Left Column - Controls
            dbc.Col([
                dbc.Card([
                    # Univariate Analysis Section
                    html.H3("Categorical Features Analysis", className="mb-3"),
                    dcc.Dropdown(
                        id="cat-uni-feature",
                        options=[
                            {"label": "State", "value": "state"},
                            {"label": "City", "value": "cityname"},
                            {"label": "Has Photo", "value": "has_photo"},
                            {"label": "Pets Allowed", "value": "pets_allowed"},
                            {"label": "Source", "value": "source"},
                            {"label":"Amenities", "value": "amenities"},
                        ],
                        placeholder="Select a categorical feature...",
                        className="mb-3",
                        style={'color': 'black'}
                    ),

                    dcc.Dropdown(
                        id="cat-uni-plot-type",
                        options=[
                            {"label": "Bar Plot", "value": "bar"},
                            {"label": "Pie Chart", "value": "pie"},
                            {"label": "Treemap", "value": "treemap"},
                            {"label": "Frequency Table", "value": "table"}
                        ],
                        multi=True,
                        placeholder="Select plot types...",
                        className="mb-3",
                        style={'color': 'black'}
                    ),

                    dbc.Button(
                        "Generate Categorical Feature Analysis",
                        id="cat-uni-button",
                        color="primary",
                        className="w-100 mb-4"
                    ),

                    # Bivariate Analysis Section
                    # html.H3("Bivariate Analysis", className="mb-3"),
                    # dcc.Dropdown(
                    #     id="cat-bi-feature1",
                    #     options=[
                    #         {"label": "State", "value": "state"},
                    #         {"label": "City", "value": "cityname"},
                    #         {"label": "Has Photo", "value": "has_photo"},
                    #         {"label": "Pets Allowed", "value": "pets_allowed"},
                    #         {"label": "Source", "value": "source"},
                    #         {"label":"Amenities", "value": "amenities"},
                    #     ],
                    #     placeholder="Select first categorical feature...",
                    #     className="mb-3",
                    #     style={'color': 'black'}
                    # ),
                    #
                    # dcc.Dropdown(
                    #     id="cat-bi-feature2",
                    #     options=[
                    #         {"label": "State", "value": "state"},
                    #         {"label": "City", "value": "cityname"},
                    #         {"label": "Has Photo", "value": "has_photo"},
                    #         {"label": "Pets Allowed", "value": "pets_allowed"},
                    #         {"label": "Source", "value": "source"},
                    #         {"label":"Amenities", "value": "amenities"},
                    #     ],
                    #     placeholder="Select second categorical feature...",
                    #     className="mb-3",
                    #     style={'color': 'black'}
                    # ),
                    #
                    # dcc.Dropdown(
                    #     id="cat-bi-plot-type",
                    #     options=[
                    #         {"label": "Stacked Bar", "value": "stacked_bar"},
                    #         {"label": "Grouped Bar", "value": "grouped_bar"},
                    #         {"label": "Heatmap", "value": "heatmap"},
                    #         {"label": "Sunburst", "value": "sunburst"},
                    #         {"label": "Contingency Table", "value": "table"}
                    #     ],
                    #     multi=True,
                    #     placeholder="Select plot types...",
                    #     className="mb-3",
                    #     style={'color': 'black'}
                    # ),
                    #
                    # dbc.Button(
                    #     "Generate Bivariate Analysis",
                    #     id="cat-bi-button",
                    #     color="primary",
                    #     className="w-100 mb-3"
                    # ),
                ], body=True)
            ], width=4),

            # Right Column - Plots
            dbc.Col([
                dbc.Card([
                    # Univariate Results
                    html.H3("Analysis Results", className="mb-3"),
                    dbc.Spinner(html.Div(id="cat-uni-plots")),

                    # Bivariate Results
                    # html.H3("Bivariate Analysis Results", className="mb-3 mt-4"),
                    # dbc.Spinner(html.Div(id="cat-bi-plots"))
                ], body=True)
            ], width=8)
        ])
    ])


def statistical_analysis_layout():
    return html.Div([
        html.H1("Statistical Analysis Dashboard", className="mb-4"),

        dbc.Tabs([
            dbc.Tab(label="State Analysis", tab_id="state-tab", children=[
                html.Div([
                    html.H2("State-wise Price Analysis with Apartment Count", className="mt-4 mb-3"),
                    dcc.Graph(id="state-price-count-plot")
                ])
            ]),

            dbc.Tab(label="City-State Analysis", tab_id="city-state-tab", children=[

                dbc.Card([
                    dbc.Row([

                        dbc.Col([
                            html.H4("Select State"),
                            dcc.Dropdown(
                                id='state-selector',
                                options=[],
                                placeholder="Select a state...",
                                style={'color': 'black'},
                                className="mb-3"
                            )
                        ], width=4),

                        # Plot Type Selection
                        dbc.Col([
                            html.H4("Select Plot Type"),
                            dcc.Dropdown(
                                id='plot-type-selector',
                                options=[
                                    {'label': 'Price Distribution', 'value': 'price_dist'},
                                    {'label': 'City Rankings', 'value': 'rankings'},
                                    {'label': 'Listings Count', 'value': 'listings'}
                                ],
                                multi=True,
                                value=['price_dist'],
                                style={'color': 'black'},
                                className="mb-3"
                            ),

                        ], width=4)
                    ])
                ], body=True, className="mb-4"),

                # Results Section
                dbc.Row([
                    # Visualizations
                    dbc.Col([
                        html.Div(id="city-plots")
                    ], width=8),

                    # Statistics Panel
                    dbc.Col([
                        dbc.Card([
                            html.H4("City Statistics", className="mb-3"),
                            html.Div(id="city-stats")
                        ], body=True)
                    ], width=4)
                ])
            ]),

            dbc.Tab(label="Bedroom-Bathroom Analysis", tab_id="bedroom-tab", children=[
                html.Div([
                    html.H2("State-wise Bedroom, Bathroom Analysis", className="mt-4 mb-3"),
                    dcc.Graph(id="bedroom-bathroom-plot"),
                    html.H2("Bedroom-Bathroom Ratio Analysis", className="mt-4 mb-3"),
                    dcc.Graph(id="bedbath-ratio-plot")
                ])
            ]),

            dbc.Tab(label="Square Feet Analysis", tab_id="sqft-tab", children=[
                html.Div([
                    html.H2("Price per Square Foot Analysis", className="mt-4 mb-3"),
                    dcc.Graph(id="sqft-price-plot")
                ])
            ]),
            dbc.Tab(label="City vs Avg Price Analysis", tab_id="city-tab", children=[
                html.Div([
                    html.H2("Most Expensive and cheapest cities Analysis", className="mt-4 mb-3"),

                    dcc.Download(id="download-graph"
                                 ),
                    html.Div([
                    html.I(
                        className="fas fa-download",

                    ),
                    html.Span("Download Graph", className="download-text", style={'margin-left':'15px'})
                    ],id="download-button-graph",
                        style={
                            'cursor': 'pointer',
                            'color': '#94a3b8',
                            'fontSize': '1.2rem',
                            'marginBottom': '10px',
                            'zIndex':9999
                        }
                    ),
                    dcc.Graph(id="city-tab-plot")
                ])
            ]),
            dbc.Tab(label="Correlation Analysis", tab_id="correlation-tab", children=[
                html.Div([
                    html.H2("Correlation Analysis", className="mt-4 mb-3"),
                    dcc.Graph(id="correlation-heatmap")
                ])
            ]),
            # dbc.Tab(label="Pair Plot Analysis", tab_id="pair-plot-tab", children=[
            #     html.Div([
            #         html.H2("Pair Grid with Histogram, Scatter, and KDE Plots for Numerical Columns", className="mt-4 mb-3"),
            #         dcc.Graph(id="pair-plot")
            #     ])
            # ]),

        ], id="analysis-tabs", active_tab="state-tab")
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


#############Landing page callback
@app.callback(
    [Output('content-container', 'style'),
     Output('dynamic-title', 'children'),
     Output('overview-text', 'style'),
     Output('identifiers-text', 'style'),
     Output('property-text', 'style'),
     Output('pricing-text', 'style'),
     Output('additional-text', 'style')],
    [Input('heading-size-slider', 'value'),
     Input('heading-size-input', 'value')]
)
def update_text_sizes(slider_val, input):

    try:
        if input.startswith('H'):
            size = int(input[1])
            if 1 <= size <= 6:
                heading_size = size
            else:
                heading_size = slider_val
        else:
            heading_size = slider_val
    except:
        heading_size = slider_val

    font_size_map = {
        1: {'heading': '2.5rem', 'body': '1.2rem'},
        2: {'heading': '2rem', 'body': '1.1rem'},
        3: {'heading': '1.75rem', 'body': '1rem'},
        4: {'heading': '1.5rem', 'body': '0.95rem'},
        5: {'heading': '1.25rem', 'body': '0.9rem'},
        6: {'heading': '1rem', 'body': '0.85rem'}
    }

    font_sizes = font_size_map[heading_size]
    heading_component = getattr(html, f'H{heading_size}')

    container_style = {'fontSize': font_sizes['body']}
    content_style = {'fontSize': font_sizes['body']}

    return (
        container_style,
        [heading_component("Information Visualization", className="header-title")],
        content_style,
        content_style,
        content_style,
        content_style,
        content_style
    )



@app.callback(
    Output('heading-size-input', 'value'),
    [Input('heading-size-slider', 'value')]
)
def update_heading_input(slider):
    return f'H{slider}'


@app.callback(
    Output('heading-size-slider', 'value'),
    [Input('heading-size-input', 'value')]
)
def update_heading_slider(input):
    try:
        if input.startswith('H'):
            size = int(input[1])
            if 1 <= size <= 6:
                return size
    except:
        pass
    return dash.no_update

@app.callback(
    Output('final-row-count', 'children'),
    Input('row-range-slider', 'value'),
)
def update_apts_data_loaded(slider_value):
    global df
    if len(original_df)!=len(df):
        df = original_df

    min_rows, max_rows = slider_value
    df = df.iloc[min_rows:max_rows].copy()

    return html.Label(f"Dataset has been reduced to {len(df):,} rows", style={'color': '#94a3b8'})


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

    if cleaned_state is None:
        cleaned_state = {
            'categorical': original_data['categorical'],
            'numerical': original_data['numerical']
        }

    if feature_type == "categorical":
        working_df = pd.DataFrame(cleaned_state.get('categorical', original_data['categorical']))
    else:
        working_df = pd.DataFrame(cleaned_state.get('numerical', original_data['numerical']))

    try:
        if cleaning_method == "display":
            pass
        elif cleaning_method == "remove":
            working_df = working_df.dropna()
        elif cleaning_method == "duplicate":
            working_df = working_df.drop_duplicates()
        elif cleaning_method == "mode":
            working_df = working_df.fillna(working_df.mode().iloc[0])
        elif cleaning_method == "ffill":
            working_df = working_df.fillna(method='ffill')
        elif cleaning_method == "bfill":
            working_df = working_df.fillna(method='bfill')
        elif feature_type == "numerical":
            if cleaning_method == "mean":
                working_df = working_df.fillna(working_df.mean())
            elif cleaning_method == "median":
                working_df = working_df.fillna(working_df.median())

    except Exception as e:
        print(f"Error in cleaning: {str(e)}")
        return dash.no_update, dash.no_update, f"Error applying cleaning method: {str(e)}", None

    missing_vals = working_df.isnull().sum()

    results_table = html.Table(
        [html.Tr([html.Th("Column"), html.Th("Missing Values")])] +
        [html.Tr([html.Td(col), html.Td(str(missing_vals[col]))])
         for col in working_df.columns],
        className="table-auto w-full"
    )

    new_cleaned_state = cleaned_state.copy()
    new_cleaned_state[feature_type] = working_df.to_dict('records')

    summary = html.Div([
        html.P(f"Selected Feature Type: {feature_type}"),
        html.P(f"Applied Cleaning Method: {cleaning_method}"),
        html.P(f"Total rows: {len(working_df)}"),
        html.P(f"Total missing values: {missing_vals.sum()}")
    ])

    return results_table, summary, "", new_cleaned_state

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
    elif method == "zscore":
        numeric_data = pd.to_numeric(apts[feature], errors='coerce')
        valid_data = numeric_data.dropna()
        zscore_stat = (valid_data - valid_data.mean()) / valid_data.std()
        z_score = np.abs(zscore_stat)
        outlier_mask = z_score < 3
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
            try:
                clean_data = np.array(data.dropna())
                k2, p_val = stats.normaltest(clean_data, nan_policy='omit')
                result = {
                    'test': "D'Agostino-Pearson Test",
                    'statistic': float(k2),
                    'p_value': float(p_val),
                    'is_normal': bool(p_val > alpha)

                }

            except Exception as e:

                result = {
                    'test': "D'Agostino-Pearson Test",
                    'statistic': None,
                    'p_value': None,
                    'is_normal': None,
                    'error': f"Could not compute test: {str(e)}"
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

def create_kde_density_plot(df, feature):
    kde = stats.gaussian_kde(df[feature])
    x_range = np.linspace(df[feature].min(), df[feature].max(), 200)
    kde_values = kde(x_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values,
        fill='tozeroy',
        line=dict(
            width=2.5
        ),
        opacity=0.6,
        name=feature
    ))

    fig.update_layout(
        title=f'KDE Plot of {feature}',
        template='plotly_dark',
        showlegend=False,
        xaxis_title=feature,
        yaxis_title='Density'
    )
    return fig

#reg plot
def create_reg_plot(df, feature):
    fig = px.scatter(
        df,
        x=df[feature].rank().to_numpy(),
        y=df[feature].to_numpy(),
        trendline="ols",
        title=f'Regression Plot of {feature}'
    )
    fig.update_traces(
        marker=dict(
            opacity=0.5,
        )
    )
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.line.color = 'purple'
            trace.line.width = 1.5
    return fig


def create_dist_rug_plot(df, feature):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df[feature],
        histnorm='probability density',
        name='Distribution',
        opacity=0.3,
    ))

    kde_x = np.linspace(df[feature].min(), df[feature].max(), 100)
    kde = stats.gaussian_kde(df[feature])
    fig.add_trace(go.Scatter(
        x=kde_x,
        y=kde(kde_x),
        mode='lines',
        name='KDE',
        line=dict(width=2)
    ))

    # Add rug plot
    fig.add_trace(go.Scatter(
        x=df[feature],
        y=[0] * len(df),
        mode='markers',
        marker=dict(
            symbol='line-ns-open',
            size=8,
            opacity=0.6,
        ),
        name='Rug'
    ))

    fig.update_layout(
        title=f'Distribution Plot of {feature} with Rug',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    return fig


def create_area_plot(df, feature):
    sorted_data = df[feature].sort_values()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(sorted_data)),
        y=sorted_data,
        fill='tozeroy',
        name=feature,
        opacity=0.6,
        line=dict(width=2),
    ))
    fig.update_layout(
        title=f'Area Plot of {feature}',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    return fig

######joint plot and pairgrid
def create_joint_plot(df, x_col, y_col):
    # Sample data if too large
    if len(df) > 5000:
        plot_df = df.sample(n=5000, random_state=42)
    else:
        plot_df = df.copy()

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(
                opacity=0.7,
                size=8
            ),
            name='Scatter'
        ),
        row=2, col=1
    )

    try:
        x = plot_df[x_col]
        y = plot_df[y_col]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        # Reduce grid size for better performance
        xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        # Add bandwidth adjustment for better KDE estimation
        kernel = stats.gaussian_kde(values, bw_method='scott')
        z = np.reshape(kernel(positions).T, xx.shape)

        fig.add_trace(
            go.Contour(
                x=np.linspace(xmin, xmax, 50),
                y=np.linspace(ymin, ymax, 50),
                z=z,
                colorscale='Viridis',
                opacity=0.7,
                showscale=False,
                name='KDE'
            ),
            row=2, col=1
        )
    except Exception as e:
        print(f"Error in KDE calculation: {str(e)}")

    fig.add_trace(
        go.Histogram(
            x=plot_df[x_col],
            nbinsx=30,
            opacity=0.7,
            name=f'{x_col} dist'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            y=plot_df[y_col],
            nbinsy=30,
            opacity=0.7,
            name=f'{y_col} dist'
        ),
        row=2, col=2
    )

    fig.update_layout(
        template='plotly_dark',
        height=800,
        title=f'Joint Plot: {x_col} vs {y_col}',
        showlegend=False
    )

    fig.update_xaxes(title_text=x_col, row=2, col=1)
    fig.update_yaxes(title_text=y_col, row=2, col=1)

    return fig




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
        elif plot_type == 'density':
            fig = create_kde_density_plot(apts, features)
        elif plot_type == 'reg':
            fig = create_reg_plot(apts, features)
        elif plot_type == 'dist_rug':
            fig = create_dist_rug_plot(apts, features)
        elif plot_type == 'area':
            fig = create_area_plot(apts, features)

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
        elif plot_type == 'joint':
            fig = create_joint_plot(apts, feature1, feature2)

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
                ),
                hoverinfo='none'
            ))

            fig.update_layout(
                title=f"Price vs Number of Listings in {state}",
                template="plotly_dark",
                height=600,
                xaxis_title="Number of Listings",
                yaxis_title="Average Price ($)"
            )
            plots.append(html.Div([
            dcc.Graph(
                id='listings-plot',
                figure=fig,
                config={'displayModeBar': True}
            ),
                dcc.Tooltip(
                    id='listings-tooltip',
                    direction='bottom',
                    style={
                        'backgroundColor': '#1e293b',
                        'color': 'white',
                        'border': '1px solid #475569',
                        'borderRadius': '6px',
                        'padding': '10px',
                        'zIndex': 9999
                    }
                )
        ], style={'position': 'relative'})
            )

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

#tooltip addition
@app.callback(
    [Output('listings-tooltip', 'show'),
    Output('listings-tooltip', 'bbox'),
    Output('listings-tooltip', 'children')],
    Input('listings-plot', 'hoverData'),
    State('state-selector', 'value')
)
def update_listings_tooltip(hover_data, state):
    if hover_data is None:
        return False, no_update, no_update
    hover_pt = hover_data['points'][0]
    bbox = hover_pt['bbox']
    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    city_stats = apts[apts['state'] == state].groupby('cityname').agg({
        'price': ['mean', 'count'],
        'square_feet': 'mean',
        'bedrooms': 'mean',
        'bathrooms': 'mean'
    }).round(2)
    city_stats.columns = ['avg_price', 'listings', 'avg_sqft', 'avg_beds', 'avg_baths']
    city_data = city_stats.iloc[hover_pt['pointIndex']]
    children = [
        html.Div([
            html.H4(hover_pt['text'], style={'margin': '0', 'color': '#e2e8f0'}),
            html.Div([
                html.P([
                    "Average Price of City: ",
                    html.Span(f"${city_data['avg_price']:,.2f}", style={'color': '#a5b4fc'})
                ], style={'margin': '5px 0'}),
                html.P([
                    "Number of Listings in City: ",
                    html.Span(f"{city_data['listings']}", style={'color': '#a5b4fc'})
                ], style={'margin': '5px 0'}),
                html.P([
                    "Avg Square Feet: ",
                    html.Span(f"{city_data['avg_sqft']:,.0f}", style={'color': '#a5b4fc'})
                ], style={'margin': '5px 0'}),
                html.P([
                    "Avg Beds/Baths for City: ",
                    html.Span(
                        f"{city_data['avg_beds']:.1f}/{city_data['avg_baths']:.1f}",
                        style={'color': '#a5b4fc'}
                    )
                ], style={'margin': '5px 0'})
            ])
        ])
    ]
    print(bbox)
    return True, bbox, children




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

#pair plot
def create_pair_grid(df, numerical_cols):
    fig = make_subplots(
        rows=len(numerical_cols),
        cols=len(numerical_cols),
        subplot_titles=[f"{col1} vs {col2}" for col1 in numerical_cols for col2 in numerical_cols],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    for i, col1 in enumerate(numerical_cols, 1):
        for j, col2 in enumerate(numerical_cols, 1):
            if i == j:
                fig.add_trace(
                    go.Histogram(
                        x=df[col1],
                        opacity=0.7
                    ),
                    row=i, col=j
                )
            elif i < j:
                fig.add_trace(
                    go.Scatter(
                        x=df[col1],
                        y=df[col2],
                        mode='markers',
                        marker=dict(
                            opacity=0.5,
                            size=6
                        )
                    ),
                    row=i, col=j
                )
            else:
                x = df[col1]
                y = df[col2]
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y.min(), y.max()

                xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])
                kernel = stats.gaussian_kde(values)
                z = np.reshape(kernel(positions).T, xx.shape)

                fig.add_trace(
                    go.Contour(
                        x=np.linspace(xmin, xmax, 50),
                        y=np.linspace(ymin, ymax, 50),
                        z=z,
                        opacity=0.7,
                        showscale=False,
                    ),
                    row=i, col=j
                )

    fig.update_layout(
        height=1000,
        title='Pair Grid with Histogram, Scatter, and KDE Plots',
        showlegend=False,
    )


    return fig



@app.callback(
    Output("pair-plot", "figure"),
    Input("analysis-tabs", "active_tab")
)
def update_pair_plot(active_tab):
    if active_tab != "pair-plot-tab":
        raise PreventUpdate

    apts = get_apartment_rent_data(df)
    apts = perform_data_cleaning(apts)

    numerical_cols = ['bathrooms', 'bedrooms', 'price', 'square_feet']
    fig = create_pair_grid(apts, numerical_cols)

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

####download

@app.callback(
    Output("download-graph", "data"),
    Input("download-button-graph", "n_clicks"),
    State("city-tab-plot", "figure"),
    prevent_initial_call=True
)
def download_figure(n_clicks, figure):
    if not n_clicks:
        raise PreventUpdate

    img_bytes = pio.to_image(figure, format="png")
    return dcc.send_bytes(img_bytes, "listings_plot.png")
# def download_figure(n_clicks, figure):
#     if not n_clicks:
#         raise PreventUpdate
#
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#         temp_path = tmp_file.name
#         pio.write_image(figure, temp_path, format="png")
#
#     return dcc.send_file(temp_path)

app.run_server(
    port = 8090,
    host = '0.0.0.0'
)