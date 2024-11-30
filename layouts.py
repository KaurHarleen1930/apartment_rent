from dash import html, dcc
import dash_bootstrap_components as dbc

####landing page
def landing_page_layout():
    return html.Div([
        html.H1("Information Visualization", className="header-title"),
        html.H2("Project: Apartment Rent Data", className="header-subtitle"),

        dbc.Row([
            # Dataset Overview Section
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Dataset Overview", className="content-title"),
                        html.P(
                            "This dataset comprises detailed information on apartment rentals, ideal for various "
                            "machine learning tasks including clustering, classification, and regression. It features "
                            "a comprehensive set of attributes that capture essential aspects of rental listings.",
                            className="content-description mb-4"
                        ),

                        # Key Features Section
                        html.Div([
                            html.H4("Key Features:", className="mb-3"),
                            html.Div([
                                html.H5("Identifiers & Location:", className="feature-title"),
                                html.P(
                                    "Includes unique identifiers (id), geographic details (address, cityname, state, latitude, longitude), and the source of the classified listing."),

                                html.H5("Property Details:", className="feature-title mt-3"),
                                html.P(
                                    "Provides information on the apartment's category, title, body, amenities, number of bathrooms, bedrooms, and square_feet (size of the apartment)."),

                                html.H5("Pricing Information:", className="feature-title mt-3"),
                                html.P(
                                    "Contains multiple features related to pricing, including price (rental price), price_display (displayed price), price_type (price in USD), and fee."),

                                html.H5("Additional Features:", className="feature-title mt-3"),
                                html.P(
                                    "Indicates whether the apartment has a photo (has_photo), whether pets are allowed (pets_allowed), and other relevant details such as currency and time of listing creation.")
                            ], className="feature-section")
                        ])
                    ])
                ], className="overview-card")
            ], width=12)
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
                    dbc.RadioItems(
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
                            {'label': "D'Agostino-Pearson Test", 'value': 'dagostino'}
                        ],
                        value=['shapiro', 'ks', 'dagostino'],
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
                            {"label": "KDE Plot", "value": "kde"}
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
                            )
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
                    dcc.Graph(id="city-tab-plot")
                ])
            ]),
            dbc.Tab(label="Correlation Analysis", tab_id="correlation-tab", children=[
                html.Div([
                    html.H2("Correlation Analysis", className="mt-4 mb-3"),
                    dcc.Graph(id="correlation-heatmap")
                ])
            ]),

        ], id="analysis-tabs", active_tab="state-tab")
    ])

