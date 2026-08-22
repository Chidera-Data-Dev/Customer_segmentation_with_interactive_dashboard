#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[2]:


# Load the data
df = load_data()


# In[3]:


# Calculate summary statistics
total_customers = len(df)
avg_spending = df['Monetary (BRL)'].mean()
avg_score = df['review_score'].mean()


# In[4]:


# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# In[5]:


# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Olist Customer Segmentation Dashboard"), width=12, className="text-center my-4")
    ], style={
        "backgroundColor": "#EAF2FF",
        "font-style": "Arial",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }),
    # Summary Statistics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("👥 Total Customers", className="card-title"),
                    html.H6(total_customers, className="card-title")
                ])
            ])
        ], style={
        "backgroundColor": "#F5F8FC",
        "borderRadius": "5px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }, width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("💲Average Spending", className="card-title"),
                    html.H6(f"BRL ${avg_spending:,.2f}", className="card-title")
                ])
            ])
        ], style={
        "backgroundColor": "#F5F8FC",
        "borderRadius": "5px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }, width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("⭐Average Review Score", className="card-title"),
                    html.H6(f"{avg_score:,.2f}", className="card-title")
                ])
            ])
        ], style={
        "backgroundColor": "#F5F8FC",
        "borderRadius": "5px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }, width=4),
    ]),
    dbc.Row([
        dbc.Col(html.Div(), width=2),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Customer City", className="card-title"),
                    dcc.Dropdown(
                        id="city-filter",
                        options=[{"label": city, "value": city} for city in df['customer_city'].unique()],
                        value=None,
                        placeholder="Select City"
                    ),
                    html.H2("K-means Clustering", className="card-title"),
                    html.H5("Number of Clusters (k)"),
                    dcc.Slider(min=2, max=12, step=1, value=2, id="k-slider")
                ])
            ])
        ], style={
        "borderRadius": "10px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }, width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div(), width=2),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Cluster Behaviour Plot", className="card-title"),
                    dcc.Graph(id="group-plot")
                ])
            ])
        ], style={
        "borderRadius": "10px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }, width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div(), width=2),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("PCA Plot", className="card-title"),
                    dcc.Graph(id="pca-scatter")
                ])
            ])
        ], style={
        "borderRadius": "10px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }, width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div(), width=2),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Cluster Distribution", className="card-title"),
                    dcc.Graph(id="bar-plot")
                ])
            ])
        ], style={
        "borderRadius": "10px",
        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
    }, width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div(), width=2),
    ], className='mb-4'),
    dbc.Row([
        dbc.Col(html.H6("ℹ️K-means clustering was applied to customer data using RFM (Recency, Frequency, Monetary) and other variables to identify distinct customer segments."), 
                width=12)
    ])

], fluid=True, style={
        "backgroundColor": "#F5F8FC",
        "minHeight": "100vh",
        "padding": "20px"
    })


# # KMeans Modeling Function & Callback

# In[6]:


def k_means_model(k=2, selected_city=None):  
    """Build ``KMeans`` model based on ``df``.

    Parameters
    ----------
    selected_city : string
        selects a city if needed for specialization

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.

    """
    # get features
    filt_df = df[df['customer_city'] == selected_city] if selected_city else df
    X=filt_df.select_dtypes(include=['int64', 'float64'])
    # Build model
    model=make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42, n_init=10))
    model.fit(X)
    
    return model


# # Cluster Grouping Function & Callback

# In[7]:


def get_cluster_group(k=2, selected_city=None):  
    """Getting Cluster labels
    ``KMeans`` labels.

    Parameters
    ----------
    selected_city : string
        selects a city if needed for specialization
        
    k : int, default=2
        Number of clusters.
    """
    # get features
    filt_df = df[df['customer_city'] == selected_city] if selected_city else df
    X=filt_df.select_dtypes(include=['int64', 'float64'])

    # add cluster labels
    model = k_means_model(k=k, selected_city=selected_city)
    X["clusters"] = model.named_steps["kmeans"].labels_
    inspection = X.groupby('clusters').mean()
    return inspection



@app.callback(
    Output("group-plot", "figure"), Input("k-slider", "value"), Input('city-filter', 'value')
)
def serve_group_plot(k=2, selected_city=None):
    """Plot a Group bar plot of ``Inspection`` to view customer behaviour.

    Parameters
    ----------
    selected_city : string
        selects a city if needed for specialization
        
    k : int, default=2
        Number of clusters.
    """
    # Build Bar Chart
    fig = px.bar( 
        data_frame=get_cluster_group(k=k, selected_city=selected_city),
        barmode="group",
        title="Mean customer behvaiour by Cluster"
    )
    fig.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",

    font=dict(
        family="Arial",
        color="#17365D"
    ),

    title_font=dict(
        size=20,
        color="#0B3B82"
    ),

    margin=dict(
        l=50,
        r=30,
        t=50,
        b=50
    ),

    legend=dict(
        bgcolor="rgba(255,255,255,0)"
    ), xaxis_title="Clusters", yaxis_title="Mean Value")
    return fig


# # PCA Plot Function & Callback

# In[8]:


def get_pca_labels(k=2, selected_city=None):  
    """
    ``KMeans`` labels.

    Parameters
    ----------
    selected_city : string
        selects a city if needed for specialization
        
    k : int, default=2
        Number of clusters.
    """
    # get features
    filt_df = df[df['customer_city'] == selected_city] if selected_city else df
    X=filt_df.select_dtypes(include=['int64', 'float64'])
    
    # Build transformer
    transformer = PCA(n_components=2, random_state=42)
    
    # transform data
    X_t = transformer.fit_transform(X)
    X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])

    # add labels
    model = k_means_model(k=k, selected_city=selected_city)
    X_pca["labels"] = model.named_steps["kmeans"].labels_.astype(str)
    X_pca.sort_values("labels", inplace=True)
    return X_pca



@app.callback(
    Output("pca-scatter", "figure"), Input("k-slider", "value"), Input('city-filter', 'value')
)
def serve_scatter_plot(k=2, selected_city=None):
    """Build 2D scatter plot of ``df`` with ``KMeans`` labels.

    Parameters
    ----------
    selected_city : string
        selects a city if needed for specialization
        
    k : int, default=2
        Number of clusters.
    """
    # Build Bar Chart
    fig = px.scatter(
        data_frame=get_pca_labels(k=k, selected_city=selected_city),
        x="PC1",
        y="PC2",
        color="labels",
        opacity=0.35,
        title="PCA Representation of Customer Segments"
    )
    fig.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",

    font=dict(
        family="Arial",
        color="#17365D"
    ),

    title_font=dict(
        size=20,
        color="#0B3B82"
    ),

    margin=dict(
        l=50,
        r=30,
        t=50,
        b=50
    ),

    legend=dict(
        bgcolor="rgba(255,255,255,0)"
    )

    )
    return fig


# # Cluster Distribution Function & Callback

# In[9]:


def get_cluster_labels(k=2, selected_city=None):  
    """Getting Cluster labels
    ``KMeans`` labels.

    Parameters
    ----------
    selected_city : string
        selects a city if needed for specialization
        
    k : int, default=2
        Number of clusters.
    """
    # get features
    filt_df = df[df['customer_city'] == selected_city] if selected_city else df
    X=filt_df.select_dtypes(include=['int64', 'float64'])

    # add cluster labels
    model = k_means_model(k=k, selected_city=selected_city)
    X["clusters"] = model.named_steps["kmeans"].labels_
    distribution = X["clusters"].value_counts()
    return distribution



@app.callback(
    Output("bar-plot", "figure"), Input("k-slider", "value"), Input('city-filter', 'value')
)
def serve_group_plot(k=2, selected_city=None):
    """Plot a Group bar plot of ``Inspection`` to view customer behaviour.

    Parameters
    ----------
    k : int, default=2
        Number of clusters.
    """
    # Build Bar Chart
    fig = px.bar( 
        data_frame=get_cluster_labels(k=k, selected_city=selected_city),
        orientation="h",
        title="Cluster Distribution"
    )
    fig.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",

    font=dict(
        family="Arial",
        color="#17365D"
    ),

    title_font=dict(
        size=20,
        color="#0B3B82"
    ),

    margin=dict(
        l=50,
        r=30,
        t=50,
        b=50
    ),

    legend=dict(
        bgcolor="rgba(255,255,255,0)"
    ), xaxis_title="Count", yaxis_title="Clusters")
    return fig


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


# Run the app
if __name__ == '__main__':
    app.run(debug=True)


