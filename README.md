# Customer Segmentation Using K-Means Clustering
Overview
This project uses customer transaction and behavioural data from the Olist Brazilian E-Commerce dataset to identify distinct customer segments using unsupervised machine learning.

The analysis combines RFM (Recency, Frequency, Monetary) analysis with additional customer behaviour features before applying K-Means clustering.

The goal is to understand different customer behaviours and translate the resulting clusters into meaningful business segments.

---

Problem Statement
Businesses often have customers with very different purchasing behaviours.

Instead of treating every customer the same, this project aims to answer:

Who are the most valuable customers?
Which customers are inactive?
Which customers are more likely to return?
Which customers appear dissatisfied?
How can customers be grouped based on their behaviour?

---

Dataset
The project uses the Brazilian E-Commerce Public Dataset by Olist.

The dataset contains information about orders, customers, products, payments, reviews, and delivery.

Due to the size of the raw dataset, the original files are not included in this repository.

Dataset:

[Olist Brazilian E-Commerce Dataset](YOUR_DATASET_LINK)

---

Data Pipeline
The project begins by querying and joining relevant tables from the Olist database.

The workflow includes:

Exploring the database
Joining relevant tables
Preparing customer-level data
Performing RFM analysis
Creating additional behavioural features
Merging the engineered features
Scaling numerical features
Evaluating different numbers of clusters
Training the final K-Means model
Interpreting the resulting customer segments

---

## Feature Engineering

### RFM Analysis

Three core RFM features were used:

- **Recency** — how recently a customer made a purchase
- **Frequency** — how often a customer made a purchase
- **Monetary** — how much a customer spent

Additional behavioural information was also incorporated, including customer review and delivery-related features.

---

## Clustering Method

K-Means clustering was used to group customers according to their behavioural characteristics.

Before applying K-Means, the numerical features were standardized using `StandardScaler`.

Different values of `k` were evaluated using:

- Inertia
- Silhouette Score

Based on the analysis, **5 clusters** were selected for the final model.

---
## Customer Segments

The five resulting clusters were interpreted as:

| Cluster | Segment |
|---|---|
| 0 | Inactive Customers |
| 1 | Repeat Customers |
| 2 | Dissatisfied Customers |
| 3 | Recent Customers |
| 4 | High-Value Customers |

### Inactive Customers

Customers with relatively low monetary value and high recency.

### Repeat Customers

Customers with higher purchasing frequency and relatively strong monetary value.

### Dissatisfied Customers

Customers associated with lower review scores and relatively longer delivery times.

### Recent Customers

Customers who purchased more recently and generally experienced shorter delivery times and relatively positive reviews.

### High-Value Customers

Customers with exceptionally high monetary value but relatively low purchasing frequency.

---

## Business Recommendations
A detailed Recommendation can be found inside the notebook

---

## Interactive Dashboard

An interactive Dash dashboard was developed to explore the customer segmentation results.

The dashboard provides:

- Customer-level summary statistics
- Cluster distribution
- PCA visualization of customer segments
- Dynamic K-Means cluster selection
- City-level filtering
- Interactive cluster analysis

### Dashboard Features

Users can adjust the number of clusters and filter customers by city to explore how customer segments change across different selections.

The dashboard was built using Dash, Plotly, and Bootstrap components.

---

The dashboard allows the segmentation results to be explored visually and provides a more accessible way to understand the customer groups.

[View Dashboard](https://customer-segmentation-with-interactive.onrender.com/)
