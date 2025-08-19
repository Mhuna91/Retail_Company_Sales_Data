#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
# app.py

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st


# In[2]:


# STREAMLIT APP

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

st.title("Retail Company Sales Dashboard")


# In[8]:


file_path = "Retail_Company_Sales_Data.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

print(df.columns.tolist())


# In[9]:


df['Order Date'] = pd.to_datetime(df['Order Date'])


# In[10]:


region_filter = st.sidebar.multiselect(
    "Select Region", df['Region'].unique(), default=df['Region'].unique()
)
channel_filter = st.sidebar.multiselect(
    "Select Sales Channel", df['Sales Channel'].unique(), default=df['Sales Channel'].unique()
)

# Apply filters
filtered_df = df[
    (df['Region'].isin(region_filter)) & (df['Sales Channel'].isin(channel_filter))
]


# In[13]:


# Cache data loading
@st.cache_data
def load_data():
    file_path = r"C:\Users\HP\Desktop\ENDIP BIU\Retail_Company_Sales_Data.xlsx"
    df = pd.read_excel(file_path, engine="openpyxl")

    # Standardize column names: lowercase, no spaces
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])

    return df

df = load_data()


# In[14]:


# Sidebar filters

st.sidebar.header("Filters")
region_filter = st.sidebar.multiselect(
    "Select Region", df['region'].unique(), default=df['region'].unique()
)
channel_filter = st.sidebar.multiselect(
    "Select Sales Channel", df['sales_channel'].unique(), default=df['sales_channel'].unique()
)

# Apply filters
filtered_df = df[
    (df['region'].isin(region_filter)) & (df['sales_channel'].isin(channel_filter))
]


# In[15]:


# KPIs
total_sales = filtered_df['sales_amount'].sum()
total_orders = filtered_df['order_id'].nunique()
avg_discount = filtered_df['discount'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Avg Discount", f"{avg_discount:.2%}")

# Sales trend
st.subheader("Sales Trend Over Time")
sales_trend = (
    filtered_df.groupby('order_date')['sales_amount'].sum().reset_index()
)
fig_trend = px.line(
    sales_trend, x="order_date", y="sales_amount",
    title="Daily Sales Trend", markers=True
)
st.plotly_chart(fig_trend, use_container_width=True)


# In[16]:


# Sales by region
st.subheader("Sales by Region")
region_sales = filtered_df.groupby('region')['sales_amount'].sum().reset_index()
fig_region = px.bar(
    region_sales, x='region', y='sales_amount',
    title="Sales by Region", text_auto=True
)
st.plotly_chart(fig_region, use_container_width=True)

# Prophet forecast
st.subheader("30-Day Sales Forecast (Prophet)")
df_forecast = sales_trend.rename(columns={"order_date": "ds", "sales_amount": "y"})

m = Prophet()
m.fit(df_forecast)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

fig_forecast = px.line(forecast, x="ds", y="yhat", title="Forecasted Sales")
fig_forecast.add_scatter(
    x=df_forecast['ds'], y=df_forecast['y'],
    mode="markers", name="Actual Sales"
)
st.plotly_chart(fig_forecast, use_container_width=True)


# In[17]:


# Download option
st.download_button(
    "Download Forecast Data",
    forecast.to_csv(index=False).encode("utf-8"),
    "forecast_30_days.csv",
    "text/csv"
)


# In[ ]:




