#!/usr/bin/env python
# coding: utf-8

# # {Project Title TBA}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# Retail annuity products, specifically fixed, variable and indexed annuities, are designed to meet different financial needs related to risk tolerance, income stability, and long-term growth. However, product selection is oftern influenced by a combination of customer demographics (i.e age, income, and retirement horizon) and external economic conditions such as interest rates and inflation, rather than a clear data-driven matching process.
# 
# In practice, this leads to measurable inefficiencies:
# - Customers nearing retirement (ages 55–70) may select variable annuities despite shorter time horizons, exposing them to unnecessary market volatility.
# - Younger or higher-income individuals may underutilize growth-oriented products due to conservative defaults or lack of guidance.
# - Periods of rising interest rates can shift product demand toward fixed annuities, but is this relationship being quantified at the customer-segment level?
# 
# From a data management and reporting perspective, annuity providers maintain large volumes of structured data (customer profiles, transaction histories, and product allocations), but this data is often used descriptively (reporting past sales) rather than analytically (understanding why certain products are selected and whether those selections align with customer characteristics and market conditions).
# 
# This project aims to analyze how specific variables:
# - Age (grouped into pre-retirement vs retirement cohorts)
# - Income level (segmentented into quantiles)
# - Proxy indicators of risk tolerance (derived from financial behavior)
# - Macroeconomic conditions (interest rate levels over time)
# influence the likelihood of a customer selecting a particular annuity product type.
# 
# By integrating customer level financial data with external economic indicators, this analysis will identify statistically significant relationships between customer profiles, market conditions, and product selection patterns. The goal is predictive and explanatory insights that can support:
# - targeted product recommendations
# - improved segmentation strategies
# - and better alignment between customer needs and financial products
# 

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# How do customer demographic(age and income) and macroeconomic conditions(interest rates) influence the liklihood of selecting fixed, variable, or indexed annuity products?
# 
# Addition questions:
# - How does annuity product selection vary across age groups (<40, 40–55, 55+)?
# - Does income level significantly affect the likelihood of choosing higher-risk products (variable annuities)?
# - How do changes in interest rates over time impact the distribution of annuity product types?
# - Which factors (age, income, or economic conditions) have the greatest predictive power in determining product selection?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# - A predictive model outputting probabilities such as: “A 60-year-old with moderate income during high interest rate periods has a 72% probability of selecting a fixed annuity”
# 
# - Ranked feature importance:
#     - Age -> strongest predictor
#     - Interest rate -> secondary
#     - Income -> moderate
# 
# - Visual confirmation of trends: clear seperation of product types across demographic groups
# 
# 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# Economic Indicators (API) : from Federal Reserve Economic Data (https://fred.stlouisfed.org/)
# 
# Key variables: 
# - Federal interest rates
# - Inflation rate
# - Treasury yields
# 
# Assign economic conditions to each observation and create low rate vs high rate environments.
# 
# bank.csv from Bank Marketing Dataset on Kaggle
# https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?resource=download
# 
# Key variables:
# - age
# - job / income proxy
# - marital status
# = education
# 
# Represent customer demographic and financial profiles and create derived features.
# 
# Use pandas to generate a simulated annuity products dataset:
# - real annuity product selection data is proprietary
# - Customer attributes such as age, income, and prevailing interest rates were used to assign annuity product types in a way that reflects realistic market tendencies
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# This project will follow a structured data analytics workflow to examine how customer demographics and economic conditions influence annuity product selection.
# 
# 1. Data Acquisition and Integration
# 
# Three datasets will be used:
# 
# A customer financial behavior dataset (CSV file)
# Macroeconomic data retrieved via the Federal Reserve Economic Data
# A simulated annuity product dataset generated within the notebook
# 
# The customer dataset will provide demographic variables such as age and income. Economic data (e.g., interest rates) will be mapped to each observation using a time-based or scenario-based assignment. A simulated dataset will assign annuity product types (fixed, variable, indexed) to each customer using probabilistic rules based on financial assumptions.
# 
# 2. Data Cleaning and Preprocessing
# Handle missing or inconsistent values in customer data
# Normalize and clip income values to realistic ranges
# Convert continuous variables into categorical features:
# Age → grouped into (<40, 40–55, 55+)
# Income → segmented into low, medium, and high tiers
# Ensure all datasets are aligned in structure for merging
# 3. Feature Engineering
# Create derived variables:
# Age group
# Income tier
# Interest rate category (low vs high rate environment)
# Generate the target variable:
# annuity_product (fixed, variable, indexed) using probabilistic assignment
# Merge product metadata (risk level, return type, fee level) into the final dataset
# 4. Exploratory Data Analysis (EDA)
# 
# EDA will be conducted to identify patterns and relationships:
# 
# Distribution of annuity product types across age groups
# Relationship between income tiers and product selection
# Impact of interest rate levels on product distribution
# Correlation analysis between variables
# 
# Visualizations will include:
# 
# Bar charts (product type vs demographic groups)
# Heatmaps (income vs product type)
# Line plots (interest rates vs product trends)
# 5. Predictive Modeling
# 
# A classification model will be developed to predict annuity product selection:
# 
# Logistic regression and/or decision tree classifier
# Input variables:
# Age group
# Income tier
# Interest rate
# Output:
# Probability of selecting each annuity product type
# 6. Model Evaluation and Interpretation
# Evaluate model performance using accuracy and confusion matrix
# Analyze feature importance to determine which variables most influence product selection
# Interpret results in the context of financial decision-making
# 7. Insight Generation
# 
# The final step will translate analytical results into business insights:
# 
# Identify which customer segments are most likely to choose each product type
# Evaluate how economic conditions shift product preferences
# Discuss how these insights could support better targeting and product recommendations in an annuity business context

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# Kaggle - Bank Marketing dataset https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?resource=download
# Federal Reserve Economic Data (https://fred.stlouisfed.org/)
# ChatGPT

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


from dotenv import load_dotenv
from pandas_datareader import data as pdr


load_dotenv()

api_key = os.getenv("FRED_API_KEY")

if api_key is None:
    raise ValueError("FRED_API_KEY not found in environment variables.")
else:
    print("FRED_API_KEY successfully loaded.")


# In[2]:


bank_df = pd.read_csv("data/bank.csv", sep=",")

bank_df.head()
bank_df.info()
bank_df.describe()


# In[3]:


bank_df.isnull().sum()


# The Bank Marketing dataset contains 11,162 customer records with 17 features describing demographic, financial, and behavioral attributes. The dataset includes both numerical variables(age, balance, duration, campaign) and categorical variables (job, marital status, education, loan status). No missing values are present but several categorical fields require encoding for machine learning. The dataset will serve as the primary customer behavioral dataset for modeling annuity product selection. 

# In[4]:


# Convert categorical variables (important for ML later)
categorical_cols = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome", "deposit"
]

for col in categorical_cols:
    bank_df[col] = bank_df[col].astype("category")


# In[5]:


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2024, 12, 31)

fed_funds = pdr.DataReader("FEDFUNDS", "fred", start, end)


# In[6]:


fed_funds = fed_funds.reset_index()
fed_funds["year"] = fed_funds["DATE"].dt.year

# yearly average interest rate
annual_rates = fed_funds.groupby("year")["FEDFUNDS"].mean().to_dict()


# In[7]:


base = bank_df.copy()

base.describe()

# Add a unique customer ID for easier merging and analysis
base["customer_id"] = range(1, len(base) + 1)
base.head()


# In[8]:


base["balance"].hist(bins=50)
plt.title("Distribution of Account Balances")


# In[9]:


base["year"] = np.random.choice(list(annual_rates.keys()), size=len(base))
base["interest_rate"] = base["year"].map(annual_rates)


# In[10]:


rate_min = base["interest_rate"].min()
rate_max = base["interest_rate"].max()

base["rate_scaled"] = (base["interest_rate"] - rate_min) / (rate_max - rate_min)


# In[11]:


base["macro_adjustment"] = 0.9 +(base["rate_scaled"] * 0.2)


# In[12]:


base["income"] = (base["base_income"] * base["macro_adjustment"] +
                  np.random.normal(0, 10000, len(base))).clip(20000, 200000)

base.head()


# # Exploratory Data Analysis
# This exploratory data analysis examines how customer demographics (age, income, risk score) and macroeconomic conditions (interest rates) are associated with annuity product selection (fixed, variable, indexed). The goal is to identify distributional patterns, correlations, and potential segmentation behavior within the dataset.

# In[ ]:





# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

