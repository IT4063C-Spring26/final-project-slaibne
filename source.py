#!/usr/bin/env python
# coding: utf-8

# # Annuity Product Selection Analysis Using Customer Behavior and Macroeconomic Conditions
# 
# ![Banner](./assets/banner.jpeg)

# ## 1. Project Overview

# ### 1.1 Topic
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

# ### 1.2 Project Question
# How do customer demographic(age and income) and macroeconomic conditions(interest rates) influence the liklihood of selecting fixed, variable, or indexed annuity products?
# 
# Addition questions:
# - How does annuity product selection vary across age groups (<40, 40–55, 55+)?
# - Does income level significantly affect the likelihood of choosing higher-risk products (variable annuities)?
# - How do changes in interest rates over time impact the distribution of annuity product types?
# - Which factors (age, income, or economic conditions) have the greatest predictive power in determining product selection?

# ### 1.3 Expected Answer / Hypothesis
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

# ### 1.4 Data Sources
# 1. Bank Marketing Dataset (Kaggle)
# - Customer demographics and behavioral features
# - Used as the foundation for customer profiles
# 
# 2. FRED Economic Data (API)
# - Federal Funds Rate (FEDFUNDS)
# - Used to model macroeconomic conditions
# 
# 3. Simulated Annuity Dataset
# - Generated within this notebook
# - Combines customer + economic data to assign product choices

# ### 1.5 Approach and Analysis

# This project follows an end-to-end data analysis workflow designed to evaluate how customer characteristics and macroeconomic conditions influence annuity product selection.
# 
# The workflow includes:

# 1. **Data acquisition and preparation**  
#    The Bank Marketing dataset is loaded as the customer-level foundation, and FRED interest rate data is retrieved as the macroeconomic layer.
# 
# 2. **Feature engineering**  
#    Since direct income and risk tolerance are not available, income and risk score are engineered using customer demographics, job category, loan indicators, prior engagement, and macroeconomic conditions.
# 
# 3. **Simulation of annuity product selection**  
#    Fixed, variable, and indexed annuity product choices are generated using probabilistic logic based on risk score, retirement horizon, and interest rate environment.
# 
# 4. **Exploratory data analysis**  
#    Visualizations and statistical summaries are used to examine product selection patterns across age groups, income levels, risk scores, and economic conditions.
# 
# 5. **Machine learning implementation**  
#    Classification models are trained to predict annuity product selection. The models are evaluated using accuracy, precision, recall, F1-score, confusion matrices, and feature importance.
# 
# 6. **Business interpretation**  
#    Final insights are interpreted in the context of financial services reporting, customer segmentation, and annuity product strategy.
# 
# This workflow reflects how data can be transformed from raw customer records into an analytical dataset that supports reporting, segmentation, and predictive modeling in a financial services environment.

# ## 2. Setup

# ### 2.1 Imports

# In[1]:


# The following libraries are used for data manipulation, visualization, and machine learning:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import seaborn as sns


from dotenv import load_dotenv
from pandas_datareader import data as pdr

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


np.random.seed(42)


# A random seed is set to make simulated values reproducible. This ensures that income generation, year assignment, and simulated annuity product assignment remain consistent across notebook runs.

# ### 2.2 Environment Variables
# 

# This project uses the Federal Reserve Economic Data (FRED) API to retrieve macroeconomic indicators.
# 
# The API key is stored securely using environment variables to prevent exposing sensitive information in the notebook or repository.
# 
# The key is loaded from a `.env` file using the `python-dotenv` package.

# In[3]:


load_dotenv()

api_key = os.getenv("FRED_API_KEY")

if api_key is None:
    raise ValueError("FRED_API_KEY not found in environment variables.")
else:
    print("FRED_API_KEY successfully loaded.")


# ## 3. Data Loading, Preparation, and Validation

# ### 3.1 Load Bank Marketing Dataset

# The Bank Marketing dataset is used as the foundational customer dataset for this analysis. It provides demographic, financial, and behavioral attributes that are later transformed into features relevant for annuity product selection.

# In[4]:


bank_df = pd.read_csv("data/bank.csv", sep=",")


# ### 3.2 Initial Data Inspection
# 

# Initial inspection is performed to understand the structure, data types, and distributions of the dataset.
# 
# This step helps identify:
# - data types for each feature
# - potential missing values
# - distributions and outliers in numerical variables

# In[5]:


bank_df.head()
bank_df.info()
bank_df.describe()


# The Bank Marketing dataset contains 11,162 customer records with 17 features describing demographic, financial, and behavioral attributes. The dataset includes both numerical variables(age, balance, duration, campaign) and categorical variables (job, marital status, education, loan status). No missing values are present but several categorical fields require encoding for machine learning. The dataset will serve as the primary customer behavioral dataset for modeling annuity product selection. 

# In[6]:


print("Shape:", bank_df.shape)
print("Missing values:", bank_df.isnull().sum().sum())
print("Duplicate rows:", bank_df.duplicated().sum())
bank_df.head()


# The dataset contains no missing values. Features include both categorical and numerical variables representing customer demographics and financial behavior.

# The raw Bank Marketing dataset was inspected before transformation. This confirms the number of rows and columns, checks whether missing values exist, and identifies whether duplicate records are present before feature engineering begins.

# In[7]:


categorical_cols = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome", "deposit"
]


# Categorical variables are explicitly converted to the 'category' data type to improve memory efficiency and ensure proper handling during exploratory data analysis. This step does not perform encoding but prepares the dataset for grouping and visualization.

# ### 3.3 Load FRED Interest Rate Data

# Macroeconomic data is retrieved from the Federal Reserve Economic Data (FRED) API.
# 
# The Federal Funds Rate is used as a proxy for the broader interest rate environment, which influences annuity product attractiveness.
# 
# A time range is defined to align economic conditions with the simulated customer dataset.

# In[ ]:


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2024, 12, 31)

fred_data = pdr.DataReader("FEDFUNDS", "fred", start, end)
fred_data = fred_data.resample("YE").mean()

fred_data.head()


# In[9]:


annual_rates = fred_data["FEDFUNDS"].to_dict()


# Annual average interest rates are computed and converted into a dictionary for mapping to customer observations.

# In[10]:


annual_rates_df = pd.DataFrame(
    list(annual_rates.items()),
    columns=["year", "interest_rate"]
)

annual_rates_df.head()


# In[11]:


print("FRED years:", annual_rates_df["year"].min(), "-", annual_rates_df["year"].max())
print("Missing interest rates:", annual_rates_df["interest_rate"].isnull().sum())
annual_rates_df.describe()


# In[16]:


fred_data.head()
fred_data.columns


# In[14]:


# Reset index safely
fred_data = fred_data.reset_index()

# Rename index column if needed
if "DATE" not in fred_data.columns:
    fred_data.rename(columns={fred_data.columns[0]: "DATE"}, inplace=True)

# Extract year
fred_data["year"] = fred_data["DATE"].dt.year

# Compute yearly average interest rate
annual_rates = fred_data.groupby("year")["FEDFUNDS"].mean().to_dict()


# The FRED data was aggregated to yearly average Federal Funds Rates so it could be joined to customer records by assigned year. Validation checks confirm that the yearly rate table contains no missing interest rate values and covers the expected date range.

# ### 3.4 Create Base Customer Dataset
# 

# The base dataframe is created from the original Bank Marketing dataset and serves as the main working dataset for feature engineering. This keeps the raw source data (bank_df) separate from the transformed analytical dataset.

# In[18]:


base = bank_df.copy()

base.describe()

base["customer_id"] = range(1, len(base) + 1)


# In[19]:


print("Unique customer IDs:", base["customer_id"].nunique())
print("Total rows:", len(base))
print("Customer ID is unique:", base["customer_id"].is_unique)


# A unique customer_id was created to support traceability and future merging. The validation check confirms that each row has one unique identifier.

# ## 4. Feature Engineering and Simulation
# 

# The unique customer_id was created in the base dataset preparation step. It is retained here as a stable identifier throughout feature engineering and modeling.

# ### 4.2 Create Income Estimate

# The Bank Marketing dataset does not contain a direct income variable. To support annuity product analysis, income is estimated using job category as a baseline indicator of earning potential.
# 
# This is an engineered approximation, not a directly observed value. The purpose is to create a realistic financial capacity variable that can be used in downstream segmentation and modeling.

# In[ ]:


job_income_map = {
    "management": 90000,
    "technician": 70000,
    "admin.": 50000,
    "services": 40000,
    "blue-collar": 45000,
    "retired": 30000,
    "student": 25000,
    "unemployed": 20000
}

base["base_income"] = base["job"].map(job_income_map)

# Use median because income is usually skewed; mean would be pulled upward by high-income jobs.
base["base_income"] = base["base_income"].fillna(base["base_income"].median())


# Income is derived from education level and adjusted using macroeconomic conditions.

# Median imputation is used as a fallback for any job categories that are not included in the mapping. Median is preferred over mean because income-related variables are often skewed, and the median is less affected by high-income outliers.

# In[ ]:


base[["job", "base_income"]].drop_duplicates().sort_values("base_income")


# In[ ]:


print("Missing base_income:", base["base_income"].isnull().sum())
base["base_income"].describe()


# Base income was assigned using job category because the source dataset does not include direct income. The validation check confirms that every customer received a base income value after mapping and median fallback handling.

# ### 4.3 Add Macroeconomic Interest Rate Features

# Each customer record is assigned a simulated observation year. That year is then mapped to the corresponding annual average Federal Funds Rate from FRED.
# 
# This allows customer-level records to be enriched with a macroeconomic context variable.

# In[20]:


base["year"] = np.random.choice(list(annual_rates.keys()), size=len(base))
base["interest_rate"] = base["year"].map(annual_rates)

print("Missing interest rates:", base["interest_rate"].isnull().sum())

base[["customer_id", "year", "interest_rate"]].head()


# In[ ]:


base.groupby("year")["interest_rate"].mean().head()


# Each customer record was assigned a year and matched to the corresponding yearly average Federal Funds Rate. The validation check confirms that no customer records are missing interest rate values after mapping. We are using mean to ensure that the mapped annual values are consistent. All records for the same year should have the same interest rate.

# In[ ]:


rate_min = base["interest_rate"].min()
rate_max = base["interest_rate"].max()

base["rate_scaled"] = (base["interest_rate"] - rate_min) / (rate_max - rate_min)


# In[ ]:


base["macro_adjustment"] = 0.9 +(base["rate_scaled"] * 0.2)


# In[ ]:


base[["interest_rate", "rate_scaled", "macro_adjustment"]].describe()


# In[ ]:


print("Rate scaled min/max:", base["rate_scaled"].min(), base["rate_scaled"].max())
print("Macro adjustment min/max:", base["macro_adjustment"].min(), base["macro_adjustment"].max())


# In[ ]:


base["income"] = (base["base_income"] * base["macro_adjustment"] +
                  np.random.normal(0, 10000, len(base))).clip(20000, 200000)

base.head()


# Income was not directly available in the dataset, so it was synthetically generated using a hybrid approach. A base income level was assigned using job categories to reflect individual earning potential. This base value was then adjusted using macroeconomic conditions derived from Federal Reserve interest rate data. Interest rates were normalized and used to create a macro adjustment factor, simulating the effect of broader economic conditions on income levels. Random noise was added to preserve variability across individuals. This approach ensures that income reflects both micro-level characteristics and macroeconomic trends.

# In[ ]:


def compute_risk_score(row):
    score = 0

    # Age (younger = more risk tolerance)
    if row["age"] < 40:
        score += 2
    elif row["age"] < 55:
        score += 1
    else:
        score -= 1

    # Income (higher = more risk capacity)
    if row["income"] > 90000:
        score += 2
    elif row["income"] > 50000:
        score += 1

    # Loans (constraints reduce risk tolerance)
    if row["housing"] == "yes":
        score -= 1
    if row["loan"] == "yes":
        score -= 1

    # Engagement (more engagement = openness to risk products)
    if row["previous"] > 0:
        score += 1

    # Normalize to 1–10 scale
    return np.clip(score + 5, 1, 10)

base["risk_score"] = base.apply(compute_risk_score, axis=1)


# In[ ]:


base["retirement_horizon"] = np.maximum(65 - base["age"], 1)


# In[ ]:


def assign_product(row):
    probs = {
        "fixed": 0.33,
        "variable": 0.33,
        "indexed": 0.34
    }

    if row["risk_score"] <= 3:
        probs["fixed"] += 0.30
    elif row["risk_score"] >= 7:
        probs["variable"] += 0.30

    if row["retirement_horizon"] < 10:
        probs["fixed"] += 0.20
    else:
        probs["variable"] += 0.10

    if row["interest_rate"] > 3:
        probs["fixed"] += 0.20
    else:
        probs["variable"] += 0.20

    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    return np.random.choice(list(probs.keys()), p=list(probs.values()))

base["annuity_product"] = base.apply(assign_product, axis=1)


# In[ ]:


products = pd.DataFrame({
    "product": ["fixed", "variable", "indexed"],
    "risk_level": ["low", "high", "medium"],
    "avg_return": [0.03, 0.07, 0.05],
    "fee_percent": [0.01, 0.025, 0.015]
})


# In[ ]:


model = base.copy()

product_meta = pd.DataFrame({
    "annuity_product": ["fixed", "variable", "indexed"],
    "risk_level": ["low", "high", "medium"],
    "avg_return": [0.03, 0.07, 0.05]
})

model = model.merge(product_meta, on="annuity_product")
print(model.columns)
model.head()


# The base dataframe is used as an intermediate feature-engineerign layer where customer attributes and macroeconomic variables are constructed and validated. The model dataframe represents the finalized modeling dataset, including the target variable (annuity_product) and any additional product level attributes and is used for machine learning and final analysis.

# In[ ]:


print(base.columns)
base.head()


# # Exploratory Data Analysis

# 
# This exploratory data analysis examines how customer demographics, financial capacity, behavioral indicators, and macroeconomic conditions influence annuity product selection. The dataset combines:
# 
# Customer-level data from the bank marketing dataset
# Simulated financial variables (income, risk score, retirement horizon)
# Macroeconomic indicators (interest rates)
# 
# The objective of this EDA is to:
# 
# Understand variable distributions
# Identify relationships between features
# Detect data quality issues
# Prepare the dataset for machine learning

# This bar chart examines how annuity product selection varies across age groups. Customers are segmented into three life stages: under 40, 40–55, and 55+.
# 
# 

# In[ ]:


# Create age groups
base["age_group"] = pd.cut(base["age"], bins=[0, 40, 55, 100], labels=["<40", "40-55", "55+"])

plt.figure()
sns.countplot(data=base, x="age_group", hue="annuity_product")
plt.title("Annuity Product Selection by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()


# The visualization shows a clear behavioral trend:
# 
# Younger customers (<40) tend to select more variable annuities
# Middle-aged customers show a more balanced distribution
# Older customers (55+) prefer fixed annuities
# 
# This aligns with financial theory, where risk tolerance decreases with age and proximity to retirement increases demand for stable income products.

# This boxplot shows how income levels vary across different annuity product types.
# 
# 

# In[ ]:


plt.figure()
sns.boxplot(data=base, x="annuity_product", y="income")
plt.title("Income Distribution by Annuity Product Type")
plt.xlabel("Annuity Product")
plt.ylabel("Income")
plt.show()


# Key observations:
# 
# Customers selecting variable annuities generally have higher median income
# Fixed annuity customers tend to have lower income levels
# Indexed products fall in the middle range
# 
# This suggests that income is strongly associated with risk capacity, where higher-income individuals are more likely to accept market exposure in exchange for higher potential returns.

# This line plot shows how annuity product demand changes over time under different interest rate environments.
# 
# 

# In[ ]:


rate_trend = base.groupby(["year", "annuity_product"]).size().unstack()

rate_product = base.groupby(["year", "annuity_product"]).size().unstack()

rate_product.plot(kind="line", marker="o")
plt.title("Annuity Product Selection Over Time (Interest Rate Environment Proxy)")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend(title="Product Type")
plt.show()


# Findings:
# 
# During high interest rate periods (e.g., 2022–2024), fixed annuity demand increases
# During low interest rate periods (e.g., 2020–2021), variable annuities become more common
# Indexed products remain relatively stable across time
# 
# This demonstrates that macroeconomic conditions (interest rates from FRED) influence product preference at the market level, shifting demand rather than changing individual risk profiles.

# Below is a correlation heatmap that highlights the relationships between demographic, financial, behavioral, and engineered features in the dataset.

# In[ ]:


plt.figure(figsize=(10,6))

numeric_cols = [
    "age", # core demographic
    "balance", # raw financial behavior
    "base_income", # engineered baseline income
    "income", # final income after macro adjustment and noise
    "interest_rate", # macroeconomic factor
    "rate_scaled", # transformation
    "macro_adjustment", # macroeconomic adjustment
    "retirement_horizon", # lifetime financial planning # lifecycle
    "risk_score", # risk assessment # key engineered feature
    "campaign", # marketing campaign  # behavioral
    "previous" # previous contact history # behavioral
]

corr = base[numeric_cols].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Engineered and Source Features")
plt.show()


# Several strong correlations are expected due to the feature engineering process. For example, age and retirement horizon exhibit a near-perfect negative correlation (-0.99), as retirement horizon was directly derived from age. Similarly, base income and income show a strong positive correlation (0.88), reflecting that income was generated using base income with macroeconomic adjustments.
# 
# More importantly, the heatmap reveals meaningful behavioral relationships. Risk score shows a moderate positive correlation with income (0.50) and base income (0.46), indicating that higher-income individuals tend to have higher risk tolerance. Additionally, risk score is negatively correlated with age (-0.62) and positively correlated with retirement horizon (0.63), suggesting that younger individuals with longer investment horizons are more likely to take on risk. These findings align with established financial theory and confirm that the engineered risk score is capturing realistic customer behavior.
# 
# Interest rates show minimal correlation with most variables, which is expected since they represent external macroeconomic conditions rather than individual customer attributes. Their influence is incorporated during the annuity product assignment process rather than directly affecting demographic features.
# 
# Finally, behavioral variables such as campaign and previous contact attempts exhibit weak correlations with financial features, indicating that they capture independent aspects of customer behavior. This diversity of signals is beneficial for predictive modeling, as it reduces redundancy and improves model generalization.
# 
# Overall, the heatmap validates both the internal consistency of the engineered features and the presence of meaningful relationships that will support downstream machine learning models.

# In[ ]:


# To avoid mulicollinearity for cleaner model later on
drop_cols = ["base_income", "rate_scaled", "macro_adjustment"]


# This boxplot evaluates whether the engineered risk_score meaningfully influences annuity product assignment. 

# In[ ]:


plt.figure(figsize=(8, 5))
sns.boxplot(data=base, x="annuity_product", y="risk_score")
plt.title("Risk Score Distribution by Annuity Product Type")
plt.xlabel("Annuity Product")
plt.ylabel("Engineered Risk Score")
plt.show()


# The median risk score for customers selecting variable annuities is higher than for those selecting fixed or indexed annuities, indicating that higher-risk individuals are more likely to be associated with variable products. The interquartile range is also shifted upwards for variable annuities, suggesting a concentrated distribution of higher-risk customers. In contrast, the interquartile ranges for fixed and indexed annuities exhibit broader distributions with lower median values, reflecting more conservative or moderate risk profiles.
# However, there is noticeable overlap between all three product categories, indicating that risk score is not the sole determinant of product selection. This aligns with the project design, where additional factors such as retirement horizon and macroeconomic conditions (interest rates) also influence the probabilistic assignment of annuity products.

# In[ ]:


base.groupby("annuity_product")["risk_score"].agg(["mean", "median", "std"])


# This aggregation summarizes the distribution of risk scores across annuity product types using the mean, median, and standard deviation. The mean provides the average risk level within each product category, while the median represents the typical customer and is less sensitive to extreme values. The standard deviation measures the variability of risk scores within each group.
# 
# The results show that variable annuities have both the highest mean (6.65) and median (7), indicating that customers selecting variable products tend to have higher risk tolerance. Fixed and indexed annuities have lower and similar medians (6), suggesting more moderate risk profiles.
# 
# Additionally, the standard deviation is slightly lower for variable annuities, indicating that high-risk customers are more consistently grouped within this product type. In contrast, fixed annuities show a wider spread of risk scores, suggesting a more diverse set of customer risk profiles.
# 
# These findings reinforce the validity of the engineered risk score and confirm that it meaningfully influences annuity product assignment while still allowing for realistic variation in customer behavior.

# # Data Cleaning and Transformation
# This section prepares the dataset for analysis and machine learning by addressing missing values, duplicates, outliers, and data type inconsistencies. The goal is to ensure data quality, consistency, and suitability for modeling.

# In[ ]:


base.isnull().sum()


# No missing values were identified across any variables, including engineered features such as income, risk score, and macroeconomic adjustments. This indicates that the data integration and feature engineering processes were successfully implemented without introducing null values.
# 
# As a result, no imputation or row removal was required, and the dataset retains its full size for analysis and modeling.
# 
# The absence of missing values is expected because key variables such as income and risk score were generated programmatically during the feature engineering stage, ensuring full population coverage across all observations.

# In[ ]:


base.duplicated().sum()


# The dataset was evaluated for duplicate records to ensure that each observation represents a unique customer instance. No duplicate rows were identified, indicating that the dataset maintains one-to-one integrity across observations. As a result, no duplicate removal was required.

# In[ ]:


base[["income", "balance", "duration", "campaign"]].describe()


# Income :
# - Is within reasonable bounds
# - Was already clipped during generation

# Balance: 
# - Extremely large values -> skew average
# - Negative balances -> can distort income proxy logic

# In[ ]:


base["balance"] = base["balance"].clip(lower=0, upper=50000)


# Duration :
# - Max = 3881 (This is extreme compared to typical range.)

# In[ ]:


base["duration"] = base["duration"].clip(upper=2000)


# Campaign :
# - Max = 63 contacts (Unrealistic when compared to the mean ~2.5)

# In[ ]:


base["campaign"] = base["campaign"].clip(upper=20)


# Outliers were identified by examining summary statistics of key numerical variables, including income, balance, duration, and campaign. While income values were within a realistic range due to prior clipping during feature engineering, other variables exhibited extreme values that could negatively impact analysis and modeling.
# 
# The balance variable contained both negative values and extreme positive outliers (maximum of 81,204). Negative balances may represent overdrafts but are not suitable for modeling financial capacity in this context, so values were clipped at a lower bound of zero. Additionally, an upper bound of 50,000 was applied to reduce the influence of extreme values.
# 
# The duration variable showed a maximum value of 3,881, significantly higher than the typical range, indicating potential outliers in call duration. These values were clipped at 2,000 to prevent distortion in analysis.
# 
# Similarly, the campaign variable contained extreme values (maximum of 63), which are not representative of typical marketing contact frequency. These values were capped at 20 to maintain realistic behavioral patterns.
# 
# Clipping was chosen instead of removing observations to preserve dataset size while minimizing the impact of extreme values. This approach ensures that statistical summaries and machine learning models are not disproportionately influenced by outliers.

# The decision to retain and adjust outliers rather than remove them aligns with real-world financial datasets, where extreme values may exist but should be controlled rather than discarded.

# 

# ## Data Type and Feature Transformations

# In[ ]:


# Convert categorical variables
categorical_cols = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "poutcome", "deposit", "age_group"
]

for col in categorical_cols:
    base[col] = base[col].astype("category")


# Create rate environment (macro feature)
base["rate_environment"] = np.where(
    base["interest_rate"] > 3,
    "high_rate",
    "low_rate"
)


# Data type and feature transformations were applied to improve the structure, interpretability, and modeling readiness of the dataset. Categorical variables such as job, marital status, education, and loan indicators were explicitly converted to the category data type. This improves memory efficiency and prepares the dataset for encoding during the machine learning stage.
# 
# In addition to data type conversions, feature transformations were used to enhance interpretability and align with real-world financial decision-making. The age_group variable was used to segment customers into lifecycle categories (<40, 40–55, 55+), reflecting differences in investment behavior and retirement proximity.
# 
# A new macroeconomic feature, rate_environment, was created by categorizing interest rates into high-rate and low-rate conditions. This transformation simplifies the interpretation of macroeconomic effects and reflects how financial institutions typically evaluate market conditions.
# 
# These transformations ensure that the dataset is not only clean but also structured in a way that supports both exploratory analysis and predictive modeling.

# > Continuous variables were preserved for modeling precision, while categorical transformations were introduced to enhance interpretability and support segmentation analysis.

# # Machine Learning Plan

# This project will use a multi-class classification approach to predict annuity product selection (fixed, variable, or indexed) based on customer demographics, financial attributes, and macroeconomic conditions. Classification is appropriate because the target variable represents discrete product categories rather than continuous values.
# 
# Multiple models will be explored to compare performance and interpretability. Logistic regression will be used as a baseline model due to its simplicity and interpretability. Decision trees will be applied to capture non-linear relationships between variables such as risk score, income, and interest rate. Additionally, a random forest model will be considered to improve predictive performance through ensemble learning and reduce overfitting.

# The model will use a combination of demographic, financial, behavioral, and macroeconomic features. Key predictors include age, income, risk score, retirement horizon, and interest rate. Engineered features such as age group and rate environment are included to capture lifecycle segmentation and macroeconomic regimes. The target variable is annuity product type, representing the financial product selected by each customer.

# ## Challenges

# ### Synthetic Data

# One key challenge is that the annuity product variable is generated through a probabilistic simulation rather than observed real-world behavior. This introduces a risk that the model may learn patterns that reflect the simulation logic rather than true customer behavior.

# ### Multicollinearity

# Several engineered features, such as income and base income or rate scaled and macro adjustment, exhibit high correlation. This may introduce multicollinearity, particularly in linear models such as logistic regression, and may require feature selection or regularization.

# ### Categorical Encoding

# The dataset contains multiple categorical variables that must be encoded into numerical form before training machine learning models. Improper encoding could lead to loss of information or model bias.

# ### Class Balance

# The distribution of annuity product types may be imbalanced depending on the simulation parameters, which can affect model performance and bias predictions toward the majority class.

# ### How these challenges will be addressed

# These challenges will be addressed through structured preprocessing and model design. A machine learning pipeline will be implemented using scikit-learn to handle missing values, encode categorical variables, and scale numerical features consistently. Multicollinearity will be mitigated through feature selection or by using tree-based models that are less sensitive to correlated inputs. Class imbalance will be evaluated and, if necessary, addressed using techniques such as class weighting or resampling. Finally, multiple models will be tested and compared to ensure that results are robust and not dependent on a single modeling approach.

# ### Evaluation Metrics

# Model performance will be evaluated using classification metrics including accuracy, precision, recall, and F1-score. A confusion matrix will also be used to assess how well the model distinguishes between different annuity product types. These metrics provide a comprehensive evaluation of both overall performance and class-level prediction quality.

# # Machine Learning Implementation

# In[ ]:


# Defining features and target
target = "annuity_product"

feature_cols = [
    "age",
    "income",
    "risk_score",
    "retirement_horizon",
    "interest_rate",
    "campaign",
    "previous",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "deposit",
    "age_group",
    "rate_environment"
]

X = base[feature_cols]
y = base[target]


# These features were selected because they represent the major drivers of annuity product selextion in the project: customer demographics, financial capacity, behavioral engagement, debt obligations, lifecycle stage, and macroeconomic conditions. The target variable is annuity_product, which classifies customers into fixed, variable, or indexed annuity product categories.

# ## Train / Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# An 80/20 train-test split was used so the model could learn from most of the dataset while preserving unseen data for evaluation. Stratified sampling was used because the target variable contains multiple product categories, and stratification helps preserve the same product distribution in both the training and testing sets.

# ## Preprocessing pipelines

# In[ ]:


numeric_features = [
    "age",
    "income",
    "risk_score",
    "retirement_horizon",
    "interest_rate",
    "campaign",
    "previous"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "deposit",
    "age_group",
    "rate_environment"
]


# In[ ]:


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# Numerical variables are imputed using the median because financial variables such as income can be skewed by high-value customers. Standard scaling is applied so large-scale variables like income do not dominate smaller-scale variables like risk score or interest rate. Categorical variables are imputed using the most frequent value and encoded using one-hot encoding so machine learning models can interpret them numerically.

# ## Train multiple models

# In[ ]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}


# To address class imbalance, class weighting can be applied to penalize misclassification of underrepresented product types. This encourages the model to better learn patterns associated with fixed and indexed annuities rather than defaulting to the dominant variable category.

# In[ ]:


model_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    model_pipelines[name] = pipeline


# ## Evaluate models

# In[ ]:


for name, pipeline in model_pipelines.items():
    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print(name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred))


# Each model was evaluated on the same test set using accuracy, precision, recall, and F1-score. Accuracy measures overall correctness, while precision, recall, and F1-score provide more detail about how well each annuity product category is predicted.

# The machine learning results indicate that all models consistently predict variable annuities with high recall, while struggling to accurately classify fixed and indexed products. This suggests that the dataset contains a dominant pattern in which a large proportion of customers fall into profiles associated with variable annuity selection.
# 
# Logistic regression achieved the highest overall accuracy (51%), but this performance is largely driven by its ability to correctly classify variable annuities (recall = 0.93). However, it completely fails to identify indexed annuities, indicating that linear decision boundaries are insufficient to distinguish more nuanced product categories.
# 
# The decision tree model provides slightly more balanced performance across classes, capturing non-linear relationships between risk score, income, and interest rate. However, it still exhibits a strong bias toward variable annuities, reflecting the underlying data distribution.
# 
# The random forest model attempts to generalize across multiple decision trees and produces more balanced precision across categories, but at the cost of slightly lower accuracy. This suggests that while ensemble methods improve robustness, they cannot fully overcome the class imbalance present in the dataset.
# 
# Overall, these results indicate that annuity product selection is not evenly distributed across categories. Instead, the combination of demographic factors, financial characteristics, and macroeconomic conditions strongly favors variable annuities in the simulated dataset. Fixed and indexed products appear to require more specific conditions, making them harder for models to distinguish.
# 
# This outcome is consistent with the probabilistic simulation logic used to generate the dataset and highlights the importance of considering class distribution when evaluating model performance.

# ## Confusion matrix

# In[ ]:


best_model = model_pipelines["Random Forest"]

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# The confusion matrix provides a detailed view of model performance across annuity product categories. The model demonstrates strong performance in identifying variable annuities, correctly classifying 888 instances. However, it struggles significantly with fixed and indexed annuities, frequently misclassifying them as variable.
# 
# A large number of fixed annuities (409 instances) and indexed annuities (362 instances) are incorrectly predicted as variable. This indicates that the model is biased toward the variable class, which is consistent with the overall distribution of the dataset and the probabilistic logic used to generate product assignments.
# 
# Indexed annuities are particularly difficult for the model to distinguish, as they share characteristics with both fixed and variable products. This overlap results in low classification accuracy for this category and highlights the complexity of modeling hybrid financial products.
# 
# These results suggest that while the model effectively captures dominant patterns associated with higher-risk, growth-oriented customers, it has difficulty identifying more nuanced or conservative profiles. This reflects both the underlying data distribution and the inherent complexity of annuity product segmentation.
# 
# Overall, the confusion matrix reveals that model performance is strongly influenced by class imbalance and overlapping feature distributions, rather than purely model limitations.

# >To improve classification performance, techniques such as class weighting or resampling could be applied to better balance predictions across product types. Additionally, further feature engineering or threshold tuning may help distinguish indexed annuities more effectively.

# # Feature Importance

# In[ ]:


# Get trained model
rf_pipeline = model_pipelines["Random Forest"]

# Extract model and preprocessor
rf_model = rf_pipeline.named_steps["model"]
preprocessor = rf_pipeline.named_steps["preprocessor"]

# Get feature names AFTER preprocessing
numeric_features = [
    "age", "income", "risk_score", "retirement_horizon",
    "interest_rate", "campaign", "previous"
]

categorical_features = [
    "job", "marital", "education", "housing",
    "loan", "deposit", "age_group", "rate_environment"
]

# Get encoded categorical names
encoded_cat_features = preprocessor.named_transformers_["categorical"]\
    .named_steps["encoder"].get_feature_names_out(categorical_features)

# Combine all feature names
all_features = list(numeric_features) + list(encoded_cat_features)

# Get importance values
importances = rf_model.feature_importances_

# Create dataframe
feature_importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feature_importance_df.head(15)


# In[ ]:


top_features = feature_importance_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.show()


# The feature importance analysis from the Random Forest model reveals that income is the most influential predictor of annuity product selection, followed by interest rate, age, and retirement horizon. This indicates that both financial capacity and macroeconomic conditions play a dominant role in determining product choice.
# 
# The high importance of interest rates supports the project’s hypothesis that external economic conditions significantly impact annuity demand. Similarly, age and retirement horizon reflect lifecycle effects, where younger customers with longer time horizons are more likely to select higher-risk products.
# 
# Interestingly, behavioral variables such as campaign and previous interactions also contribute meaningfully, suggesting that customer engagement and prior financial activity influence decision-making.
# 
# The engineered risk_score feature shows lower importance than expected. This is likely because it is derived from other variables such as age and income, which are already included in the model. As a result, the model relies more on the original features rather than the aggregated risk score.
# 
# Overall, the results demonstrate that annuity product selection is driven by a combination of financial capacity, economic conditions, and customer lifecycle characteristics, aligning closely with real-world financial behavior.

# # Final Section: Insights, Conclusions, and Business Implications

# Final Insights and Conclusions
# 
# This project set out to analyze how customer demographics and macroeconomic conditions influence annuity product selection. By integrating customer-level behavioral data with economic indicators and a simulated annuity dataset, several meaningful insights were identified.
# 
# 1. Financial Capacity is the Strongest Driver
# 
# Income emerged as the most important predictor of annuity product selection. Customers with higher incomes were significantly more likely to be associated with higher-risk products such as variable annuities, while lower-income individuals tended toward more stable products like fixed annuities. This indicates that financial capacity plays a central role in determining risk tolerance and investment strategy.
# 
# 2. Macroeconomic Conditions Significantly Influence Product Selection
# 
# Interest rates were consistently one of the top predictors across both exploratory analysis and machine learning models. The data shows that:
# 
# Higher interest rate environments are associated with increased selection of fixed annuities
# Lower interest rate environments shift preference toward variable annuities
# 
# This confirms that external economic conditions materially impact customer decision-making, reinforcing the importance of incorporating macroeconomic data into financial product analysis.
# 
# 
# 3. Lifecycle Factors Shape Risk Behavior
# 
# Age and retirement horizon together form a strong lifecycle signal:
# 
# Younger customers with longer retirement horizons tend to select higher-risk products
# Older customers nearing retirement shift toward more conservative options
# 
# This reflects well-established financial planning principles and demonstrates that the model captures realistic behavioral patterns.
# 
# 4. Behavioral Data Adds Predictive Value
# 
# Variables derived from the bank dataset, such as campaign (number of contacts) and previous (prior interactions), contributed meaningfully to prediction performance. This suggests that:
# 
# Customer engagement and interaction history influence financial decisions
# More engaged customers may be more open to complex or higher-risk financial products
# 
# This aligns with real-world financial services, where relationship depth and interaction frequency often drive product adoption.
# 
# 5. Engineered Risk Score Shows Redundancy
# 
# Although a composite risk_score was engineered to represent customer risk tolerance, it was less important than expected in the model. This is because the underlying variables used to construct it (such as age, income, and financial behavior) were already included individually.
# 
# As a result, the model relied more heavily on the raw features rather than the aggregated score, demonstrating a case of feature redundancy.

# ## Limitations

# While the project provides meaningful insights, several limitations should be acknowledged:
# 
# The annuity product dataset is simulated rather than real-world data
# Product assignment logic is based on assumed financial behavior, not observed outcomes
# The Bank Marketing dataset was repurposed and does not directly represent annuity customers
# Class imbalance is present, with the model favoring the “variable” product class
# Certain features (such as education or deposit status) showed low predictive value
# 
# These limitations highlight the challenges of working with proxy datasets and simulated environments.

# ## Future Improvements

# To enhance the accuracy and realism of the analysis, future work could include:
# 
# - Incorporating real annuity transaction data (if available)
# - Improving class balance through resampling techniques (SMOTE, class weighting)
# - Refining product assignment logic using more granular financial rules
# - Adding additional macroeconomic indicators (inflation, unemployment rates)
# - Exploring more advanced models (gradient boosting, XGBoost)
# - Building a recommendation system based on predicted probabilities

# ## Resources and References
# Kaggle - Bank Marketing dataset https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?resource=download
# Federal Reserve Economic Data (https://fred.stlouisfed.org/)
# ChatGPT

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

