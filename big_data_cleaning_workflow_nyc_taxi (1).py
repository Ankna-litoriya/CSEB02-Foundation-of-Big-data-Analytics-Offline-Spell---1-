# ============================================
# UNIT–1: Introduction to Big Data
# Activity 1: Design and Implementation of a Data Cleaning Workflow
# Course Outcome: CO1 – Building a Data Cleaning Pipeline
# Dataset: NYC Taxi Trip Records (Kaggle)
# ============================================

# --------------------------------------------------
# CELL 1: Import Required Libraries
# --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# CELL 2: Locate Dataset Directory (KaggleHub / Kaggle)
# --------------------------------------------------
# This path points to the KaggleHub cached dataset directory
base_path = "/root/.cache/kagglehub/datasets/nagasai524/nyc-taxi-trip-records-from-jan-2023-to-jun-2023/versions/1"

print("Files available in dataset directory:")
for file in os.listdir(base_path):
    print(file)


# --------------------------------------------------
# CELL 3: Load and Merge All CSV Files (Big Data Ingestion)
# --------------------------------------------------
csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

df_list = []

for file in csv_files:
    file_path = os.path.join(base_path, file)
    try:
        temp_df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        temp_df = pd.read_csv(file_path, encoding='latin1')
    df_list.append(temp_df)

# Combine all monthly datasets into one DataFrame
df = pd.concat(df_list, ignore_index=True)

print("\nCombined Dataset Shape:", df.shape)
df.head()


# --------------------------------------------------
# CELL 4: Initial Data Inspection
# --------------------------------------------------
df.info()


# --------------------------------------------------
# CELL 5: Statistical Data Profiling
# --------------------------------------------------
df.describe(include='all')


# --------------------------------------------------
# CELL 6: Missing Value Analysis
# --------------------------------------------------
missing_summary = df.isnull().sum()
missing_summary


# --------------------------------------------------
# CELL 7: Missing Value Heatmap (Data Messiness)
# --------------------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.show()


# --------------------------------------------------
# CELL 8: Duplicate Detection
# --------------------------------------------------
duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)


# --------------------------------------------------
# CELL 9: Structural Corrections
# --------------------------------------------------
# Convert pickup datetime to proper datetime format
if 'tpep_pickup_datetime' in df.columns:
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')

# Standardize payment type column
if 'payment_type' in df.columns:
    df['payment_type'] = df['payment_type'].astype(str).str.lower().str.strip()

print("Structural corrections applied")


# --------------------------------------------------
# CELL 10: Missing Value Treatment
# --------------------------------------------------
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled")


# --------------------------------------------------
# CELL 11: Outlier Detection and Handling (IQR Method)
# --------------------------------------------------
if 'trip_distance' in df.columns:
    Q1 = df['trip_distance'].quantile(0.25)
    Q3 = df['trip_distance'].quantile(0.75)
    IQR = Q3 - Q1

    before_rows = df.shape[0]
    df = df[(df['trip_distance'] >= Q1 - 1.5 * IQR) &
            (df['trip_distance'] <= Q3 + 1.5 * IQR)]
    after_rows = df.shape[0]

    print("Outliers removed:", before_rows - after_rows)


# --------------------------------------------------
# CELL 12: Duplicate Removal
# --------------------------------------------------
before_rows = df.shape[0]
df = df.drop_duplicates()
after_rows = df.shape[0]

print("Duplicate rows removed:", before_rows - after_rows)


# --------------------------------------------------
# CELL 13: Feature Scaling (Optional)
# --------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

if 'trip_distance' in df.columns:
    scaler = MinMaxScaler()
    df['trip_distance_scaled'] = scaler.fit_transform(df[['trip_distance']])

print("Feature scaling completed")


# --------------------------------------------------
# CELL 14: Save Fully Cleaned Dataset
# --------------------------------------------------
output_file = "nyc_taxi_cleaned_full.csv"
df.to_csv(output_file, index=False)

print("Cleaned dataset saved as:", output_file)


# --------------------------------------------------
# CELL 15: Performance Experiment – Pandas
# --------------------------------------------------
start_time = time.time()
_ = df.groupby(df.columns[0]).size()
pandas_time = time.time() - start_time

print("Pandas execution time:", pandas_time)


# --------------------------------------------------
# CELL 16: Performance Experiment – Dask
# --------------------------------------------------
import dask.dataframe as dd

ddf = dd.from_pandas(df, npartitions=8)

start_time = time.time()
_ = ddf.groupby(ddf.columns[0]).size().compute()
dask_time = time.time() - start_time

print("Dask execution time:", dask_time)


# --------------------------------------------------
# CELL 17: Performance Comparison Summary
# --------------------------------------------------
comparison = pd.DataFrame({
    'Framework': ['Pandas', 'Dask'],
    'Execution Time (seconds)': [pandas_time, dask_time]
})

comparison


# --------------------------------------------------
# END OF NOTEBOOK
# --------------------------------------------------