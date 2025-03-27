import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

# Seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()

# Generate simple data
num_samples = 200

data = {
    "ID": [f"EMP_{i:03d}" for i in range(1, num_samples + 1)],
    "Name": [fake.name() for _ in range(num_samples)],
    "Age": np.random.randint(22, 65, size=num_samples),
    "Salary": np.random.normal(50000, 10000, size=num_samples).astype(int),
    "Department": np.random.choice(
        ["Sales", "HR", "Tech", "Marketing", "Finance"],
        size=num_samples,
        p=[0.25, 0.15, 0.3, 0.15, 0.15]
    ),
    "Hire_Date": [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(num_samples)],
    "Performance_Score": np.random.choice(["High", "Medium", "Low"], size=num_samples, p=[0.3, 0.4, 0.3]),
    "Experience_Years": np.random.randint(1, 20, size=num_samples),
    "Gender": np.random.choice(["Male", "Female", "Other"], size=num_samples, p=[0.5, 0.4, 0.1]),
    "Sales_Q1": np.random.randint(10000, 100000, size=num_samples),
    "Sales_Q2": np.random.randint(10000, 100000, size=num_samples),
    "Email": [fake.email() for _ in range(num_samples)],
    "Address": [fake.address().replace("\n", ",") for _ in range(num_samples)]
}

# Convert to DataFrame
test_df = pd.DataFrame(data)

# Add missing values
missing_mask = np.random.rand(num_samples) < 0.1
test_df.loc[missing_mask, "Age"] = np.nan
test_df.loc[missing_mask, "Salary"] = np.nan

# Add outliers in salary
outlier_mask = np.random.rand(num_samples) < 0.05
test_df.loc[outlier_mask, "Salary"] = np.random.normal(200000, 50000, size=outlier_mask.sum())

# Add duplicates
duplicate = test_df.sample(5).copy()
test_df = pd.concat([test_df, duplicate], ignore_index=True)

# Adding categorical col with some typos for testing
test_df.loc[5:9, "Department"] = "Tech!"  # Use .loc to avoid chained assignment
test_df["Department"] = test_df["Department"].replace("Tech!", "Tech")

# Ensure the directory exists
os.makedirs("../data", exist_ok=True)

# Define the absolute path
file_path = "d:/Python/Data-Alchemy/data/testing_data.csv"

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Save the file using the absolute path
try:
    test_df.to_csv(file_path, index=False)
    print("Data saved as testing_data.csv")
except Exception as e:
    print(f"Failed to save data: {e}")

# Print the full path of the saved file
print(f"Data saved at: {file_path}")



