import os
import pandas as pd
from faker import Faker
import numpy as np

#Initialize Faker to generate fake data
fake=Faker()

#Define the numebr of rows to generate
num_rows=1000

#Generate random data using Faker
data={"Name":[fake.name() if np.random.rand()>0.1 else np.nan for _ in range(num_rows)],
      "Address":[fake.address().replace("\n",",") if np.random.rand()>0.15 else np.nan for _ in range(num_rows)],
      "Email":[fake.email() if np.random.rand()>0.18 else np.nan for _ in range(num_rows)],
      "Phone Number":[fake.phone_number() if np.random.rand()>0.2 else np.nan for _ in range(num_rows)],
      "Job Title": [fake.job() if np.random.rand()>0.12 else np.nan for _ in range(num_rows)],
      "City": [fake.city() if np.random.rand()>0.2 else np.nan for _ in range(num_rows)],
      "State": [fake.state() if np.random.rand()>0.1 else np.nan for _ in range(num_rows)],
      "Zip Code": [fake.postcode() if np.random.rand()>0.1 else np.nan for _ in range(num_rows)],
      "Country": [fake.country() if np.random.rand()>0.1 else np.nan for _ in range(num_rows)],
      "Date of Birth": [fake.date_of_birth() if np.random.rand()>0.1 else np.nan for _ in range(num_rows)]}

#Create a pandas DataFrame from the dictionary
df= pd.DataFrame(data)

#additinal complexities, some rows have all fields missing
# 5 % of rows are entirely NaN
df.loc[np.random.choice(df.index,size=int(num_rows*0.05), replace=False)]=np.nan

project_root=os.getcwd()
data_directory=os.path.join(project_root,"data")
os.makedirs(data_directory,exist_ok=True)
file_path=os.path.join(data_directory,"random_data_for_test.csv")

#save the data to a CSV file
df.to_csv(file_path,index=False)

#save the data to a json file
directory=os.path.join(data_directory,"random_data_for_test.json")
df.to_json(directory,orient="records")


print(f"Random Data Generated and saved to {directory}")



