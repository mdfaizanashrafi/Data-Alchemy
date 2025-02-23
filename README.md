**🚀 NumPy + Pandas + SciPy + Matplotlib + MongoDB + MySQL Playground 🌟**

_**Built with ❤️ by MD FAIZAN ASHRAFI**_



<div style="display: flex; gap: 300px; align-items: center;justify-content: center;flex-wrap: wrap;">
    <img src="https://numpy.org/images/logo.svg" alt="NumPy Logo" style="width: 90px; height: auto;" title="NumPy">
    <img src="https://scipy.org/images/logo.svg" alt="SciPy Logo" style="width: 90px; height: auto;" title="Pandas">
    <img src="https://matplotlib.org/stable/_static/logo2.svg" alt="Matplotlib Logo" width="150">
    <img src="https://webimages.mongodb.com/_com_assets/cms/kuyjf3vea2hg34taa-horizontal_default_slate_blue.svg?auto=format%252Ccompress" alt="MongoDB Logo" width="150">
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/MySQL_logo.svg/150px-MySQL_logo.svg.png" alt="MySQL Logo" width="150">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/330px-TensorFlow_logo.svg.png" alt="ML Logo" width="150">
</div>



_Welcome to the **NumPy_Pandas_Playground**! 🎉 This project is your one-stop shop for experimenting with **numerical computations**, **data manipulation**, **scientific computing**, **visualizations**, and **database integrations**. Whether you're a beginner or an experienced data scientist, this modular playground has everything you need to explore and learn._

_**🌟 Features**_
**NumPy Utilities** 🧮: Array operations, linear algebra, and more.
**Pandas Utilities 📊**: Data cleaning, normalization, filtering, and analysis.
**SciPy Utilities 🔬**: Optimization, interpolation, signal processing, and advanced math.
**Matplotlib Integration 📈**: Bar charts, scatter plots, heatmaps, and other visualizations.
**Database Integration 🗄️**:
**MongoDB** : NoSQL database operations.
**MySQL**: Relational database operations.

_**Future Expansions**🚀:_

**Machine Learning models (Scikit-learn, TensorFlow).**
**Web app interface using Flask/FastAPI.**

**📂 Project Structure**
The project is organized into the following folders and files:
```
NumPy_Pandas_Playground/
├── README.md                  📝 Documentation
├── requirements.txt           📋 List of dependencies
├── data/                      📁 Folder for saved data
│   ├── dataset.csv            📄 Example dataset
│   └── random_array.csv       📄 Example saved file
├── utils/                     🛠️ Utility functions
│   ├── numpy_utils.py         🧮 NumPy-specific utilities
│   ├── pandas_utils.py        📊 Pandas-specific utilities
│   ├── scipy_utils.py         🔬 SciPy-specific utilities
│   ├── visualization_utils.py 📈 Matplotlib visualization utilities
│   └── db_operations.py       🗄️ Database operations (MongoDB, MySQL)
├── notebooks/                 📓 Jupyter Notebooks for experimentation
│   ├── explore_numpy.ipynb    🧮 Notebook for NumPy experiments
│   ├── explore_pandas.ipynb   📊 Notebook for Pandas experiments
│   ├── explore_scipy.ipynb    🔬 Notebook for SciPy experiments
│   └── explore_ml.ipynb       🤖 Placeholder for ML experiments
├── scripts/                   🏃 Scripts for running tasks
│   ├── playground.py          🎢 Main script for experimentation
│   └── db_operations.py       🗄️ Script for database operations
└── tests/                     ✅ Unit tests for your utilities
    ├── test_numpy_utils.py    🧮 Tests for NumPy utilities
    ├── test_pandas_utils.py   📊 Tests for Pandas utilities
    └── test_scipy_utils.py    🔬 Tests for SciPy utilities
```

**🚀 Getting Started**
Follow these steps to set up the project locally:

**Clone the Repository 📥:**
git clone https://github.com/mdfaizanashrafi/NumPy_Pandas_Playground.git
cd NumPy_Pandas_Playground

**Install Dependencies 📦:**
pip install -r requirements.txt

**Run the Playground Script ▶️:**
python scripts/playground.py

_**🛠️ Usage Examples**_
Here are some examples of how to use the utilities in this project:

**Load and Clean a Dataset Using Pandas 📊**
from utils.pandas_utils import load_csv, handle_missing_values

df = load_csv("data/dataset.csv")
df_cleaned = handle_missing_values(df, method='fill', fill_value=0)

**Perform Optimization Using SciPy 🔬**
from utils.scipy_utils import optimize_function

result = optimize_function(lambda x: x**2 + 5*x + 6, bounds=[(-10, 10)])
print(result)

**Plot a Bar Chart Using Matplotlib 📈**
from utils.visualization_utils import plot_bar_chart
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
})
plot_bar_chart(df, x_column="name", y_column="age")

**Connect to MongoDB 🗄️**
from scripts.db_operations import connect_mongodb

mongo_db = connect_mongodb("mongodb://localhost:27017/", "test_db")
print("MongoDB Collections:", mongo_db.list_collection_names())

**Connect to MySQL 🗄️**
from scripts.db_operations import connect_mysql

mysql_conn = connect_mysql("localhost", "root", "password", "test_db")
print("MySQL Connection Successful:", mysql_conn.is_connected())

_**🌱 Future Expansions**_
**Machine Learning 🤖:** Implement ML models using libraries like Scikit-learn or TensorFlow.
**Web App 🌐:** Build a web interface using Flask or FastAPI to interact with your utilities.

_**🤝 Contributing**_
Contributions are always welcome! 🙌 Feel free to:

Open issues for bugs or feature requests.
Submit pull requests with improvements.

_**🌟 Acknowledgments**_
Inspired by the open-source community 🌍.



