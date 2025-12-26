

# 💰 Advanced Personal Finance Analytics Dashboard

An end-to-end data analytics application built with **Streamlit** that helps users track, analyze, and predict their financial health. This dashboard integrates multiple data sources to provide deep insights into spending habits, investment growth, and future expense forecasting.

## 🚀 Key Features

* **Multi-Source Data Integration:** Seamlessly merges data from Expenses, Income, and Investment CSV files.
* **Deep Data Cleaning:** Robust preprocessing using `pandas` to handle missing values, date formatting, and categorical grouping.
* **Predictive Modeling:** Implemented **Linear Regression** (using `numpy` & `scipy`) to forecast future monthly expenses based on historical trends.
* **Statistical Deep-Dive:** Analysis of spending patterns, including variance analysis and category-wise breakdowns.
* **Interactive Visualizations:** Over 12+ dynamic charts using `matplotlib` and `seaborn`, including:
* Monthly Cash Flow (Income vs. Expense)
* Investment Growth Trajectory
* Category-wise Spending Donut Charts
* Expense Correlation Heatmaps


* **Smart Insights:** Automated text-based insights (e.g., *"Warning: Your dining expenses increased by 40% this month!"*).
* **Dynamic Filtering:** Filter data by date range, transaction category, or payment mode via the sidebar.

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **Frontend:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Mathematics:** Scipy (for Regression Analysis)

## 📦 Project Structure

```text
├── app.py                 # Main Streamlit application
├── data_analyzer.py       # Data cleaning & statistical logic
├── visualizations.py      # Custom plotting functions
├── sample_data/           # Folder containing CSVs (Income, Expenses, etc.)
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation

```

## ⚙️ Installation & Usage

1. **Clone the Repo:**

git clone https://github.com/alok66868/finance-analytics-dashboard.git



2. **Setup Virtual Environment:**

python -m venv venv
.\venv\Scripts\activate



3. **Install Dependencies:**

pip install -r requirements.txt



4. **Run the Dashboard:**

streamlit run app.py


