

# 💰 Personal Finance Analytics Dashboard with Predictive Modeling

A comprehensive, end-to-end data analytics application built with **Streamlit**. This dashboard empowers users to track, analyze, and forecast their financial health by integrating multiple data sources and using machine learning for expense prediction.

## 🚀 Key Features

* **Multi-Source Integration:** Seamlessly processes and merges data from separate CSV sources (Income, Expenses, and Investments).
* **Advanced Data Cleaning:** Built with `Pandas` for robust data manipulation, handling missing values, and categorical grouping.
* **Predictive Forecasting:** Implements **Linear Regression** via `NumPy` to analyze historical trends and forecast future monthly expenses.
* **15+ Dynamic Visualizations:** Leverages `Matplotlib` and `Seaborn` to create interactive charts, including:
* Monthly Cash Flow Trends.
* Category-wise Spending Breakdowns (Donut/Bar charts).
* Investment Growth Trajectories.
* Correlation Heatmaps for spending behavior.


* **Automated Financial Insights:** Smart logic that generates actionable insights (e.g., "Your spending in the 'Dining' category has increased by 40% compared to last month").
* **Interactive Sidebar:** Custom filters for Month-wise, Year-wise, and Category-wise analysis.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualizations:** Matplotlib, Seaborn, Plotly
* **Statistical Modeling:** NumPy 

## 📁 Directory Structure

* `app.py`: Main entry point for the Streamlit dashboard.
* `data_analyzer.py`: Backend logic for data cleaning and statistical calculations.
* `visualizations.py`: Custom module containing all plotting functions.
* `sample_data/`: Folder containing raw CSV datasets.
* `requirements.txt`: List of Python dependencies.

## ⚙️ Setup and Installation

1. **Clone the repository:**
   
    git clone https://github.com/your_username/finance-analytics-dashboard.git



3. **Activate your virtual environment:**

    python -m venv venv

    .\venv\Scripts\activate



4. **Install dependencies:**

    pip install -r requirements.txt



4. **Run the application:**

     streamlit run app.py



