import pandas as pd
import numpy as np
from datetime import datetime

class DataAnalyzer:
    """Class to handle all data analysis operations"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.date_cols = self._detect_date_columns()
    
    def _detect_date_columns(self):
        """Automatically detect date columns"""
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def get_basic_info(self):
        """Get basic dataset information"""
        info = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols),
            'date_columns': len(self.date_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum()
        }
        return info
    
    def get_summary_statistics(self):
        """Get statistical summary for numeric columns"""
        if not self.numeric_cols:
            return None
        
        stats = self.df[self.numeric_cols].describe()
        return stats
    
    def get_missing_data_info(self):
        """Analyze missing data"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Percentage': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        missing_df = missing_df.sort_values('Missing_Count', ascending=False)
        
        return missing_df
    
    def get_correlation_matrix(self):
        """Calculate correlation matrix for numeric columns"""
        if len(self.numeric_cols) < 2:
            return None
        
        corr_matrix = self.df[self.numeric_cols].corr()
        return corr_matrix
    
    def detect_outliers(self, column):
        """Detect outliers using IQR method"""
        if column not in self.numeric_cols:
            return None
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(self.df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def get_category_distribution(self, column):
        """Get distribution for categorical column"""
        if column not in self.categorical_cols:
            return None
        
        dist = self.df[column].value_counts()
        dist_pct = (dist / len(self.df)) * 100
        
        dist_df = pd.DataFrame({
            'Category': dist.index,
            'Count': dist.values,
            'Percentage': dist_pct.values
        })
        
        return dist_df


class FinancialAnalyzer:
    """Class for financial-specific analysis"""
    
    def __init__(self, df, amount_col=None, category_col=None, date_col=None):
        self.df = df.copy()
        self.amount_col = amount_col or self._detect_amount_column()
        self.category_col = category_col or self._detect_category_column()
        self.date_col = date_col or self._detect_date_column()
        
        if self.date_col:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df = self.df.sort_values(self.date_col)
    
    def _detect_amount_column(self):
        """Auto-detect amount/price column"""
        amount_keywords = ['amount', 'price', 'cost', 'expense', 'revenue', 'sales', 'total']
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    return col
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else None
    
    def _detect_category_column(self):
        """Auto-detect category column"""
        category_keywords = ['category', 'type', 'class', 'group']
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in category_keywords):
                return col
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        return categorical_cols[0] if len(categorical_cols) > 0 else None
    
    def _detect_date_column(self):
        """Auto-detect date column"""
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return col
        return None
    
    def get_total_summary(self):
        """Get total financial summary"""
        if not self.amount_col:
            return None
        
        summary = {
            'total': self.df[self.amount_col].sum(),
            'average': self.df[self.amount_col].mean(),
            'median': self.df[self.amount_col].median(),
            'max': self.df[self.amount_col].max(),
            'min': self.df[self.amount_col].min(),
            'std': self.df[self.amount_col].std()
        }
        
        return summary
    
    def get_category_summary(self):
        """Get summary by category"""
        if not self.amount_col or not self.category_col:
            return None
        
        category_summary = self.df.groupby(self.category_col)[self.amount_col].agg([
            ('Total', 'sum'),
            ('Average', 'mean'),
            ('Count', 'count'),
            ('Max', 'max'),
            ('Min', 'min')
        ]).round(2)
        
        category_summary = category_summary.sort_values('Total', ascending=False)
        category_summary['Percentage'] = (category_summary['Total'] / category_summary['Total'].sum() * 100).round(2)
        
        return category_summary
    
    def get_time_series_data(self, freq='D'):
        """Get time series aggregated data"""
        if not self.date_col or not self.amount_col:
            return None
        
        time_series = self.df.set_index(self.date_col)[self.amount_col].resample(freq).sum()
        return time_series
    
    def get_monthly_summary(self):
        """Get monthly summary"""
        if not self.date_col or not self.amount_col:
            return None
        
        self.df['Month'] = self.df[self.date_col].dt.to_period('M')
        monthly = self.df.groupby('Month')[self.amount_col].agg([
            ('Total', 'sum'),
            ('Average', 'mean'),
            ('Count', 'count')
        ]).round(2)
        
        return monthly
    
    def calculate_moving_average(self, window=7):
        """Calculate moving average"""
        if not self.date_col or not self.amount_col:
            return None
        
        time_series = self.get_time_series_data('D')
        if time_series is None:
            return None
        
        ma = time_series.rolling(window=window).mean()
        return ma
    
    def get_top_transactions(self, n=10):
        """Get top N transactions"""
        if not self.amount_col:
            return None
        
        top = self.df.nlargest(n, self.amount_col)
        return top
    
    def predict_next_period(self):
        """Simple linear prediction for next period"""
        if not self.date_col or not self.amount_col:
            return None
        
        monthly = self.get_monthly_summary()
        if monthly is None or len(monthly) < 2:
            return None
        
        values = monthly['Total'].values
        x = np.arange(len(values))
        
        coefficients = np.polyfit(x, values, 1)
        next_value = np.polyval(coefficients, len(values))
        
        return {
            'predicted_amount': round(next_value, 2),
            'trend': 'Increasing' if coefficients[0] > 0 else 'Decreasing',
            'slope': round(coefficients[0], 2)
        }
    
    def get_insights(self):
        """Generate automatic insights"""
        insights = []
        
        if self.amount_col and self.category_col:
            cat_summary = self.get_category_summary()
            if cat_summary is not None and len(cat_summary) > 0:
                top_category = cat_summary.index[0]
                top_pct = cat_summary.iloc[0]['Percentage']
                insights.append(f"💡 {top_category} accounts for {top_pct:.1f}% of total spending")
        
        if self.date_col and self.amount_col:
            monthly = self.get_monthly_summary()
            if monthly is not None and len(monthly) >= 2:
                recent_avg = monthly.iloc[-1]['Average']
                prev_avg = monthly.iloc[-2]['Average']
                change = ((recent_avg - prev_avg) / prev_avg) * 100
                
                if abs(change) > 10:
                    direction = "increased" if change > 0 else "decreased"
                    insights.append(f"📊 Recent spending has {direction} by {abs(change):.1f}%")
        
        if self.amount_col:
            outliers = self.df[self.amount_col].quantile(0.95)
            high_transactions = len(self.df[self.df[self.amount_col] > outliers])
            if high_transactions > 0:
                insights.append(f"⚠️ {high_transactions} high-value transactions detected")
        
        return insights