import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Set style for matplotlib/seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class Visualizer:
    """Class to handle all visualizations"""
    
    def __init__(self, df):
        self.df = df
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # ========== MATPLOTLIB CHARTS ==========
    
    def plot_line_chart(self, x_col, y_col, title="Line Chart"):
        """Create line chart with matplotlib"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.df[x_col], self.df[y_col], marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_bar_chart(self, x_col, y_col, title="Bar Chart", horizontal=False):
        """Create bar chart with matplotlib"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if horizontal:
            ax.barh(self.df[x_col], self.df[y_col], color=self.colors[0])
            ax.set_xlabel(y_col, fontsize=12)
            ax.set_ylabel(x_col, fontsize=12)
        else:
            ax.bar(self.df[x_col], self.df[y_col], color=self.colors[0])
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            plt.xticks(rotation=45)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    def plot_pie_chart(self, values_col, labels_col, title="Distribution"):
        """Create pie chart with matplotlib"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        data = self.df.groupby(labels_col)[values_col].sum()
        
        ax.pie(data.values, labels=data.index, autopct='%1.1f%%',
               startangle=90, colors=self.colors)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    # ========== SEABORN CHARTS ==========
    
    def plot_heatmap(self, data, title="Correlation Heatmap"):
        """Create heatmap with seaborn"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_box_plot(self, column, by_category=None, title="Box Plot"):
        """Create box plot with seaborn"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if by_category:
            sns.boxplot(data=self.df, x=by_category, y=column, palette='Set2', ax=ax)
            plt.xticks(rotation=45)
        else:
            sns.boxplot(data=self.df, y=column, palette='Set2', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    def plot_violin_plot(self, column, by_category=None, title="Distribution"):
        """Create violin plot with seaborn"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if by_category:
            sns.violinplot(data=self.df, x=by_category, y=column, palette='muted', ax=ax)
            plt.xticks(rotation=45)
        else:
            sns.violinplot(data=self.df, y=column, palette='muted', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_histogram(self, column, bins=30, title="Distribution"):
        """Create histogram with KDE using seaborn"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=self.df, x=column, bins=bins, kde=True, 
                     color=self.colors[0], ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        return fig
    
    def plot_count_plot(self, column, title="Count Plot"):
        """Create count plot with seaborn"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=self.df, x=column, palette='Set2', ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_pair_plot(self, columns, hue=None):
        """Create pair plot with seaborn"""
        if len(columns) > 5:
            columns = columns[:5]
        
        fig = sns.pairplot(self.df[columns], hue=hue, palette='husl', 
                          diag_kind='kde', plot_kws={'alpha': 0.6})
        return fig.fig
    
    # ========== PLOTLY INTERACTIVE CHARTS ==========
    
    def plot_interactive_line(self, x_col, y_col, title="Interactive Line Chart"):
        """Create interactive line chart with plotly"""
        fig = px.line(self.df, x=x_col, y=y_col, title=title,
                     markers=True, line_shape='spline')
        
        fig.update_layout(
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_interactive_bar(self, x_col, y_col, title="Interactive Bar Chart", 
                            color_col=None):
        """Create interactive bar chart with plotly"""
        fig = px.bar(self.df, x=x_col, y=y_col, title=title,
                    color=color_col, text=y_col)
        
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(template='plotly_white', height=500)
        
        return fig
    
    def plot_interactive_pie(self, values_col, names_col, title="Interactive Pie Chart"):
        """Create interactive pie chart with plotly"""
        data = self.df.groupby(names_col)[values_col].sum().reset_index()
        
        fig = px.pie(data, values=values_col, names=names_col, title=title,
                    hole=0.3)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        return fig
    
    def plot_scatter_3d(self, x_col, y_col, z_col, color_col=None, 
                       title="3D Scatter Plot"):
        """Create 3D scatter plot with plotly"""
        fig = px.scatter_3d(self.df, x=x_col, y=y_col, z=z_col,
                           color=color_col, title=title)
        
        fig.update_layout(height=600)
        
        return fig
    
    def plot_area_chart(self, x_col, y_col, title="Area Chart"):
        """Create area chart with plotly"""
        fig = px.area(self.df, x=x_col, y=y_col, title=title)
        
        fig.update_layout(
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_sunburst(self, path_cols, values_col, title="Sunburst Chart"):
        """Create sunburst chart with plotly"""
        fig = px.sunburst(self.df, path=path_cols, values=values_col,
                         title=title)
        
        fig.update_layout(height=600)
        
        return fig
    
    def plot_treemap(self, path_cols, values_col, title="Treemap"):
        """Create treemap with plotly"""
        fig = px.treemap(self.df, path=path_cols, values=values_col,
                        title=title)
        
        fig.update_layout(height=600)
        
        return fig
    
    def plot_waterfall(self, categories, values, title="Waterfall Chart"):
        """Create waterfall chart with plotly"""
        fig = go.Figure(go.Waterfall(
            name="Amount",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            text=values,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=500
        )
        
        return fig
    
    def plot_gauge(self, value, max_value, title="Gauge Chart"):
        """Create gauge chart with plotly"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': max_value * 0.8},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': "lightgray"},
                    {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=400)
        
        return fig
    
    def plot_candlestick(self, date_col, open_col, high_col, low_col, close_col,
                        title="Candlestick Chart"):
        """Create candlestick chart with plotly"""
        fig = go.Figure(data=[go.Candlestick(
            x=self.df[date_col],
            open=self.df[open_col],
            high=self.df[high_col],
            low=self.df[low_col],
            close=self.df[close_col]
        )])
        
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        return fig
    
    def plot_animated_bar_race(self, date_col, category_col, value_col,
                               title="Animated Bar Chart"):
        """Create animated bar chart race with plotly"""
        fig = px.bar(self.df, x=value_col, y=category_col,
                    animation_frame=date_col,
                    orientation='h',
                    title=title,
                    range_x=[0, self.df[value_col].max() * 1.1])
        
        fig.update_layout(height=600)
        
        return fig


class FinancialVisualizer:
    """Specialized visualizations for financial data"""
    
    def __init__(self, df, amount_col, category_col=None, date_col=None):
        self.df = df
        self.amount_col = amount_col
        self.category_col = category_col
        self.date_col = date_col
    
    def plot_spending_over_time(self):
        """Plot spending trend over time"""
        if not self.date_col or not self.amount_col:
            return None
        
        daily_spending = self.df.groupby(self.date_col)[self.amount_col].sum().reset_index()
        
        fig = px.line(daily_spending, x=self.date_col, y=self.amount_col,
                     title="Spending Over Time",
                     markers=True)
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Amount",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_category_breakdown(self):
        """Plot category-wise breakdown"""
        if not self.category_col or not self.amount_col:
            return None
        
        category_total = self.df.groupby(self.category_col)[self.amount_col].sum().reset_index()
        category_total = category_total.sort_values(self.amount_col, ascending=True)
        
        fig = px.bar(category_total, y=self.category_col, x=self.amount_col,
                    title="Spending by Category",
                    orientation='h',
                    text=self.amount_col)
        
        fig.update_traces(texttemplate='₹%{text:.2s}', textposition='outside')
        fig.update_layout(template='plotly_white', height=500)
        
        return fig
    
    def plot_monthly_trend(self):
        """Plot monthly spending trend"""
        if not self.date_col or not self.amount_col:
            return None
        
        self.df['Month'] = pd.to_datetime(self.df[self.date_col]).dt.to_period('M').astype(str)
        monthly = self.df.groupby('Month')[self.amount_col].sum().reset_index()
        
        fig = px.bar(monthly, x='Month', y=self.amount_col,
                    title="Monthly Spending Trend",
                    text=self.amount_col)
        
        fig.update_traces(texttemplate='₹%{text:.2s}', textposition='outside')
        fig.update_layout(template='plotly_white', height=500)
        
        return fig
    
    def plot_spending_heatmap(self):
        """Create day-wise spending heatmap"""
        if not self.date_col or not self.amount_col or not self.category_col:
            return None
        
        pivot_data = self.df.pivot_table(
            values=self.amount_col,
            index=pd.to_datetime(self.df[self.date_col]).dt.day_name(),
            columns=self.category_col,
            aggfunc='sum',
            fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
        ax.set_title("Spending Heatmap: Day vs Category", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig