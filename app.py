import streamlit as st
import pandas as pd
import numpy as np
from data_analyzer import DataAnalyzer, FinancialAnalyzer
from visualizations import Visualizer, FinancialVisualizer
import io

# Page config
st.set_page_config(
    page_title="Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Advanced Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/analytics.png", width=150)
        st.title("🎛️ Control Panel")
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📁 Upload CSV File",
            type=['csv'],
            help="Upload your data in CSV format"
        )
        
        st.markdown("---")
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
            
            # Analysis mode selection
            st.subheader("🔍 Select Analysis Mode")
            analysis_mode = st.radio(
                "Choose mode:",
                ["💰 Financial Analytics", "📈 Generic Data Analysis"],
                help="Select the type of analysis you want to perform"
            )
            
            st.markdown("---")
            
            # Additional options
            st.subheader("⚙️ Options")
            show_raw_data = st.checkbox("Show Raw Data", value=False)
            show_statistics = st.checkbox("Show Statistics", value=True)
            show_insights = st.checkbox("Show Insights", value=True)
    
    # Main content
    if uploaded_file is None:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info("👆 Please upload a CSV file to get started")
            
            st.markdown("### 🚀 Features")
            st.markdown("""
            - 📊 **15+ Visualization Types** - Interactive and static charts
            - 🤖 **Automatic Analysis** - Smart column detection
            - 💰 **Financial Mode** - Expense tracking and predictions
            - 📈 **Generic Mode** - Works with any dataset
            - 📥 **Export Reports** - Download your analysis
            - 🎯 **Real-time Insights** - Automated recommendations
            """)
            
            st.markdown("### 📝 Sample Data Format")
            
            sample_data = pd.DataFrame({
                'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'Category': ['Food', 'Transport', 'Entertainment'],
                'Amount': [500, 200, 800],
                'Description': ['Groceries', 'Uber', 'Movie']
            })
            
            st.dataframe(sample_data, use_container_width=True)
            
            # Download sample data
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data",
                data=csv,
                file_name="sample_expenses.csv",
                mime="text/csv"
            )
        
        return
    
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        
        if df.empty:
            st.error("❌ The uploaded file is empty!")
            return
        
        # Show raw data if requested
        if show_raw_data:
            st.subheader("📋 Raw Data")
            st.dataframe(df, use_container_width=True)
            st.markdown("---")
        
        # Initialize analyzers
        data_analyzer = DataAnalyzer(df)
        visualizer = Visualizer(df)
        
        # Get basic info
        basic_info = data_analyzer.get_basic_info()
        
        # Display key metrics
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Rows", f"{basic_info['rows']:,}")
        with col2:
            st.metric("Total Columns", basic_info['columns'])
        with col3:
            st.metric("Numeric Columns", basic_info['numeric_columns'])
        with col4:
            st.metric("Categorical Columns", basic_info['categorical_columns'])
        with col5:
            st.metric("Missing Values", basic_info['missing_values'])
        
        st.markdown("---")
        
        # Analysis based on mode
        if analysis_mode == "💰 Financial Analytics":
            financial_analysis(df, data_analyzer, visualizer, show_statistics, show_insights)
        else:
            generic_analysis(df, data_analyzer, visualizer, show_statistics, show_insights)
            
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted.")


def financial_analysis(df, data_analyzer, visualizer, show_statistics, show_insights):
    """Financial analytics mode"""
    
    st.header("💰 Financial Analytics")
    
    # Auto-detect columns
    amount_col = None
    category_col = None
    date_col = None
    
    # Detect amount column
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['amount', 'price', 'cost', 'expense', 'revenue']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    if not amount_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        amount_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    # Detect category column
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['category', 'type', 'class']):
            category_col = col
            break
    
    if not category_col:
        categorical_cols = df.select_dtypes(include=['object']).columns
        category_col = categorical_cols[0] if len(categorical_cols) > 0 else None
    
    # Detect date column
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if not amount_col:
        st.warning("⚠️ Could not detect amount column. Please make sure your CSV has numeric values.")
        return
    
    # Initialize financial analyzer
    fin_analyzer = FinancialAnalyzer(df, amount_col, category_col, date_col)
    fin_visualizer = FinancialVisualizer(df, amount_col, category_col, date_col)
    
    # Financial summary
    summary = fin_analyzer.get_total_summary()
    
    if summary:
        st.subheader("💵 Financial Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Amount", f"₹{summary['total']:,.2f}")
        with col2:
            st.metric("Average", f"₹{summary['average']:,.2f}")
        with col3:
            st.metric("Maximum", f"₹{summary['max']:,.2f}")
        with col4:
            st.metric("Minimum", f"₹{summary['min']:,.2f}")
    
    st.markdown("---")
    
    # Insights
    if show_insights:
        insights = fin_analyzer.get_insights()
        if insights:
            st.subheader("🎯 Key Insights")
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Categories", "📅 Time Series", "🔍 Advanced"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if category_col:
                st.markdown("#### Category Distribution")
                fig = fin_visualizer.plot_category_breakdown()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if category_col:
                st.markdown("#### Category Pie Chart")
                fig = visualizer.plot_pie_chart(amount_col, category_col, "Spending Distribution")
                if fig:
                    st.pyplot(fig)
    
    with tab2:
        if category_col:
            category_summary = fin_analyzer.get_category_summary()
            if category_summary is not None:
                st.markdown("#### Category-wise Analysis")
                st.dataframe(category_summary, use_container_width=True)
                
                st.markdown("#### Box Plot by Category")
                fig = visualizer.plot_box_plot(amount_col, category_col, "Amount Distribution by Category")
                if fig:
                    st.pyplot(fig)
    
    with tab3:
        if date_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Spending Over Time")
                fig = fin_visualizer.plot_spending_over_time()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Monthly Trend")
                fig = fin_visualizer.plot_monthly_trend()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            monthly_summary = fin_analyzer.get_monthly_summary()
            if monthly_summary is not None:
                st.markdown("#### Monthly Summary")
                st.dataframe(monthly_summary, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if date_col and category_col:
                st.markdown("#### Spending Heatmap")
                fig = fin_visualizer.plot_spending_heatmap()
                if fig:
                    st.pyplot(fig)
        
        with col2:
            st.markdown("#### Amount Distribution")
            fig = visualizer.plot_histogram(amount_col, bins=30, title="Amount Distribution")
            if fig:
                st.pyplot(fig)
        
        # Predictions
        prediction = fin_analyzer.predict_next_period()
        if prediction:
            st.markdown("#### 📈 Next Period Prediction")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Amount", f"₹{prediction['predicted_amount']:,.2f}")
            with col2:
                st.metric("Trend", prediction['trend'])
            with col3:
                st.metric("Slope", f"{prediction['slope']:,.2f}")
    
    # Statistics
    if show_statistics:
        st.markdown("---")
        st.subheader("📊 Statistical Analysis")
        
        stats = data_analyzer.get_summary_statistics()
        if stats is not None:
            st.dataframe(stats, use_container_width=True)
    
    # Export option
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    if st.button("Generate Excel Report"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            if category_col:
                cat_summary = fin_analyzer.get_category_summary()
                if cat_summary is not None:
                    cat_summary.to_excel(writer, sheet_name='Category Summary')
            if date_col:
                monthly = fin_analyzer.get_monthly_summary()
                if monthly is not None:
                    monthly.to_excel(writer, sheet_name='Monthly Summary')
        
        output.seek(0)
        st.download_button(
            label="📥 Download Excel Report",
            data=output,
            file_name="financial_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


def generic_analysis(df, data_analyzer, visualizer, show_statistics, show_insights):
    """Generic data analysis mode"""
    
    st.header("📈 Generic Data Analysis")
    
    # Column selection
    st.subheader("🎯 Select Columns for Analysis")
    
    numeric_cols = data_analyzer.numeric_cols
    categorical_cols = data_analyzer.categorical_cols
    
    if not numeric_cols and not categorical_cols:
        st.warning("⚠️ No suitable columns found for analysis.")
        return
    
    # Visualizations
    st.markdown("---")
    st.subheader("📊 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Relationships", "📈 Comparisons", "🎨 Advanced"])
    
    with tab1:
        if numeric_cols:
            col1, col2 = st.columns(2)
            
            for idx, col in enumerate(numeric_cols[:4]):
                with col1 if idx % 2 == 0 else col2:
                    st.markdown(f"#### {col} Distribution")
                    fig = visualizer.plot_histogram(col, bins=30, title=f"{col} Distribution")
                    if fig:
                        st.pyplot(fig)
        
        if categorical_cols:
            st.markdown("#### Categorical Distributions")
            for col in categorical_cols[:3]:
                fig = visualizer.plot_count_plot(col, f"{col} Count")
                if fig:
                    st.pyplot(fig)
    
    with tab2:
        if len(numeric_cols) >= 2:
            st.markdown("#### Correlation Heatmap")
            corr_matrix = data_analyzer.get_correlation_matrix()
            if corr_matrix is not None:
                fig = visualizer.plot_heatmap(corr_matrix, "Correlation Matrix")
                if fig:
                    st.pyplot(fig)
            
            if len(numeric_cols) >= 3:
                st.markdown("#### Pair Plot")
                cols_for_pair = numeric_cols[:4]
                fig = visualizer.plot_pair_plot(cols_for_pair)
                if fig:
                    st.pyplot(fig)
    
    with tab3:
        if numeric_cols and categorical_cols:
            st.markdown("#### Box Plots by Category")
            
            num_col = st.selectbox("Select Numeric Column", numeric_cols, key="box_num")
            cat_col = st.selectbox("Select Category Column", categorical_cols, key="box_cat")
            
            if num_col and cat_col:
                fig = visualizer.plot_box_plot(num_col, cat_col, f"{num_col} by {cat_col}")
                if fig:
                    st.pyplot(fig)
    
    with tab4:
        if len(numeric_cols) >= 2:
            st.markdown("#### Interactive Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", df.columns, key="scatter_x")
                y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                
                if x_col and y_col:
                    fig = visualizer.plot_interactive_line(x_col, y_col, f"{y_col} vs {x_col}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(numeric_cols) >= 3:
                    st.markdown("#### 3D Scatter Plot")
                    x_3d = st.selectbox("X", numeric_cols, index=0, key="3d_x")
                    y_3d = st.selectbox("Y", numeric_cols, index=1, key="3d_y")
                    z_3d = st.selectbox("Z", numeric_cols, index=2, key="3d_z")
                    
                    color_col = st.selectbox("Color by", [None] + categorical_cols, key="3d_color")
                    
                    if x_3d and y_3d and z_3d:
                        fig = visualizer.plot_scatter_3d(x_3d, y_3d, z_3d, color_col, "3D Scatter")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    if show_statistics:
        st.markdown("---")
        st.subheader("📊 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Descriptive Statistics")
            stats = data_analyzer.get_summary_statistics()
            if stats is not None:
                st.dataframe(stats, use_container_width=True)
        
        with col2:
            st.markdown("#### Missing Data Analysis")
            missing = data_analyzer.get_missing_data_info()
            if missing is not None and not missing.empty:
                st.dataframe(missing, use_container_width=True)
            else:
                st.success("✅ No missing values detected!")
    
    # Outlier detection
    if numeric_cols:
        st.markdown("---")
        st.subheader("🔍 Outlier Detection")
        
        outlier_col = st.selectbox("Select column for outlier analysis", numeric_cols)
        
        if outlier_col:
            outlier_info = data_analyzer.detect_outliers(outlier_col)
            if outlier_info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Outliers Found", outlier_info['count'])
                with col2:
                    st.metric("Percentage", f"{outlier_info['percentage']:.2f}%")
                with col3:
                    st.metric("Range", f"{outlier_info['lower_bound']:.2f} - {outlier_info['upper_bound']:.2f}")
    
    # Export
    st.markdown("---")
    if st.button("Download Analysis Report"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="analysis_report.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()