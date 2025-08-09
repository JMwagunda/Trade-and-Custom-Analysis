import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Trade & Customs Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e8f4fd, #ffffff, #e8f4fd);
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #ecf0f1;
        border-left: 5px solid #3498db;
        border-radius: 5px;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .prediction-result {
        font-size: 1.2rem;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        padding: 1rem;
        background-color: #d5f4e6;
        border-radius: 10px;
        border: 2px solid #27ae60;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df_enhanced' not in st.session_state:
    st.session_state.df_enhanced = None
if 'models' not in st.session_state:
    st.session_state.models = {}

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset"""
    try:
        # You can modify this to load from uploaded file or default dataset
        df = pd.read_csv('cleaned_customs_data.csv', parse_dates=['Receipt_Date'])
        
        # Enhanced Feature Engineering
        df['Receipt_Year'] = df['Receipt_Date'].dt.year
        df['Receipt_Month'] = df['Receipt_Date'].dt.month
        df['Receipt_Quarter'] = df['Receipt_Date'].dt.quarter
        df['Days_Since_Start'] = (df['Receipt_Date'] - df['Receipt_Date'].min()).dt.days
        
        df['CIF_FOB_Ratio'] = df['CIF_Value_N'] / (df['FOB_Value_N'] + 1)
        df['Tax_Efficiency'] = df['Total_Tax_N'] / (df['CIF_Value_N'] + 1)
        df['Value_per_KG'] = df['CIF_Value_N'] / (df['Mass_KG'] + 1)
        
        # Categorical encoding
        le_country = LabelEncoder()
        le_office = LabelEncoder()
        le_hs = LabelEncoder()
        
        df['Country_Encoded'] = le_country.fit_transform(df['Country_of_Origin'])
        df['Office_Encoded'] = le_office.fit_transform(df['Custom_Office'])
        df['HS_Category_Encoded'] = le_hs.fit_transform(df['HS_Code_Category'])
        
        # Risk features
        df['High_Value_Threshold'] = (df['CIF_Value_N'] > df['CIF_Value_N'].quantile(0.95)).astype(int)
        df['Tax_Anomaly'] = ((df['Tax_to_Value_Ratio'] < df['Tax_to_Value_Ratio'].quantile(0.05)) | 
                            (df['Tax_to_Value_Ratio'] > df['Tax_to_Value_Ratio'].quantile(0.95))).astype(int)
        
        encoders = {'country': le_country, 'office': le_office, 'hs_category': le_hs}
        
        return df, encoders
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_data
def calculate_kpis(df):
    """Calculate comprehensive KPIs"""
    kpis = {}
    
    # Import Volume and Value KPIs
    kpis['import_volume_value'] = {
        'total_fob': df['FOB_Value_N'].sum(),
        'total_cif': df['CIF_Value_N'].sum(),
        'avg_fob_per_transaction': df['FOB_Value_N'].mean(),
        'avg_cif_per_transaction': df['CIF_Value_N'].mean(),
        'top_7_countries_by_value': df.groupby('Country_of_Origin')['CIF_Value_N'].sum().nlargest(7),
        'avg_mass_per_transaction': df['Mass_KG'].mean()
    }
    
    # Taxation and Revenue KPIs
    kpis['taxation_revenue'] = {
        'total_tax_collected': df['Total_Tax_N'].sum(),
        'avg_tax_per_transaction': df['Total_Tax_N'].mean(),
        'avg_tax_to_value_ratio': df['Tax_to_Value_Ratio'].mean(),
        'top_5_tax_contributing_importers': df.groupby('Importer')['Total_Tax_N'].sum().nlargest(5)
    }
    
    # Logistics and Shipment KPIs
    kpis['logistics_shipment'] = {
        'total_shipments': len(df),
        'avg_containers_per_importer': df.groupby('Importer')['Nbr_Of_Containers'].mean().mean()
    }
    
    # Compliance and Processing KPIs
    kpis['compliance_processing'] = {
        'transactions_per_office': df.groupby('Custom_Office')['ID'].count().sort_values(ascending=False),
        'top_5_hs_codes_by_frequency': df.groupby('HS_Code_Category')['ID'].count().nlargest(5),
        'top_5_hs_codes_by_value': df.groupby('HS_Code_Category')['CIF_Value_N'].sum().nlargest(5),
        'pct_high_value_imports': (df['Is_High_Value'].sum() / len(df)) * 100,
        'pct_suspicious_tax': (df['Is_Suspicious_Tax'].sum() / len(df)) * 100
    }
    
    return kpis

def train_models(df, encoders):
    """Train prediction models"""
    feature_columns = [
        'FOB_Value_N', 'CIF_Value_N', 'Mass_KG', 'FOB_CIF_Diff', 
        'Tax_to_Value_Ratio', 'CIF_FOB_Ratio', 'Value_per_KG',
        'Country_Encoded', 'Office_Encoded', 'HS_Category_Encoded',
        'Receipt_Year', 'Receipt_Month', 'Receipt_Quarter', 'Has_Containers',
        'High_Value_Threshold'
    ]
    
    # Prepare data
    model_df = df[feature_columns + ['Total_Tax_N']].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = model_df[feature_columns]
    y = model_df['Total_Tax_N']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    model_results = {}
    
    for name, model in models.items():
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    
    # Select best model
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
    
    return model_results, best_model_name, scaler, feature_columns, encoders

def predict_tax(transaction_data, models, best_model_name, scaler, feature_columns, encoders):
    """Predict tax for a transaction"""
    required_features = feature_columns
    
    # Create DataFrame
    trans_df = pd.DataFrame([transaction_data])
    
    # Add encoded features
    for col in required_features:
        if col not in trans_df.columns:
            if 'Encoded' in col:
                base_col = col.replace('_Encoded', '')
                if base_col == 'Country':
                    base_col = 'Country_of_Origin'
                elif base_col == 'Office':
                    base_col = 'Custom_Office'
                elif base_col == 'HS_Category':
                    base_col = 'HS_Code_Category'
                
                if base_col in trans_df.columns:
                    encoder_key = col.split('_')[0].lower()
                    if encoder_key in encoders:
                        try:
                            trans_df[col] = encoders[encoder_key].transform(trans_df[base_col])
                        except ValueError:
                            trans_df[col] = 0
                    else:
                        trans_df[col] = 0
                else:
                    trans_df[col] = 0
            else:
                trans_df[col] = 0
    
    # Select features
    X_new = trans_df[required_features]
    
    # Make prediction
    best_model = models[best_model_name]['model']
    if best_model_name == 'Linear Regression':
        X_new_scaled = scaler.transform(X_new)
        prediction = best_model.predict(X_new_scaled)
    else:
        prediction = best_model.predict(X_new)
    
    return prediction[0]

def simulate_policy(df, models, best_model_name, scaler, feature_columns, scenarios):
    """Simulate policy scenarios"""
    results = {}
    base_tax = df['Total_Tax_N'].sum()
    best_model = models[best_model_name]['model']
    
    for scenario_name, changes in scenarios.items():
        sim_df = df.copy()
        
        # Apply changes
        for column, multiplier in changes.items():
            if column in sim_df.columns:
                sim_df[column] = sim_df[column] * multiplier
        
        # Recalculate derived features
        sim_df['CIF_FOB_Ratio'] = sim_df['CIF_Value_N'] / (sim_df['FOB_Value_N'] + 1)
        sim_df['Tax_to_Value_Ratio'] = sim_df['Total_Tax_N'] / (sim_df['CIF_Value_N'] + 1)
        sim_df['Value_per_KG'] = sim_df['CIF_Value_N'] / (sim_df['Mass_KG'] + 1)
        
        # Make predictions
        X_sim = sim_df[feature_columns]
        if best_model_name == 'Linear Regression':
            X_sim_scaled = scaler.transform(X_sim)
            predicted_taxes = best_model.predict(X_sim_scaled)
        else:
            predicted_taxes = best_model.predict(X_sim)
        
        predicted_total = predicted_taxes.sum()
        change = predicted_total - base_tax
        change_pct = (change / base_tax) * 100
        
        results[scenario_name] = {
            'predicted_total': predicted_total,
            'change': change,
            'change_pct': change_pct
        }
    
    return results

# Main App Layout
def main():
    # Header
    st.markdown('<div class="main-header">🏛️ Trade & Customs Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Dashboard Overview", "💰 Tax Prediction", "🏛️ Policy Simulation", "📈 Data Visualization", "⚙️ Model Performance"]
    )
    
    # Data loading section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Management")
    
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
        with st.spinner("Loading and processing data..."):
            df, encoders = load_and_prepare_data()
            if df is not None:
                st.session_state.df_enhanced = df
                st.session_state.encoders = encoders
                st.session_state.kpis = calculate_kpis(df)
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load data")
    
    if st.sidebar.button("🤖 Train Models") and st.session_state.data_loaded:
        with st.spinner("Training machine learning models..."):
            models, best_model_name, scaler, feature_columns, encoders = train_models(
                st.session_state.df_enhanced, st.session_state.encoders
            )
            st.session_state.models = models
            st.session_state.best_model_name = best_model_name
            st.session_state.scaler = scaler
            st.session_state.feature_columns = feature_columns
            st.session_state.model_trained = True
            st.sidebar.success(f"✅ Models trained! Best: {best_model_name}")
    
    # Display current status
    if st.session_state.data_loaded:
        st.sidebar.markdown("**📊 Data Status:** ✅ Loaded")
        st.sidebar.markdown(f"**📈 Transactions:** {len(st.session_state.df_enhanced):,}")
    else:
        st.sidebar.markdown("**📊 Data Status:** ❌ Not loaded")
    
    if st.session_state.model_trained:
        st.sidebar.markdown(f"**🤖 Model Status:** ✅ {st.session_state.best_model_name}")
        st.sidebar.markdown(f"**🎯 Accuracy:** {st.session_state.models[st.session_state.best_model_name]['r2']:.3f}")
    else:
        st.sidebar.markdown("**🤖 Model Status:** ❌ Not trained")
    
    # Page content based on selection
    if page == "📊 Dashboard Overview":
        show_dashboard_overview()
    elif page == "💰 Tax Prediction":
        show_tax_prediction()
    elif page == "🏛️ Policy Simulation":
        show_policy_simulation()
    elif page == "📈 Data Visualization":
        show_data_visualization()
    elif page == "⚙️ Model Performance":
        show_model_performance()

def show_dashboard_overview():
    """Display dashboard overview"""
    st.markdown('<div class="section-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div class="warning-box">⚠️ Please load data first using the sidebar.</div>', unsafe_allow_html=True)
        return
    
    kpis = st.session_state.kpis
    df = st.session_state.df_enhanced
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total CIF Value",
            value=f"{kpis['import_volume_value']['total_cif']/1e12:.2f}T ₦",
            delta="Trillions"
        )
    
    with col2:
        st.metric(
            label="📊 Tax Collected",
            value=f"{kpis['taxation_revenue']['total_tax_collected']/1e9:.1f}B ₦",
            delta="Billions"
        )
    
    with col3:
        st.metric(
            label="📦 Total Transactions",
            value=f"{kpis['logistics_shipment']['total_shipments']:,}",
            delta="Count"
        )
    
    with col4:
        st.metric(
            label="📈 Avg Tax Ratio",
            value=f"{kpis['taxation_revenue']['avg_tax_to_value_ratio']:.3f}",
            delta="Percentage"
        )
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Countries Chart
        top_countries = kpis['import_volume_value']['top_7_countries_by_value']
        fig = px.bar(
            x=top_countries.values/1e9,
            y=top_countries.index,
            orientation='h',
            title="Top 7 Countries by CIF Value (Billions ₦)",
            labels={'x': 'CIF Value (Billions ₦)', 'y': 'Country'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tax Distribution Chart
        fig = px.histogram(
            df, 
            x='Tax_to_Value_Ratio',
            nbins=30,
            title="Distribution of Tax-to-Value Ratios",
            labels={'Tax_to_Value_Ratio': 'Tax-to-Value Ratio', 'count': 'Frequency'}
        )
        fig.add_vline(x=df['Tax_to_Value_Ratio'].mean(), line_dash="dash", line_color="red")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Trend Chart
    monthly_data = df.groupby(df['Receipt_Date'].dt.to_period('M')).agg({
        'CIF_Value_N': 'sum',
        'Total_Tax_N': 'sum',
        'ID': 'count'
    }).reset_index()
    monthly_data['Receipt_Date'] = monthly_data['Receipt_Date'].astype(str)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Receipt_Date'], 
            y=monthly_data['CIF_Value_N']/1e9,
            name="CIF Value (Billions ₦)",
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Receipt_Date'], 
            y=monthly_data['Total_Tax_N']/1e9,
            name="Tax Collected (Billions ₦)",
            line=dict(color='red')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=monthly_data['Receipt_Date'], 
            y=monthly_data['ID'],
            name="Transaction Count",
            opacity=0.3
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Value (Billions ₦)", secondary_y=False)
    fig.update_yaxes(title_text="Transaction Count", secondary_y=True)
    fig.update_layout(title="Monthly Trends: Values vs Transaction Volume", height=500)
    
    st.plotly_chart(fig, use_container_width=True)

def show_tax_prediction():
    """Display tax prediction interface"""
    st.markdown('<div class="section-header">💰 Tax Revenue Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.markdown('<div class="warning-box">⚠️ Please train models first using the sidebar.</div>', unsafe_allow_html=True)
        return
    
    st.markdown("### Enter Transaction Details")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Financial Information")
        fob_value = st.number_input("FOB Value (₦)", min_value=0.0, value=1046575.00, step=1000000.0, help="Free On Board value in Naira")
        cif_value = st.number_input("CIF Value (₦)", min_value=0.0, value=1119220.0, step=1000000.0, help="Cost, Insurance, and Freight value in Naira")
        mass_kg = st.number_input("Mass (KG)", min_value=0.0, value=126.0, step=1000.0, help="Total mass in kilograms")
        
        st.subheader("📅 Temporal Information")
        receipt_year = st.selectbox("Receipt Year", options=[2022, 2023, 2024], index=1)
        receipt_month = st.selectbox("Receipt Month", options=list(range(1, 13)), index=6)
        receipt_quarter = st.selectbox("Receipt Quarter", options=[1, 2, 3, 4], index=2)
    
    with col2:
        st.subheader("🌍 Geographic & Classification")
        
        # Get unique values from data for dropdowns
        df = st.session_state.df_enhanced
        countries = sorted(df['Country_of_Origin'].unique())
        offices = sorted(df['Custom_Office'].unique())
        hs_categories = sorted(df['HS_Code_Category'].unique())
        
        country_of_origin = st.selectbox("Country of Origin", options=countries, index=countries.index('China') if 'China' in countries else 0)
        custom_office = st.selectbox("Custom Office", options=offices, index=0)
        hs_code_category = st.selectbox("HS Code Category", options=hs_categories, index=0)
        
        st.subheader("📦 Logistics Information")
        has_containers = st.selectbox("Has Containers", options=[0, 1], index=1, help="1 = Yes, 0 = No")
        high_value_threshold = st.selectbox("High Value Import", options=[0, 1], index=1 if cif_value > 50000000 else 0, help="Automatically determined based on CIF value")
    
    # Calculate derived features
    fob_cif_diff = cif_value - fob_value
    tax_to_value_ratio = 0.12  # Default assumption
    cif_fob_ratio = cif_value / fob_value if fob_value > 0 else 1.1
    value_per_kg = cif_value / mass_kg if mass_kg > 0 else 1000
    
    # Display calculated features
    st.markdown("### 🧮 Calculated Features")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("FOB-CIF Difference", f"₦{fob_cif_diff:,.2f}")
    with col2:
        st.metric("CIF/FOB Ratio", f"{cif_fob_ratio:.3f}")
    with col3:
        st.metric("Value per KG", f"₦{value_per_kg:,.2f}")
    with col4:
        st.metric("Tax-to-Value Ratio", f"{tax_to_value_ratio:.3f}")
    
    # Prediction button
    if st.button("🎯 Predict Tax Revenue", type="primary", use_container_width=True):
        # Prepare transaction data
        transaction_data = {
            'FOB_Value_N': fob_value,
            'CIF_Value_N': cif_value,
            'Mass_KG': mass_kg,
            'FOB_CIF_Diff': fob_cif_diff,
            'Tax_to_Value_Ratio': tax_to_value_ratio,
            'CIF_FOB_Ratio': cif_fob_ratio,
            'Value_per_KG': value_per_kg,
            'Country_of_Origin': country_of_origin,
            'Custom_Office': custom_office,
            'HS_Code_Category': hs_code_category,
            'Receipt_Year': receipt_year,
            'Receipt_Month': receipt_month,
            'Receipt_Quarter': receipt_quarter,
            'Has_Containers': has_containers,
            'High_Value_Threshold': high_value_threshold
        }
        
        # Make prediction
        predicted_tax = predict_tax(
            transaction_data,
            st.session_state.models,
            st.session_state.best_model_name,
            st.session_state.scaler,
            st.session_state.feature_columns,
            st.session_state.encoders
        )
        
        # Display results
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Tax", f"₦{predicted_tax:,.2f}", delta=f"{(predicted_tax/cif_value)*100:.2f}% of CIF")
        
        with col2:
            st.metric("Model Used", st.session_state.best_model_name, delta=f"R² = {st.session_state.models[st.session_state.best_model_name]['r2']:.3f}")
        
        with col3:
            efficiency_ratio = predicted_tax / cif_value
            st.metric("Tax Efficiency", f"{efficiency_ratio:.3f}", delta="Tax/CIF Ratio")
        
        # Detailed breakdown
        st.markdown('<div class="prediction-result">💰 Predicted Tax Revenue: ₦{:,.2f}</div>'.format(predicted_tax), unsafe_allow_html=True)
        
        # Show feature importance if using tree-based model
        if st.session_state.best_model_name in ['Random Forest', 'Gradient Boosting']:
            importance = st.session_state.models[st.session_state.best_model_name]['model'].feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(8)
            
            fig = px.bar(
                feature_importance, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 8 Feature Importances for Prediction"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_policy_simulation():
    """Display policy simulation interface"""
    st.markdown('<div class="section-header">🏛️ Policy Impact Simulation</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.markdown('<div class="warning-box">⚠️ Please train models first using the sidebar.</div>', unsafe_allow_html=True)
        return
    
    st.markdown("### Configure Policy Scenarios")
    st.markdown("Adjust the multipliers below to simulate different policy impacts. A multiplier of 1.1 means a 10% increase, while 0.9 means a 10% decrease.")
    
    # Current baseline
    df = st.session_state.df_enhanced
    baseline_tax = df['Total_Tax_N'].sum()
    baseline_cif = df['CIF_Value_N'].sum()
    
    st.markdown("### 📊 Current Baseline")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tax Collected", f"₦{baseline_tax/1e9:.2f}B")
    with col2:
        st.metric("Total CIF Value", f"₦{baseline_cif/1e12:.2f}T")
    with col3:
        st.metric("Average Tax Rate", f"{(baseline_tax/baseline_cif)*100:.2f}%")
    
    # Policy scenario configuration
    st.markdown("### 🎛️ Policy Configuration")
    
    # Predefined scenarios
    scenario_type = st.selectbox(
        "Choose Scenario Type",
        ["Custom Configuration", "Tax Rate Adjustment", "Trade Volume Impact", "Economic Conditions", "Combined Policy"]
    )
    
    scenarios = {}
    
    if scenario_type == "Custom Configuration":
        st.markdown("#### Create Custom Scenario")
        col1, col2 = st.columns(2)
        
        with col1:
            cif_multiplier = st.slider("CIF Value Multiplier", 0.5, 2.0, 1.0, 0.05, help="Affects import volumes")
            fob_multiplier = st.slider("FOB Value Multiplier", 0.5, 2.0, 1.0, 0.05, help="Affects base import costs")
            tax_multiplier = st.slider("Tax Multiplier", 0.5, 3.0, 1.0, 0.05, help="Direct tax rate changes")
        
        with col2:
            mass_multiplier = st.slider("Mass Multiplier", 0.5, 2.0, 1.0, 0.05, help="Affects cargo volumes")
            container_multiplier = st.slider("Container Count Multiplier", 0.5, 2.0, 1.0, 0.05, help="Logistics efficiency")
        
        scenarios["Custom Policy"] = {
            'CIF_Value_N': cif_multiplier,
            'FOB_Value_N': fob_multiplier,
            'Total_Tax_N': tax_multiplier,
            'Mass_KG': mass_multiplier,
            'Nbr_Of_Containers': container_multiplier
        }
    
    elif scenario_type == "Tax Rate Adjustment":
        st.markdown("#### Tax Policy Scenarios")
        col1, col2 = st.columns(2)
        
        with col1:
            scenarios["5% Tax Increase"] = {'Total_Tax_N': 1.05}
            scenarios["10% Tax Increase"] = {'Total_Tax_N': 1.10}
            scenarios["15% Tax Increase"] = {'Total_Tax_N': 1.15}
        
        with col2:
            scenarios["5% Tax Decrease"] = {'Total_Tax_N': 0.95}
            scenarios["10% Tax Decrease"] = {'Total_Tax_N': 0.90}
            scenarios["Tax Reform (20% increase)"] = {'Total_Tax_N': 1.20}
    
    elif scenario_type == "Trade Volume Impact":
        st.markdown("#### Trade Volume Scenarios")
        scenarios["Trade Growth (15%)"] = {'CIF_Value_N': 1.15, 'FOB_Value_N': 1.15, 'Mass_KG': 1.10}
        scenarios["Trade Decline (10%)"] = {'CIF_Value_N': 0.90, 'FOB_Value_N': 0.90, 'Mass_KG': 0.95}
        scenarios["Import Restriction"] = {'CIF_Value_N': 0.80, 'FOB_Value_N': 0.80, 'Mass_KG': 0.85}
        scenarios["Trade Liberalization"] = {'CIF_Value_N': 1.25, 'FOB_Value_N': 1.20, 'Mass_KG': 1.15}
    
    elif scenario_type == "Economic Conditions":
        st.markdown("#### Economic Impact Scenarios")
        scenarios["Economic Boom"] = {'CIF_Value_N': 1.30, 'FOB_Value_N': 1.25, 'Total_Tax_N': 1.10}
        scenarios["Economic Recession"] = {'CIF_Value_N': 0.75, 'FOB_Value_N': 0.80, 'Total_Tax_N': 0.90}
        scenarios["Currency Devaluation"] = {'CIF_Value_N': 1.20, 'FOB_Value_N': 1.15, 'Total_Tax_N': 1.05}
        scenarios["Inflation Impact"] = {'CIF_Value_N': 1.12, 'FOB_Value_N': 1.10, 'Total_Tax_N': 1.08}
    
    else:  # Combined Policy
        st.markdown("#### Comprehensive Policy Packages")
        scenarios["Modernization Package"] = {'Total_Tax_N': 1.08, 'CIF_Value_N': 1.05, 'Nbr_Of_Containers': 1.10}
        scenarios["Revenue Enhancement"] = {'Total_Tax_N': 1.15, 'CIF_Value_N': 0.95}
        scenarios["Trade Facilitation"] = {'CIF_Value_N': 1.20, 'FOB_Value_N': 1.15, 'Total_Tax_N': 0.98}
        scenarios["Economic Recovery"] = {'CIF_Value_N': 1.10, 'FOB_Value_N': 1.08, 'Total_Tax_N': 1.05}
    
    # Run simulation button
    if st.button("🚀 Run Policy Simulation", type="primary", use_container_width=True):
        with st.spinner("Running policy simulations..."):
            results = simulate_policy(
                df,
                st.session_state.models,
                st.session_state.best_model_name,
                st.session_state.scaler,
                st.session_state.feature_columns,
                scenarios
            )
            
            # Display results
            st.markdown("### 📊 Simulation Results")
            
            # Results table
            results_df = pd.DataFrame({
                'Scenario': list(results.keys()),
                'Predicted Tax (₦B)': [r['predicted_total']/1e9 for r in results.values()],
                'Change (₦B)': [r['change']/1e9 for r in results.values()],
                'Change (%)': [r['change_pct'] for r in results.values()]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization of results
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of changes
                fig = px.bar(
                    results_df,
                    x='Scenario',
                    y='Change (₦B)',
                    title="Tax Revenue Impact by Scenario (Billions ₦)",
                    color='Change (₦B)',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Percentage change chart
                fig = px.bar(
                    results_df,
                    x='Scenario',
                    y='Change (%)',
                    title="Percentage Change in Tax Revenue",
                    color='Change (%)',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary insights
            best_scenario = results_df.loc[results_df['Change (%)'].idxmax(), 'Scenario']
            worst_scenario = results_df.loc[results_df['Change (%)'].idxmin(), 'Scenario']
            max_gain = results_df['Change (₦B)'].max()
            max_loss = results_df['Change (₦B)'].min()
            
            st.markdown("### 🔍 Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="success-box">
                <strong>🎯 Best Performing Scenario:</strong><br>
                {best_scenario}<br>
                <strong>Revenue Gain:</strong> ₦{max_gain:.2f}B ({results_df.loc[results_df['Change (₦B)'].idxmax(), 'Change (%)']:.2f}%)
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="warning-box">
                <strong>⚠️ Highest Risk Scenario:</strong><br>
                {worst_scenario}<br>
                <strong>Revenue Impact:</strong> ₦{max_loss:.2f}B ({results_df.loc[results_df['Change (₦B)'].idxmin(), 'Change (%)']:.2f}%)
                </div>
                """, unsafe_allow_html=True)

def show_data_visualization():
    """Display interactive data visualizations"""
    st.markdown('<div class="section-header">📈 Interactive Data Visualizations</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div class="warning-box">⚠️ Please load data first using the sidebar.</div>', unsafe_allow_html=True)
        return
    
    df = st.session_state.df_enhanced
    kpis = st.session_state.kpis
    
    # Visualization selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Country Analysis", "Tax Analysis", "Temporal Trends", "HS Code Analysis", "Custom Office Performance", "Risk Analysis"]
    )
    
    if viz_type == "Country Analysis":
        st.markdown("#### 🌍 Country-wise Import Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top countries by value
            top_countries = kpis['import_volume_value']['top_7_countries_by_value']
            fig = px.pie(
                values=top_countries.values,
                names=top_countries.index,
                title="Import Value Distribution by Country"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Country-wise tax efficiency
            country_tax_eff = df.groupby('Country_of_Origin').agg({
                'Tax_to_Value_Ratio': 'mean',
                'Total_Tax_N': 'sum',
                'CIF_Value_N': 'sum'
            }).sort_values('Total_Tax_N', ascending=False).head(10)
            
            fig = px.scatter(
                country_tax_eff,
                x='CIF_Value_N',
                y='Tax_to_Value_Ratio',
                size='Total_Tax_N',
                hover_name=country_tax_eff.index,
                title="Tax Efficiency vs Import Volume by Country",
                labels={
                    'CIF_Value_N': 'Total CIF Value (₦)',
                    'Tax_to_Value_Ratio': 'Average Tax-to-Value Ratio'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Tax Analysis":
        st.markdown("#### 💰 Tax Collection Analysis")
        
        # Tax distribution by ranges
        df['Tax_Range'] = pd.cut(df['Total_Tax_N'], 
                                bins=[0, 1e6, 10e6, 50e6, 100e6, float('inf')],
                                labels=['<1M', '1M-10M', '10M-50M', '50M-100M', '>100M'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            tax_range_counts = df['Tax_Range'].value_counts().sort_index()
            fig = px.bar(
                x=tax_range_counts.index,
                y=tax_range_counts.values,
                title="Distribution of Transactions by Tax Range",
                labels={'x': 'Tax Range (₦)', 'y': 'Number of Transactions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tax vs CIF correlation
            sample_df = df.sample(n=min(1000, len(df)))  # Sample for performance
            fig = px.scatter(
                sample_df,
                x='CIF_Value_N',
                y='Total_Tax_N',
                color='Country_of_Origin',
                title="Tax Amount vs CIF Value (Sample)",
                labels={
                    'CIF_Value_N': 'CIF Value (₦)',
                    'Total_Tax_N': 'Tax Amount (₦)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of tax efficiency by country and HS code
        tax_heatmap = df.groupby(['Country_of_Origin', 'HS_Code_Category'])['Tax_to_Value_Ratio'].mean().reset_index()
        tax_pivot = tax_heatmap.pivot(index='Country_of_Origin', columns='HS_Code_Category', values='Tax_to_Value_Ratio')
        
        # Get top 10 countries and HS codes
        top_countries_list = kpis['import_volume_value']['top_7_countries_by_value'].index[:7]
        top_hs_list = kpis['compliance_processing']['top_5_hs_codes_by_value'].index[:5]
        
        tax_pivot_filtered = tax_pivot.loc[
            tax_pivot.index.isin(top_countries_list), 
            tax_pivot.columns.isin(top_hs_list)
        ]
        
        fig = px.imshow(
            tax_pivot_filtered,
            title="Tax-to-Value Ratio Heatmap (Top Countries & HS Codes)",
            labels={'color': 'Tax-to-Value Ratio'},
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Temporal Trends":
        st.markdown("#### 📅 Time-based Analysis")
        
        # Monthly aggregation
        monthly_data = df.groupby(df['Receipt_Date'].dt.to_period('M')).agg({
            'CIF_Value_N': 'sum',
            'FOB_Value_N': 'sum',
            'Total_Tax_N': 'sum',
            'ID': 'count',
            'Mass_KG': 'sum'
        }).reset_index()
        monthly_data['Receipt_Date'] = monthly_data['Receipt_Date'].astype(str)
        
        # Multi-line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['Receipt_Date'],
            y=monthly_data['CIF_Value_N']/1e9,
            mode='lines+markers',
            name='CIF Value (₦B)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_data['Receipt_Date'],
            y=monthly_data['Total_Tax_N']/1e9,
            mode='lines+markers',
            name='Tax Collected (₦B)',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_data['Receipt_Date'],
            y=monthly_data['ID']/100,  # Scale for visibility
            mode='lines+markers',
            name='Transactions (÷100)',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Monthly Trends: Import Values, Tax Collection & Transaction Volume',
            xaxis_title='Month',
            yaxis_title='Value (Billions ₦)',
            yaxis2=dict(
                title='Scaled Transaction Count',
                overlaying='y',
                side='right'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns
        col1, col2 = st.columns(2)
        
        with col1:
            quarterly_data = df.groupby('Receipt_Quarter')['Total_Tax_N'].sum()
            fig = px.bar(
                x=['Q1', 'Q2', 'Q3', 'Q4'],
                y=quarterly_data.values/1e9,
                title="Tax Collection by Quarter",
                labels={'x': 'Quarter', 'y': 'Tax Collected (₦B)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_pattern = df.groupby('Receipt_Month')['Total_Tax_N'].sum()
            fig = px.line(
                x=monthly_pattern.index,
                y=monthly_pattern.values/1e9,
                title="Monthly Tax Collection Pattern",
                labels={'x': 'Month', 'y': 'Tax Collected (₦B)'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "HS Code Analysis":
        st.markdown("#### 📦 HS Code Category Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top HS codes by frequency
            hs_freq = kpis['compliance_processing']['top_5_hs_codes_by_frequency']
            fig = px.bar(
                x=hs_freq.values,
                y=hs_freq.index,
                orientation='h',
                title="Top HS Code Categories by Frequency",
                labels={'x': 'Number of Transactions', 'y': 'HS Code Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top HS codes by value
            hs_value = kpis['compliance_processing']['top_5_hs_codes_by_value']
            fig = px.bar(
                x=hs_value.values/1e9,
                y=hs_value.index,
                orientation='h',
                title="Top HS Code Categories by Value (₦B)",
                labels={'x': 'Total CIF Value (₦B)', 'y': 'HS Code Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # HS code performance matrix
        hs_analysis = df.groupby('HS_Code_Category').agg({
            'CIF_Value_N': ['sum', 'mean'],
            'Total_Tax_N': ['sum', 'mean'],
            'Tax_to_Value_Ratio': 'mean',
            'ID': 'count'
        }).round(2)
        
        hs_analysis.columns = ['Total_CIF', 'Avg_CIF', 'Total_Tax', 'Avg_Tax', 'Avg_Tax_Ratio', 'Transaction_Count']
        hs_analysis = hs_analysis.sort_values('Total_CIF', ascending=False).head(10)
        
        fig = px.scatter(
            hs_analysis,
            x='Total_CIF',
            y='Avg_Tax_Ratio',
            size='Transaction_Count',
            hover_name=hs_analysis.index,
            title="HS Code Performance: Value vs Tax Efficiency",
            labels={
                'Total_CIF': 'Total CIF Value (₦)',
                'Avg_Tax_Ratio': 'Average Tax-to-Value Ratio'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Custom Office Performance":
        st.markdown("#### 🏢 Custom Office Performance Analysis")
        
        office_performance = df.groupby('Custom_Office').agg({
            'Total_Tax_N': 'sum',
            'CIF_Value_N': 'sum',
            'ID': 'count',
            'Tax_to_Value_Ratio': 'mean'
        }).sort_values('Total_Tax_N', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                office_performance,
                x=office_performance.index,
                y='Total_Tax_N',
                title="Tax Collection by Custom Office",
                labels={'Total_Tax_N': 'Total Tax Collected (₦)', 'index': 'Custom Office'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                office_performance,
                x=office_performance.index,
                y='ID',
                title="Transaction Volume by Custom Office",
                labels={'ID': 'Number of Transactions', 'index': 'Custom Office'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Office efficiency scatter plot
        fig = px.scatter(
            office_performance,
            x='CIF_Value_N',
            y='Tax_to_Value_Ratio',
            size='ID',
            hover_name=office_performance.index,
            title="Custom Office Efficiency: Processing Volume vs Tax Rate",
            labels={
                'CIF_Value_N': 'Total CIF Value Processed (₦)',
                'Tax_to_Value_Ratio': 'Average Tax-to-Value Ratio'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Risk Analysis
        st.markdown("#### ⚠️ Risk and Compliance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # High value imports distribution
            high_value_dist = df['High_Value_Threshold'].value_counts()
            fig = px.pie(
                values=high_value_dist.values,
                names=['Regular Value', 'High Value'],
                title="Distribution of High-Value Imports"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tax anomaly distribution
            tax_anomaly_dist = df['Tax_Anomaly'].value_counts()
            fig = px.pie(
                values=tax_anomaly_dist.values,
                names=['Normal Tax', 'Tax Anomaly'],
                title="Distribution of Tax Anomalies"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk matrix
        risk_matrix = df.groupby(['Country_of_Origin', 'HS_Code_Category']).agg({
            'Tax_Anomaly': 'mean',
            'High_Value_Threshold': 'mean',
            'ID': 'count'
        }).reset_index()
        
        # Filter for significant combinations
        risk_matrix = risk_matrix[risk_matrix['ID'] >= 10]  # At least 10 transactions
        
        fig = px.scatter(
            risk_matrix,
            x='Tax_Anomaly',
            y='High_Value_Threshold',
            size='ID',
            color='Country_of_Origin',
            title="Risk Matrix: Tax Anomalies vs High-Value Imports",
            labels={
                'Tax_Anomaly': 'Tax Anomaly Rate',
                'High_Value_Threshold': 'High-Value Import Rate'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_performance():
    """Display model performance metrics and comparisons"""
    st.markdown('<div class="section-header">⚙️ Model Performance Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.markdown('<div class="warning-box">⚠️ Please train models first using the sidebar.</div>', unsafe_allow_html=True)
        return
    
    models = st.session_state.models
    best_model_name = st.session_state.best_model_name
    
    # Model comparison metrics
    st.markdown("### 📊 Model Performance Comparison")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Model': list(models.keys()),
        'R² Score': [models[name]['r2'] for name in models.keys()],
        'RMSE (₦M)': [models[name]['rmse']/1e6 for name in models.keys()],
        'MAE (₦M)': [models[name]['mae']/1e6 for name in models.keys()]
    })
    
    # Display metrics table
    st.dataframe(metrics_df.style.highlight_max(subset=['R² Score']).highlight_min(subset=['RMSE (₦M)', 'MAE (₦M)']), use_container_width=True)
    
    # Model comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            metrics_df,
            x='Model',
            y='R² Score',
            title="Model Accuracy Comparison (R² Score)",
            color='R² Score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            metrics_df,
            x='Model',
            y='RMSE (₦M)',
            title="Model Error Comparison (RMSE)",
            color='RMSE (₦M)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model detailed analysis
    st.markdown(f"### 🏆 Best Model Analysis: {best_model_name}")
    
    best_model_results = models[best_model_name]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{best_model_results['r2']:.4f}", help="Coefficient of determination")
    with col2:
        st.metric("RMSE", f"₦{best_model_results['rmse']/1e6:.2f}M", help="Root Mean Square Error")
    with col3:
        st.metric("MAE", f"₦{best_model_results['mae']/1e6:.2f}M", help="Mean Absolute Error")
    with col4:
        st.metric("MSE", f"₦{best_model_results['mse']/1e12:.2f}T", help="Mean Square Error")
    
    # Prediction vs Actual plot
    col1, col2 = st.columns(2)
    
    with col1:
        actual = best_model_results['actual'] / 1e6
        predicted = best_model_results['predictions'] / 1e6
        
        fig = px.scatter(
            x=actual,
            y=predicted,
            title="Predicted vs Actual Tax Revenue",
            labels={'x': 'Actual Tax (₦M)', 'y': 'Predicted Tax (₦M)'}
        )
        
        # Add perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals plot
        residuals = (best_model_results['actual'] - best_model_results['predictions']) / 1e6
        predicted = best_model_results['predictions'] / 1e6
        
        fig = px.scatter(
            x=predicted,
            y=residuals,
            title="Residuals Plot",
            labels={'x': 'Predicted Tax (₦M)', 'y': 'Residuals (₦M)'}
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for tree-based models)
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        st.markdown("### 📊 Feature Importance Analysis")
        
        importance = best_model_results['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                feature_importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Feature Importances"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                feature_importance_df.head(8),
                values='Importance',
                names='Feature',
                title="Feature Importance Distribution (Top 8)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model residuals distribution
    st.markdown("### 📈 Error Distribution Analysis")
    
    residuals = (best_model_results['actual'] - best_model_results['predictions']) / 1e6
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            x=residuals,
            nbins=50,
            title="Distribution of Prediction Errors",
            labels={'x': 'Residuals (₦M)', 'y': 'Frequency'}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot of residuals
        fig = px.box(
            y=residuals,
            title="Residuals Distribution Summary",
            labels={'y': 'Residuals (₦M)'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model stability analysis
    st.markdown("### 🔍 Model Stability Analysis")
    
    # Calculate error percentiles
    error_percentiles = np.percentile(np.abs(residuals), [25, 50, 75, 90, 95])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("25th Percentile Error", f"₦{error_percentiles[0]:.2f}M")
    with col2:
        st.metric("Median Error", f"₦{error_percentiles[1]:.2f}M")
    with col3:
        st.metric("75th Percentile Error", f"₦{error_percentiles[2]:.2f}M")
    with col4:
        st.metric("90th Percentile Error", f"₦{error_percentiles[3]:.2f}M")
    with col5:
        st.metric("95th Percentile Error", f"₦{error_percentiles[4]:.2f}M")

# File upload functionality
def show_file_upload():
    """Handle file upload functionality"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Custom Data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type="csv",
        help="Upload your customs data CSV file"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['Receipt_Date'])
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
            st.sidebar.info(f"Rows: {len(df):,}, Columns: {len(df.columns)}")
            
            if st.sidebar.button("Process Uploaded Data"):
                # Process the uploaded data (same as load_and_prepare_data but for uploaded file)
                st.session_state.df_enhanced = df
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data processed successfully!")
        
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {str(e)}")

# Export functionality
def show_export_options():
    """Show export options for results"""
    if st.session_state.data_loaded or st.session_state.model_trained:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Export Options")
        
        if st.session_state.data_loaded:
            if st.sidebar.button("📊 Export KPIs"):
                kpis = st.session_state.kpis
                kpi_summary = {
                    'Total_CIF_Value_Trillions': kpis['import_volume_value']['total_cif']/1e12,
                    'Total_Tax_Collected_Billions': kpis['taxation_revenue']['total_tax_collected']/1e9,
                    'Total_Transactions': kpis['logistics_shipment']['total_shipments'],
                    'Average_Tax_Ratio': kpis['taxation_revenue']['avg_tax_to_value_ratio'],
                    'Top_Country': kpis['import_volume_value']['top_7_countries_by_value'].index[0]
                }
                
                kpi_df = pd.DataFrame([kpi_summary])
                csv = kpi_df.to_csv(index=False)
                
                st.sidebar.download_button(
                    label="Download KPI Summary",
                    data=csv,
                    file_name="trade_customs_kpi_summary.csv",
                    mime="text/csv"
                )
        
        if st.session_state.model_trained:
            if st.sidebar.button("🤖 Export Model Results"):
                models = st.session_state.models
                model_performance = pd.DataFrame({
                    'Model': list(models.keys()),
                    'R2_Score': [models[name]['r2'] for name in models.keys()],
                    'RMSE': [models[name]['rmse'] for name in models.keys()],
                    'MAE': [models[name]['mae'] for name in models.keys()]
                })
                
                csv = model_performance.to_csv(index=False)
                
                st.sidebar.download_button(
                    label="Download Model Performance",
                    data=csv,
                    file_name="model_performance_comparison.csv",
                    mime="text/csv"
                )

# Application footer
def show_footer():
    """Display application footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🏛️ Trade & Customs Analytics Dashboard</h4>
        <p>Powered by Machine Learning | Built with Streamlit</p>
        <p><strong>Features:</strong> Tax Prediction | Policy Simulation | Interactive Analytics | Performance Monitoring</p>
        <p><em>For optimal experience, ensure your data includes all required columns and follows the expected format.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    main()
    show_file_upload()
    show_export_options()
    show_footer()

# Additional utility functions for enhanced user experience
def create_sample_data():
    """Create sample data for demonstration purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    countries = ['China', 'India', 'USA', 'Germany', 'UK', 'Japan', 'France', 'Italy']
    offices = ['HM CARGO', 'APAPA', 'TIN CAN', 'OKO AMANAM', 'LAGOS PORT']
    hs_codes = ['84', '85', '87', '72', '39', '94', '73', '90', '28', '29']
    
    sample_data = {
        'ID': range(1, n_samples + 1),
        'Receipt_Date': pd.date_range('2022-01-01', '2024-12-31', periods=n_samples),
        'Country_of_Origin': np.random.choice(countries, n_samples),
        'Custom_Office': np.random.choice(offices, n_samples),
        'HS_Code_Category': np.random.choice(hs_codes, n_samples),
        'Importer': [f'IMP_{i:04d}' for i in np.random.randint(1, 200, n_samples)],
        'FOB_Value_N': np.random.lognormal(15, 1.5, n_samples),
        'CIF_Value_N': np.random.lognormal(15.2, 1.5, n_samples),
        'Mass_KG': np.random.lognormal(8, 1.5, n_samples),
        'Nbr_Of_Containers': np.random.poisson(2, n_samples) + 1,
        'Container_Size': np.random.choice(['20ft', '40ft', 'None'], n_samples, p=[0.4, 0.5, 0.1]),
        'Has_Containers': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Is_High_Value': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Is_Suspicious_Tax': np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    }
    
    df = pd.DataFrame(sample_data)
    
    # Calculate derived fields
    df['FOB_CIF_Diff'] = df['CIF_Value_N'] - df['FOB_Value_N']
    df['Total_Tax_N'] = df['CIF_Value_N'] * np.random.normal(0.18, 0.05, n_samples)
    df['Tax_per_KG'] = df['Total_Tax_N'] / df['Mass_KG']
    df['Tax_to_Value_Ratio'] = df['Total_Tax_N'] / df['CIF_Value_N']
    
    return df

# Demo mode activation
def activate_demo_mode():
    """Activate demo mode with sample data"""
    if st.sidebar.button("🎭 Demo Mode", help="Load sample data for demonstration"):
        with st.spinner("Creating sample data for demonstration..."):
            demo_df = create_sample_data()
            st.session_state.df_enhanced = demo_df
            st.session_state.data_loaded = True
            st.session_state.kpis = calculate_kpis(demo_df)
            st.sidebar.success("✅ Demo data loaded!")
            st.sidebar.info("You can now explore all features with sample data")

# Add demo mode to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎭 Demo Mode")
activate_demo_mode()

# Instructions and help
def show_help():
    """Display help and instructions"""
    with st.sidebar.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        **Getting Started:**
        1. Load data using 'Load/Refresh Data' or 'Demo Mode'
        2. Train models using 'Train Models'
        3. Explore different sections using the navigation menu
        
        **Features:**
        - **Dashboard**: Overview of key metrics and trends
        - **Tax Prediction**: Predict tax for new transactions
        - **Policy Simulation**: Simulate policy impacts
        - **Data Visualization**: Interactive charts and analysis
        - **Model Performance**: Detailed model evaluation
        
        **Requirements:**
        Your CSV should contain columns like:
        - Receipt_Date, FOB_Value_N, CIF_Value_N
        - Total_Tax_N, Mass_KG, Country_of_Origin
        - Custom_Office, HS_Code_Category, etc.
        """)

show_help()

# Performance monitoring
def show_performance_info():
    """Show performance information"""
    with st.sidebar.expander("⚡ Performance Info", expanded=False):
        if st.session_state.data_loaded:
            df_memory = st.session_state.df_enhanced.memory_usage(deep=True).sum() / 1024**2
            st.write(f"**Data Memory Usage:** {df_memory:.2f} MB")
            st.write(f"**Rows:** {len(st.session_state.df_enhanced):,}")
            st.write(f"**Columns:** {st.session_state.df_enhanced.shape[1]}")
        else:
            st.write("No data loaded")

show_performance_info()