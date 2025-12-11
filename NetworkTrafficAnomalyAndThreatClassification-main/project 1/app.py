import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO

# ✅ Set page config FIRST — before any st.write/st.title
st.set_page_config(
    page_title="Network Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths - use absolute paths for Streamlit Cloud
STREAMLIT_CLOUD_MODEL_PATH = "/mount/src/networktrafficanomalyandthreatclassification/Models/BinaryClassification-xgboost/outputs/xgboost_model.pkl"
STREAMLIT_CLOUD_SCALER_PATH = "/mount/src/networktrafficanomalyandthreatclassification/Models/BinaryClassification-xgboost/outputs/scaler.pkl"

# Local paths (for development)
LOCAL_MODEL_PATH = "Models/BinaryClassification-xgboost/outputs/xgboost_model.pkl"
LOCAL_SCALER_PATH = "Models/BinaryClassification-xgboost/outputs/scaler.pkl"


# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .box {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFECB3;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown("<h1 class='main-header'>Network Traffic Anomaly Detection</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
    <p>This application helps you detect network anomalies in your CSV data. Upload your network traffic data, 
    and the application will use a pre-trained XGBoost model to identify potential anomalies.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for model loading and configuration
with st.sidebar:
    st.markdown("<h2 class='section-header'>Model Configuration</h2>", unsafe_allow_html=True)
    
    model_path = st.file_uploader("Upload model file (.pkl)", type=["pkl"], key="model_uploader")
    scaler_path = st.file_uploader("Upload scaler file (.pkl)", type=["pkl"], key="scaler_uploader")
    
    st.markdown("### OR")
    
    use_default = st.checkbox("Use default model", value=True)
    
    st.markdown("<h2 class='section-header'>Settings</h2>", unsafe_allow_html=True)
    threshold = st.slider("Anomaly threshold probability", 0.0, 1.0, 0.5, 0.01)
    show_advanced = st.checkbox("Show advanced analysis", value=False)
    
    st.markdown("<h2 class='section-header'>About</h2>", unsafe_allow_html=True)
    st.markdown("""
    This app uses XGBoost to detect network anomalies. 
    
    The model was trained on the NF-CSE-CIC-IDS2018 dataset.
    """)

# Function to load model and scaler
@st.cache_resource
def load_model_and_scaler(model_path=None, scaler_path=None, use_default=True):
    if use_default:
        # Try Streamlit Cloud paths first, then local paths
        if os.path.exists(STREAMLIT_CLOUD_MODEL_PATH) and os.path.exists(STREAMLIT_CLOUD_SCALER_PATH):
            model_path = STREAMLIT_CLOUD_MODEL_PATH
            scaler_path = STREAMLIT_CLOUD_SCALER_PATH
            
        elif os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_SCALER_PATH):
            model_path = LOCAL_MODEL_PATH
            scaler_path = LOCAL_SCALER_PATH
            
        else:
            st.error("⚠️ Default model or scaler not found! Make sure both files exist in your repository.")
            st.write("Checked paths:")
            st.write(f"- {STREAMLIT_CLOUD_MODEL_PATH}")
            st.write(f"- {STREAMLIT_CLOUD_SCALER_PATH}")
            st.write(f"- {LOCAL_MODEL_PATH}")
            st.write(f"- {LOCAL_SCALER_PATH}")
            return None, None
    
    try:
        # If file paths were uploaded through Streamlit
        if not isinstance(model_path, str):
            model = joblib.load(BytesIO(model_path.read()))
        else:
            model = joblib.load(model_path)

        if not isinstance(scaler_path, str):
            scaler = joblib.load(BytesIO(scaler_path.read()))
        else:
            scaler = joblib.load(scaler_path)

        st.success(f"✅ Model and scaler loaded successfully!")
        return model, scaler

    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        return None, None

# Function to preprocess data
def preprocess_data(df):
    """Remove categorical columns and prepare data for prediction."""
    categorical_cols = [
        'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
        'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'ICMP_TYPE', 'ICMP_IPV4_TYPE',
        'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE', 'SRC_IP_CLASS', 'DST_IP_CLASS',
        'ICMP_TYPE_LABEL', 'ICMP_IPV4_TYPE_LABEL', 'DNS_QUERY_TYPE_LABEL', 'FTP_RET_CATEGORY',
        'PROTOCOL_LABEL', 'L7_PROTO_LABEL', 'SRC_PORT_CATEGORY', 'DST_PORT_CATEGORY',
        'DST_SERVICE', 'SRC_SERVICE'
    ]
    
    # Check if target columns exist
    has_labels = ('Label' in df.columns)
    if has_labels:
        # For evaluation mode (with known labels)
        df_cleaned = df.drop(columns=categorical_cols, errors='ignore')
        y = df_cleaned['Label'] if 'Label' in df_cleaned.columns else None
        X = df_cleaned.drop(columns=['Label', 'Attack', 'Attack_Category'], errors='ignore')
        return X, y, has_labels
    else:
        # For prediction mode (without labels)
        df_cleaned = df.drop(columns=categorical_cols, errors='ignore')
        return df_cleaned, None, has_labels

# Function to make predictions
def predict_anomalies(model, scaler, X, threshold=0.5):
    """Make predictions and return results."""
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Get probabilities
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    return fig

# Function to generate feature importance plot
def plot_feature_importance(model, feature_names, top_n=20):
    # Create feature importance DataFrame
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Get top N features
    top_n = min(top_n, len(feature_importance_df))
    top_features = feature_importance_df.head(top_n)
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_features['Feature'],
        x=top_features['Importance'],
        orientation='h',
        marker=dict(color=top_features['Importance'], colorscale='Viridis')
    ))
    
    fig.update_layout(
        title='Top Features by Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=600,
        width=800
    )
    
    return fig, feature_importance_df

# Function to create an interactive scatter plot for anomaly visualization
def create_anomaly_scatter(df, y_pred, y_prob, features_to_plot=None):
    if features_to_plot is None or len(features_to_plot) < 2:
        # If no specific features are provided, use top 2 features with highest variance
        if df.shape[1] >= 2:
            variances = df.var().sort_values(ascending=False)
            features_to_plot = variances.index[:2].tolist()
        else:
            st.warning("Not enough features for scatter plot")
            return None
    
    plot_df = pd.DataFrame({
        'Feature1': df[features_to_plot[0]],
        'Feature2': df[features_to_plot[1]],
        'Predicted': y_pred,
        'Probability': y_prob
    })
    
    fig = px.scatter(
        plot_df, 
        x='Feature1', 
        y='Feature2', 
        color='Predicted',
        color_discrete_map={0: 'blue', 1: 'red'},
        hover_data=['Probability'],
        labels={
            'Predicted': 'Anomaly Prediction',
            'Feature1': features_to_plot[0],
            'Feature2': features_to_plot[1]
        },
        title=f'Anomaly Detection Visualization using {features_to_plot[0]} and {features_to_plot[1]}'
    )
    
    return fig

# Function to download predictions as CSV
def get_table_download_link(df, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Main application flow
def main():
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path, use_default)
    
    # Only show upload section if model loading failed
    if model is None or scaler is None:
        st.warning("⚠️ Please upload your model and scaler files or ensure default files are present.")
        return
    
    # Upload data file
    st.markdown("<h2 class='section-header'>Upload Data</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # Only proceed if we have data
    if uploaded_file is not None:
        st.markdown("<div class='success-box'>Data uploaded successfully! Processing...</div>", unsafe_allow_html=True)
        
        # Load and display data
        df = pd.read_csv(uploaded_file)
        with st.expander("Preview Data"):
            st.write(df.head())
            st.write(f"Shape: {df.shape}")
        
        # Preprocess data
        X, y, has_labels = preprocess_data(df)
        
        if X.empty:
            st.error("Error: No valid features found in the data after preprocessing")
            return
            
        # Make predictions
        with st.spinner('Making predictions...'):
            y_pred, y_prob = predict_anomalies(model, scaler, X, threshold)
        
        # Display results
        st.markdown("<h2 class='section-header'>Results</h2>", unsafe_allow_html=True)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['Anomaly_Prediction'] = y_pred
        results_df['Anomaly_Probability'] = y_prob
        
        # Summary statistics
        st.markdown("<h3>Prediction Summary</h3>", unsafe_allow_html=True)
        
        # Display in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(results_df))
        with col2:
            st.metric("Detected Anomalies", sum(y_pred))
        with col3:
            st.metric("Anomaly %", f"{(sum(y_pred) / len(y_pred) * 100):.2f}%")
        
        # Visualization
        st.markdown("<h3>Visualization</h3>", unsafe_allow_html=True)
        
        # Select features for visualization
        feature_cols = X.columns.tolist()
        
        # If we have more than 1 feature, create a scatter plot with 2 features
        if len(feature_cols) >= 2:
            if show_advanced:
                # Let user select features
                col1, col2 = st.columns(2)
                with col1:
                    feature1 = st.selectbox("Select X-axis feature", feature_cols, index=0)
                with col2:
                    remaining_features = [f for f in feature_cols if f != feature1]
                    feature2 = st.selectbox("Select Y-axis feature", remaining_features, index=0)
                selected_features = [feature1, feature2]
            else:
                # Use top 2 features by variance
                variances = X.var().sort_values(ascending=False)
                selected_features = variances.index[:2].tolist()
            
            # Create scatter plot
            scatter_fig = create_anomaly_scatter(X, y_pred, y_prob, selected_features)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Display feature importance
        st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
        importance_fig, importance_df = plot_feature_importance(model, X.columns)
        st.plotly_chart(importance_fig, use_container_width=True)
        
        # Evaluation metrics if true labels are available
        if has_labels and y is not None:
            st.markdown("<h3>Model Evaluation</h3>", unsafe_allow_html=True)
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(y, y_pred)
            st.pyplot(cm_fig)
            
            # Display classification report
            st.subheader("Classification Report")
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
        
        # Advanced analysis
        if show_advanced:
            st.markdown("<h3>Advanced Analysis</h3>", unsafe_allow_html=True)
            
            # Distribution of anomaly probabilities
            st.subheader("Anomaly Probability Distribution")
            fig = px.histogram(
                results_df, 
                x='Anomaly_Probability',
                color='Anomaly_Prediction',
                marginal='violin',
                nbins=50,
                title='Distribution of Anomaly Probabilities'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature importance table
            st.subheader("Feature Importance Rankings")
            st.dataframe(importance_df)
            
            # Give option to download feature importance as CSV
            st.markdown(
                get_table_download_link(importance_df, "feature_importance.csv"),
                unsafe_allow_html=True
            )
        
        # Display predicted results table
        st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
        
        # Filter options
        show_filter = st.checkbox("Show only anomalies", value=False)
        if show_filter:
            filtered_results = results_df[results_df['Anomaly_Prediction'] == 1]
        else:
            filtered_results = results_df
        
        # Display results
        st.dataframe(filtered_results)
        
        # Give option to download full results as CSV
        st.markdown(
            get_table_download_link(results_df, "anomaly_predictions.csv"),
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
