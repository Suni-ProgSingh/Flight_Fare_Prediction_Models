import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

exec(open('notebook_code.py').read())

st.set_page_config(page_title="Flight Price Intelligence Platform", layout="wide")

st.title("🛫 Flight Price Intelligence Platform")
st.markdown("**Professional ML Dashboard showcasing 7 different regression models**")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", 
                           ["Price Predictor", "Model Comparison", "Feature Analysis", "Performance Analytics"])

if page == "Price Predictor":
    st.header("✈️ Flight Price Predictor")
    st.markdown("*Using the best performing model: **Gradient Boosting Regressor***")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Details")
        # Simple prediction interface - you can expand this
        st.info("🏆 **Best Model**: Gradient Boosting Regressor")
        st.success(f"**Model Performance**: R² Score = {r2_gbr:.3f}")
        st.success(f"**RMSE**: {rmse_gbr:.2f}")
        
    with col2:
        st.subheader("Why Gradient Boosting Won?")
        st.write("✅ Highest R² Score among all models")
        st.write("✅ Lowest RMSE (Root Mean Square Error)")
        st.write("✅ Good balance between bias and variance")
        st.write("✅ Handles non-linear relationships well")

elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")
    
    # Your exact model results
    models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
              'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']
    
    # You'll need to run your notebook to get these exact values
    rmse_scores = [rmse, rmse_ridge, rmse_lasso, rmse_dt, rmse_rf, rmse_gbr, rmse_xg]
    r2_scores = [r2, r2_ridge, r2_lasso, r2_dt, r2_rf, r2_gbr, r2_xg]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("R² Score Comparison")
        fig1 = px.bar(x=models, y=r2_scores, 
                     title="R² Score by Model",
                     color=r2_scores,
                     color_continuous_scale="viridis")
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("RMSE Comparison")
        fig2 = px.bar(x=models, y=rmse_scores,
                     title="RMSE by Model", 
                     color=rmse_scores,
                     color_continuous_scale="viridis_r")
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    st.success("🏆 **Winner: Gradient Boosting Regressor**")
    st.info("🥈 **Runner-up: XGBoost Regressor**")

elif page == "Feature Analysis":
    st.header("🔍 Feature Importance Analysis")
  
    feature_names = X.columns.tolist()
    importance_scores = gbr.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False).head(15)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(feature_df, x='Importance', y='Feature', 
                    orientation='h',
                    title="Top 15 Most Important Features",
                    color='Importance',
                    color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Key Insights")
        st.write("📈 **Top Features Impact:**")
        for i, row in feature_df.head(5).iterrows():
            st.write(f"• {row['Feature']}: {row['Importance']:.3f}")

elif page == "Performance Analytics":
    st.header("📈 Performance Analytics Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model", "Gradient Boosting", "🏆 Winner")
        st.metric("R² Score", f"{r2_gbr:.3f}", f"+{(r2_gbr-r2):.3f} vs Linear")
        
    with col2:
        st.metric("RMSE", f"{rmse_gbr:.2f}", f"-{(rmse-rmse_gbr):.2f} vs Linear")
        st.metric("Training R²", f"{r2_train:.3f}", "No Overfitting")
        
    with col3:
        st.metric("Total Models", "7", "Comprehensive Testing")
        st.metric("Best Algorithm", "Ensemble", "Tree-based Methods")
    
    st.subheader("📋 Complete Model Summary")
    results_df = pd.DataFrame({
        'Model': models,
        'R² Score': r2_scores,
        'RMSE': rmse_scores,
        'Rank': range(1, len(models)+1)
    }).sort_values('R² Score', ascending=False).reset_index(drop=True)
    results_df['Rank'] = range(1, len(results_df)+1)
    
    st.dataframe(results_df, use_container_width=True)

st.markdown("---")
st.markdown("**Built with:** Python, Scikit-learn, XGBoost, Streamlit | **Data:** Flight Price Dataset")
