import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(page_title="Telecom !!!",page_icon=":bar_chart:",layout = "wide")

st.title('Customer Satisfaction Dashboard')

st.header('Upload Engagement and Experience Scores CSV Files')

uploaded_eng_file = st.file_uploader("choose an engagement Scores CSV file",type = "csv")
uploaded_exp_file = st.file_uploader("choose an experience Scores CSV file",type = "csv")

    
if uploaded_eng_file is not None and uploaded_exp_file is not None:
    # Read uploaded CSV files into DataFrames
    try:
        eng_df = pd.read_csv(uploaded_eng_file)
        exp_df = pd.read_csv(uploaded_exp_file)
    except Exception as e:  # Catch potential errors during file reading
        st.error(f"An error occurred while reading the files: {e}")
        # Optionally, provide instructions on correcting CSV format issues

    else:
        st.write("Engagement Scores Data")
        st.write(eng_df.head())

        st.write("Experience Scores Data")
        st.write(exp_df.head())

        # Combine data (assuming columns are compatible)
        df = pd.concat([eng_df, exp_df], axis=1)

        # Handle duplicate column names if necessary
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
        
        df1 = df.fillna(df.median(numeric_only = True))
        
        # calculate satisfaction Score
        df1['Satisfaction Score'] = df1[['engagement_score','Experience Score']].mean(axis = 1)
        
        #st.header('1.satisfaction Score Distribution')
        fig_sat_dist = px.histogram(df1,x = 'Satisfaction Score',nbins=30,title = 'Distribution of satisfaction Scores')
        st.plotly_chart(fig_sat_dist)
        
        st.header('2. Top 10 satisfied customers')
        
        # Display Top 10 satisfied customer
        top_satisfied_customers = df1[['MSISDN/Number','Satisfaction Score']].sort_values(by='Satisfaction Score',ascending = False).head(10)
        st.write(top_satisfied_customers)
        
        # spliting data
        
        X = df1[['engagement_score','Experience Score']]
        y = df1['Satisfaction Score']
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Split data into training and testing sets (adjust test_size as needed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest Regressor model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train,y_train)

        # Make predictions on the testing set
        y_pred_rf = rf_model.predict(X_test)

        # Calculate and display evaluation metrics
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)

        st.write(f'Random Forest Mean Squared Error: {mse_rf:.4f}')
        st.write(f'Random Forest R-squared: {r2_rf:.4f}')
        
        # save and Load Model 
        
        joblib.dump(rf_model,'rf_model.pkl')
        with open('rf_model.pkl','rb') as file:
         loaded_rf_model = joblib.load(file)
            
        st.header('4. Model Performance')

        fig_actual_vs_predicted_rf = go.Figure()
        fig_actual_vs_predicted_rf.add_trace(go.Scatter(
            x=y_test,
            y=y_pred_rf,
            mode='markers',
            name='Predicted vs Actual',
            marker=dict(color='blue')
        ))

        fig_actual_vs_predicted_rf.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))

        fig_actual_vs_predicted_rf.update_layout(
            title='Random Forest Regressor: Actual vs Predicted Satisfaction Score',
            xaxis_title='Actual Satisfaction Score',
            yaxis_title='Predicted Satisfaction Score'
        )
        st.plotly_chart(fig_actual_vs_predicted_rf)

        fig_residual_rf = go.Figure()
        residual_rf = y_test - y_pred_rf

        fig_residual_rf.add_trace(go.Scatter(
            x=y_pred_rf,
            y=residual_rf,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue')
        ))

        fig_residual_rf.add_trace(go.Scatter(
            x=[y_pred_rf.min(), y_pred_rf.max()],
            y=[0, 0],
            mode='lines',
            name='Zero Residual Line',
            line=dict(color='red', dash='dash')
        ))

        fig_residual_rf.update_layout(
            title='Random Forest Regressor Residual Plot',
            xaxis_title='Predicted Satisfaction Score',
            yaxis_title='Residuals'
        )
        st.plotly_chart(fig_residual_rf)
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance',ascending=False)

        fig_features_importance = px.bar(feature_importance, x ='Feature', y='Importance', title='Feature Importances')
        st.plotly_chart(fig_features_importance)

        # K-Means Clustering
        st.header('5. K-Means Clustering')

        # Normalize the data (using StandardScaler for better scaling)
        scaler = StandardScaler()  # Create a StandardScaler object
        normalized_df = pd.DataFrame(scaler.fit_transform(df1[['engagement_score', 'Experience Score']]))


        # KMeans with 3 number of clusters 
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(normalized_df)
        df1['cluster'] = kmeans.labels_

        # Visualize the clusters (consider scatter plot or boxplots)
        st.subheader('Cluster Distribution')

        # Assuming 'engagement_score' and 'Experience Score' are your features, create a scatter plot
        st.plotly_chart(px.scatter(df1, x='engagement_score', y='Experience Score', color='cluster', title='Customer Clusters'))


        # After fitting the KMeans model
        cluster_labels = {
            0: "High Engagement, High Experience",  # Replace with meaningful labels
            1: "Moderate Engagement, Moderate Experience",
            2: "Low Engagement, Low Experience"  # Replace with meaningful labels
        }

    # Calculate average scores per cluster
    avg_scores = df1.groupby('cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean',
        'engagement_score': 'mean'
    }).reset_index()

    # Display cluster interpretation and average scores
    st.subheader("Cluster Interpretation")
    for i, row in avg_scores.iterrows():
        cluster = cluster_labels[row['cluster']]
        satisfaction = row['Satisfaction Score']
        experience = row['Experience Score']
        engagement = row['engagement_score']
        st.write(f"- {cluster}: Satisfaction={satisfaction:.2f}, Experience={experience:.2f}, Engagement={engagement:.2f}")
       
    # Average Satisfaction Scores by Cluster

    avg_sat_exp = df1.groupby('cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean',
            'engagement_score': 'mean'
    }).reset_index()

    fig_avg_scores = px.bar(avg_sat_exp, x='cluster', y=['Satisfaction Score', 'Experience Score', 'engagement_score'],
                         title='Average Scores by Cluster', labels={'cluster': 'Cluster'})
    st.plotly_chart(fig_avg_scores)

    # Plotting clusters

    fig_clusters = go.Figure()
    fig_clusters.add_trace(go.Scatter(
        x=df1['engagement_score'],
        y=df1['Experience Score'],
        mode='markers',
        marker=dict(color=df1['cluster'], colorscale='viridis', size=10),
        name='Clustered Data'
    ))  # Add closing parenthesis here

    fig_clusters.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),

        name='Centroids'
    ))

    fig_clusters.update_layout(
        title='Engagement vs. Experience Score with Clustering',
        xaxis_title='Engagement Score',
        yaxis_title='Experience Score'
    )

    # Add a colorbar to the scatter plot (optional)
    fig_clusters.update_layout(
        coloraxis_showscale=True
    )

    st.plotly_chart(fig_clusters)
    
    st.header('6. User count by cluster')
    user_count = df1.groupby('cluster').size().reset_index(name='user_count')
    
    fig_user_count = px.bar(user_count,x = 'cluster',y ='user_count',title = 'Number of Users by cluster')
    st.plotly_chart(fig_user_count)
    
    st.write("Dashboard successfully created.")
else:
    st.write("please upload both engagement and experience scores csv files.")
