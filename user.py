import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Title and Configuration
st.set_page_config(page_title="App User Behavior Segmentation Dashboard", layout="wide")

st.title("📊 App User Behavior & Segmentation Dashboard")
st.write("This dashboard comprehensively presents the behavioral analysis results of 50,000 app users using K-Means Clustering and PCA technology.")

# 2. Load CSV Data
@st.cache_data
def load_data():
    df = pd.read_csv('clustered_users_with_pca.csv')
    return df

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("🔍 Cluster Filter")
cluster_options = sorted(df['cluster'].unique())
selected_cluster = st.sidebar.selectbox("Select Cluster to Analyze:", cluster_options)

# Filtered data based on selection
filtered_df = df[df['cluster'] == selected_cluster]

# 4. Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", len(df))
col2.metric(f"Cluster {selected_cluster} Users", len(filtered_df))
col3.metric("Total Clusters", len(cluster_options))

st.markdown("---")

# 5. Standard Core Visualizations (Cluster Distribution & PCA Separation)
st.subheader("📈 1. Cluster Distribution & PCA Behavioral Separation")

c1, c2 = st.columns(2)

with c1:
    st.write("### *User Count per Cluster*")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='cluster', palette='viridis', ax=ax)
    ax.set_xlabel("Clusters")
    ax.set_ylabel("User Count")
    st.pyplot(fig)

with c2:
    st.write("### *Clear Behavioral Separation using PCA*")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='cluster', palette='Set1', alpha=0.6, ax=ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

st.markdown("---")

# 6. Standard Key Business Feature Charts (2x2 Grid Layout for Clean Look)
st.subheader("💡 2. Standard Business & Engagement Feature Analysis")

# Row 1: Sessions per Week & Churn Risk Score
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.write("### *Average Sessions per Week*")
    fig_s1, ax_s1 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='cluster', y='sessions_per_week', palette='crest', ax=ax_s1, errorbar=None)
    ax_s1.set_xlabel("Clusters")
    ax_s1.set_ylabel("Avg Sessions / Week")
    st.pyplot(fig_s1)

with row1_col2:
    st.write("### *Average Churn Risk Score*")
    fig_s2, ax_s2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='cluster', y='churn_risk_score', palette='magma', ax=ax_s2, errorbar=None)
    ax_s2.set_xlabel("Clusters")
    ax_s2.set_ylabel("Avg Churn Risk Score")
    st.pyplot(fig_s2)

# Row 2: Subscription Type & Daily Active Minutes
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.write("### *Subscription Type Distribution*")
    fig_s3, ax_s3 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='cluster', hue='subscription_type', palette='Set2', ax=ax_s3)
    ax_s3.set_xlabel("Clusters")
    ax_s3.set_ylabel("User Count")
    st.pyplot(fig_s3)

with row2_col2:
    st.write("### *Daily Active Minutes*")
    fig_s4, ax_s4 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='cluster', y='daily_active_minutes', palette='coolwarm', ax=ax_s4, errorbar=None)
    ax_s4.set_xlabel("Clusters")
    ax_s4.set_ylabel("Avg Daily Active Minutes")
    st.pyplot(fig_s4)

st.markdown("---")

# 7. Customer-Level Identification Table for Selected Cluster
st.subheader("👥 3. Customer-Level Identification Table")
st.write(f"Displaying specific customer records belonging to *Cluster {selected_cluster}*:")
st.dataframe(filtered_df[['user_id', 'age', 'gender', 'country', 'device_type', 'engagement_score', 'cluster']].head(100))

st.markdown("---")

# 8. Business Insights & Final Summary
st.subheader("📌 4. Business Insights & Final Summary")
cluster_insights = {
    0: "*Cluster 0 Profile:* Stable engagement with moderate activity. *Actionable Strategy:* Introduce loyalty rewards and premium upselling options to maximize customer lifetime value (LTV).",
    1: "*Cluster 1 Profile:* Exhibits a high churn risk score and distinct behavioral shifts. *Actionable Strategy:* Immediate retention campaigns, special discounts, and targeted push notifications are critical here.",
    2: "*Cluster 2 Profile:* Demonstrates high frequency and very low churn risk. *Actionable Strategy:* Maintain the current product experience, reward brand loyalty, and encourage community engagement.",
    3: "*Cluster 3 Profile:* Represents moderate-risk users with mixed activity patterns. *Actionable Strategy:* Monitor behavioral trends closely and offer personalized content recommendations."
}

selected_insight = cluster_insights.get(selected_cluster, "Analyze specific cluster characteristics to deploy targeted marketing campaigns.")

st.info(f"""
- *Active Analysis for Cluster {selected_cluster}:* {selected_insight}
- *Targeted Decision Making:* By isolating this specific cluster, marketing and product teams can avoid broad, ineffective campaigns and deploy precise interventions.
""")
