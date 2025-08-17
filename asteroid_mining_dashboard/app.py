import streamlit as st
import pandas as pd
import joblib
from utils import get_resource_summary  # <-- IMPORT from your new utils file

st.set_page_config(layout="wide")
st.title("🚀 Asteroid Mining Potential Dashboard")

# --- Load Data and Model ---
df = pd.read_csv("final_dashboard_data.csv")

# --- Sidebar Filters ---
st.sidebar.title("Filters")
min_aps = st.sidebar.slider("Minimum Asteroid Potential Score (APS)", 0, 100, 70)
# ... other filters ...

filtered_df = df[df['aps'] >= min_aps]

# --- Key Insights Section ---
st.header("Key Insights")
top_asteroid = filtered_df.sort_values('aps', ascending=False).iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Top Mining Candidate", top_asteroid['full_name'])
col2.metric("Top APS Score", f"{top_asteroid['aps']}/100")
col3.metric("Total High-Value Targets", f"{len(filtered_df[filtered_df['aps'] > 80])}")
st.markdown("---")

# --- Main Asteroid List ---
st.header("Asteroid Mission Shortlist")

# Add a selectbox for a clean detailed view
selected_asteroid_name = st.selectbox(
    "Select an Asteroid for a Detailed Profile",
    options=filtered_df.sort_values('aps', ascending=False)['full_name']
)

if selected_asteroid_name:
    asteroid_data = df[df['full_name'] == selected_asteroid_name].iloc[0]

    st.subheader(f"Profile: {asteroid_data['full_name']}")

    # --- Use info tooltips to explain everything ---
    aps_col, res_col, acc_col = st.columns(3)
    aps_col.metric(
        "Overall Potential (APS)",
        f"{asteroid_data['aps']}/100",
        help="A combined score of resource value, mission accessibility, and our data confidence. Higher is better."
    )
    res_col.metric(
        "Resource Confidence",
        f"{int(asteroid_data['resource_confidence_score']*100)}%",
        help="Our confidence in the predicted composition based on the number of observations."
    )
    acc_col.metric(
        "Mission Accessibility",
        f"{int(asteroid_data['mission_accessibility_score']*100)}/100",
        help="A score representing how easy and fuel-efficient it is to reach this asteroid."
    )

    st.info(f"**Resource Summary:** {get_resource_summary(asteroid_data['predicted_class'])}")

    # Add expander for detailed charts and data
    with st.expander("View Detailed Analysis"):
        # Your charts (like the 3D orbit plot) and raw data go here.
        st.write("Detailed charts and raw data would be displayed here.")
