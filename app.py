import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="University of Alabama, Huntsville: Engagement Analysis",
    layout="wide",
)

# Title
st.markdown("<h1 style='text-align: center;'>University of Alabama, Huntsville: Engagement Analysis</h1>", unsafe_allow_html=True)

# Overall Sentiment Map
st.markdown("<h2 style='margin-top: 30px;'>Overall Sentiment Analysis of Classroom Spaces by Buildings</h2>", unsafe_allow_html=True)

# Map and Legend Layout
col1, col2 = st.columns([3, 1])
with col1:
    # Display Map
    sentiment_map = folium.Map(location=[34.728, -86.639], zoom_start=15, scrollWheelZoom=False)
    folium_static(sentiment_map, width=800, height=500)

with col2:
    # Legend
    st.markdown("<h3>Legend</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><span style='color: green;'>🟢 Positive (> 0.2)</span></li>
        <li><span style='color: orange;'>🟠 Neutral (-0.2 to 0.2)</span></li>
        <li><span style='color: red;'>🔴 Negative (< -0.2)</span></li>
    </ul>
    <p><strong>Total Responses:</strong> 950</p>
    """, unsafe_allow_html=True)

# Explore Themes and Responses
st.markdown("<h2 style='margin-top: 30px;'>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)
themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

# Theme Analysis
if selected_theme:
    st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)
    # Replace hardcoded data with dynamic filtering
    building_data = {
        "Buildings": ["Wilson Hall", "Shelby Center", "Business Administration Building", "Engineering Building"],
        "Sentiment": ["🟢", "🟠", "🔴", "🟢"],
    }
    df_theme = pd.DataFrame(building_data)
    st.dataframe(df_theme)

    st.markdown(f"<h3>Key Responses for '{selected_theme}'</h3>", unsafe_allow_html=True)
    responses = [
        {"response": "Spacious and modern classrooms in Wilson Hall.", "sentiment": "🟢"},
        {"response": "Good lighting but limited space in Shelby Center.", "sentiment": "🟠"},
        {"response": "Overcrowded and outdated facilities in Business Administration Building.", "sentiment": "🔴"},
        {"response": "Great accessibility features in Engineering Building.", "sentiment": "🟢"},
    ]
    for res in responses[:3]:  # Show top 3 responses
        st.markdown(f"{res['sentiment']} {res['response']}")

# Sentiment Classification by Buildings
st.markdown("<h2 style='margin-top: 30px;'>Sentiment Classification by Buildings</h2>", unsafe_allow_html=True)

# Example Treemap Data
building_summary = {
    "Buildings": ["Morton Hall", "Olin B. King Technology Hall", "Shelby Center", "Engineering Building"],
    "Average_Sentiment": [0.3, -0.5, 0.1, 0.4],
    "Count": [100, 80, 120, 150],
}
df_summary = pd.DataFrame(building_summary)
fig = px.treemap(
    df_summary,
    path=["Buildings"],
    values="Count",
    color="Average_Sentiment",
    color_continuous_scale="RdYlGn",
    title="Building Sentiment Treemap",
)
st.plotly_chart(fig, use_container_width=True)

# Building Selection for Details
selected_building = st.selectbox("Select a Building for Details:", df_summary["Buildings"])
if selected_building:
    st.markdown(f"<h3>Details for {selected_building}</h3>", unsafe_allow_html=True)
    avg_sentiment = df_summary.loc[df_summary["Buildings"] == selected_building, "Average_Sentiment"].values[0]
    count = df_summary.loc[df_summary["Buildings"] == selected_building, "Count"].values[0]
    st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
    st.write(f"**Total Responses:** {count}")
    st.markdown("<h4>Key Responses:</h4>", unsafe_allow_html=True)
    for res in responses[:3]:
        st.markdown(f"{res['sentiment']} {res['response']}")
