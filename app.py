import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="University of Alabama, Huntsville Engagement Analysis",
    layout="wide"
)

# Title
st.markdown("<h1 style='text-align: center;'>University of Alabama, Huntsville Engagement Analysis</h1>", unsafe_allow_html=True)

# Overall Sentiment Map
st.markdown("<h2 style='margin-top: 30px;'>Overall Sentiment Map</h2>", unsafe_allow_html=True)

# Create columns for map and legend
col1, col2 = st.columns([3, 1])  # Map takes 3/4 of the space, legend takes 1/4
with col1:
    # Display the map
    st.markdown("<h3>Sentiment Map</h3>", unsafe_allow_html=True)

    # Example Folium map
    sentiment_map = folium.Map(location=[34.728, -86.639], zoom_start=15)
    folium_static(sentiment_map, width=800, height=500)  # Adjusted map width and height

with col2:
    # Display the legend
    st.markdown("<h3 style='margin-top: 20px;'>Legend</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><span style='color: green;'>🟢 Positive (> 0.2)</span></li>
        <li><span style='color: orange;'>🟠 Neutral (-0.2 to 0.2)</span></li>
        <li><span style='color: red;'>🔴 Negative (< -0.2)</span></li>
    </ul>
    <p><strong>Total Responses:</strong> 950</p>
    """, unsafe_allow_html=True)

# Explore Emerging Themes and Responses
st.markdown("<h2 style='margin-top: 30px;'>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)

# Create theme selection
themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

# Display buildings and key responses related to the selected theme
if selected_theme:
    st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)
    # Example buildings with sentiment dots
    building_data = {
        "Buildings": ["Wilson Hall", "Shelby Center", "Business Administration Building", "Engineering Building"],
        "Sentiment": ["🟢", "🟠", "🔴", "🟢"]
    }
    df = pd.DataFrame(building_data)
    st.dataframe(df)

    # Example key responses
    st.markdown(f"<h3>Key Responses for '{selected_theme}'</h3>", unsafe_allow_html=True)
    responses = [
        {"response": "Spacious and modern classrooms in Wilson Hall.", "sentiment": "🟢"},
        {"response": "Good lighting but limited space in Shelby Center.", "sentiment": "🟠"},
        {"response": "Overcrowded and outdated facilities in Business Administration Building.", "sentiment": "🔴"},
        {"response": "Great accessibility features in Engineering Building.", "sentiment": "🟢"}
    ]
    for res in responses[:3]:  # Show only 3-5 responses
        st.markdown(f"<p>{res['sentiment']} <em>{res['response']}</em></p>", unsafe_allow_html=True)

# Sentiment Classification by Buildings
st.markdown("<h2 style='margin-top: 30px;'>Sentiment Classification by Buildings</h2>", unsafe_allow_html=True)

# Example Treemap
building_summary = {
    "Buildings": ["Morton Hall", "Olin B. King Technology Hall", "Shelby Center", "Engineering Building"],
    "Average_Sentiment": [0.3, -0.5, 0.1, 0.4],
    "Count": [100, 80, 120, 150]
}
df_summary = pd.DataFrame(building_summary)
fig = px.treemap(
    df_summary,
    path=["Buildings"],
    values="Count",
    color="Average_Sentiment",
    color_continuous_scale="RdYlGn",
    title="Building Sentiment Treemap"
)
st.plotly_chart(fig, use_container_width=True)

# Display building details when clicked (simulate selection)
selected_building = st.selectbox("Select a Building for Details:", df_summary["Buildings"])
if selected_building:
    st.markdown(f"<h3>Details for {selected_building}</h3>", unsafe_allow_html=True)
    avg_sentiment = df_summary.loc[df_summary["Buildings"] == selected_building, "Average_Sentiment"].values[0]
    count = df_summary.loc[df_summary["Buildings"] == selected_building, "Count"].values[0]
    st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
    st.write(f"**Total Responses:** {count}")
    st.markdown("<h4>Key Responses:</h4>", unsafe_allow_html=True)
    # Example responses
    for res in responses[:3]:
        st.markdown(f"<p>{res['sentiment']} <em>{res['response']}</em></p>", unsafe_allow_html=True)
