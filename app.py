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
st.markdown(
    "<h1 style='text-align: center;'>University of Alabama, Huntsville: Engagement Analysis</h1>",
    unsafe_allow_html=True,
)

# Section 1: Overall Sentiment Map
st.subheader("Overall Sentiment Map")

# Example dataset for buildings and sentiment
# Replace `df` and `building_data` with actual uploaded data
df = pd.DataFrame({
    "Latitude": [34.728, 34.727, 34.729, 34.725],
    "Longitude": [-86.642, -86.641, -86.643, -86.640],
    "Buildings Name": ["Wilson Hall", "Shelby Center", "Business Administration Building", "Engineering Building"],
    "Avg_Sentiment": [0.3, -0.1, -0.4, 0.5],
    "Count": [50, 40, 30, 60],
    "Themes": ["Spacious, Lighting", "Comfort", "Accessibility", "Collaborative"],
})

map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
folium_map = folium.Map(location=map_center, zoom_start=15)

for _, row in df.iterrows():
    sentiment_color = (
        "green" if row["Avg_Sentiment"] > 0.2 else "red" if row["Avg_Sentiment"] < -0.2 else "orange"
    )
    popup_content = f"""
    <strong>{row['Buildings Name']}</strong><br>
    Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
    Responses: {row['Count']}<br>
    Themes: {row['Themes']}
    """
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=10,
        color=sentiment_color,
        fill=True,
        fill_color=sentiment_color,
        popup=folium.Popup(popup_content, max_width=250),
    ).add_to(folium_map)

col1, col2 = st.columns([4, 1])
with col1:
    folium_static(folium_map)
with col2:
    st.markdown("**Legend**")
    st.markdown("🟢 Positive (> 0.2)")
    st.markdown("🟠 Neutral (-0.2 to 0.2)")
    st.markdown("🔴 Negative (< -0.2)")
    st.markdown(f"**Total Responses:** {df['Count'].sum()}")

# Section 2: Explore Emerging Themes and Responses
st.markdown("<h2 style='margin-top: 30px;'>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)

themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

if selected_theme:
    st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)

    # Replace hardcoded data with actual logic
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

# Section 3: Sentiment Classification by Buildings
st.markdown("<h2 style='margin-top: 30px;'>Sentiment Classification by Buildings</h2>", unsafe_allow_html=True)

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
