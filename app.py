import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer

# Page Configuration
st.set_page_config(
    page_title="University of Alabama, Huntsville: Engagement Analysis",
    layout="wide"
)

# Page Title
st.title("University of Alabama, Huntsville: Engagement Analysis")

# Section 1: Overall Sentiment Analysis of Classroom Spaces by Buildings
st.header("Overall Sentiment Analysis of Classroom Spaces by Buildings")

# Display Legend
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
    <span style="color: green; font-weight: bold;">● Positive (&gt; 0.2)</span>
    <span style="color: orange; font-weight: bold;">● Neutral (-0.2 to 0.2)</span>
    <span style="color: red; font-weight: bold;">● Negative (&lt; -0.2)</span>
    <span style="font-weight: bold;">Total Responses: 950</span>
</div>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache
def load_data():
    # Replace 'your_dataset.csv' with the path to your CSV file
    df = pd.read_csv('your_dataset.csv', encoding='ISO-8859-1')
    return df

df = load_data()

# Create Map
map_center = [34.72, -86.64]  # Example coordinates for the university
m = folium.Map(location=map_center, zoom_start=16, scrollWheelZoom=False)

# Add Sentiment Markers to Map
for _, row in df.iterrows():
    color = (
        "green" if row["Average_Sentiment"] > 0.2 else
        "orange" if -0.2 <= row["Average_Sentiment"] <= 0.2 else
        "red"
    )
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=f"<b>{row['Buildings Name']}</b><br>Avg Sentiment: {row['Average_Sentiment']:.2f}<br>Responses: {row['Count']}"
    ).add_to(m)

# Render Map
folium_static(m, width=1000, height=500)

# Section 2: Theme Exploration
st.header("Select a Theme to Explore")

theme_options = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
selected_theme = st.radio("Choose a theme:", theme_options)

if selected_theme:
    # Filter Data by Theme
    filtered_df = df[df["Theme"] == selected_theme]  # Assuming "Theme" is a column in your dataset

    # Display Buildings Mentioning the Theme
    st.subheader(f"Buildings Mentioning '{selected_theme}'")
    buildings_table = filtered_df[["Buildings Name", "Average_Sentiment"]]
    buildings_table["Sentiment"] = buildings_table["Average_Sentiment"].apply(
        lambda x: "🟢 Positive" if x > 0.2 else "🟠 Neutral" if -0.2 <= x <= 0.2 else "🔴 Negative"
    )
    st.write(buildings_table[["Buildings Name", "Sentiment"]].reset_index(drop=True))

    # Display Key Responses for the Theme
    st.subheader(f"Key Responses for '{selected_theme}'")
    responses = filtered_df["Responses"].head(5)  # Assuming "Responses" contains text
    for response in responses:
        sentiment_score = SentimentIntensityAnalyzer().polarity_scores(response)["compound"]
        color = "green" if sentiment_score > 0.2 else "orange" if -0.2 <= sentiment_score <= 0.2 else "red"
        st.markdown(f'<span style="color:{color};">●</span> {response}', unsafe_allow_html=True)

# Section 3: Sentiment Classification by Buildings
st.header("Sentiment Classification by Buildings")

# Create Treemap
building_summary = df.groupby("Buildings Name").agg(
    {"Average_Sentiment": "mean", "Count": "sum"}
).reset_index()
fig = px.treemap(
    building_summary,
    path=["Buildings Name"],
    values="Count",
    color="Average_Sentiment",
    color_continuous_scale="RdYlGn",
    title="Building Sentiment Treemap"
)
st.plotly_chart(fig, use_container_width=True)

# Building Details
selected_building = st.selectbox("Select a Building for Details:", building_summary["Buildings Name"])
if selected_building:
    st.subheader(f"Details for {selected_building}")
    building_data = df[df["Buildings Name"] == selected_building]
    avg_sentiment = building_data["Average_Sentiment"].mean()
    total_responses = building_data["Count"].sum()

    st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
    st.write(f"**Total Responses:** {total_responses}")

    st.subheader("Key Responses:")
    key_responses = building_data["Responses"].head(5)
    for response in key_responses:
        sentiment_score = SentimentIntensityAnalyzer().polarity_scores(response)["compound"]
        color = "green" if sentiment_score > 0.2 else "orange" if -0.2 <= sentiment_score <= 0.2 else "red"
        st.markdown(f'<span style="color:{color};">●</span> {response}', unsafe_allow_html=True)
