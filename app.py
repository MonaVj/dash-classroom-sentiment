import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page Configuration
st.set_page_config(
    page_title="University of Alabama, Huntsville: Engagement Analysis",
    layout="wide",
)

# Load Data
@st.cache
def load_data():
    data = pd.read_csv("ClassroomOpenResponses.csv")
    analyzer = SentimentIntensityAnalyzer()
    data["Sentiment"] = data["Tell us about your classroom"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    data["Sentiment_Category"] = data["Sentiment"].apply(
        lambda x: "Positive" if x > 0.2 else "Neutral" if -0.2 <= x <= 0.2 else "Negative"
    )
    return data

df = load_data()

# Title
st.markdown("<h1 style='text-align: center;'>University of Alabama, Huntsville: Engagement Analysis</h1>", unsafe_allow_html=True)

# Section 1: Overall Sentiment Map
st.markdown("<h2>Sentiment Analysis of Classroom Spaces by Buildings</h2>", unsafe_allow_html=True)

# Map and Legend
col1, col2 = st.columns([3, 1])
with col1:
    sentiment_map = folium.Map(location=[34.728, -86.639], zoom_start=15, scrollWheelZoom=False)

    for _, row in df.iterrows():
        color = "green" if row["Sentiment"] > 0.2 else "orange" if -0.2 <= row["Sentiment"] <= 0.2 else "red"
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Building: {row['Buildings Name']}<br>Sentiment: {row['Sentiment']:.2f}",
        ).add_to(sentiment_map)

    folium_static(sentiment_map, width=800, height=500)

with col2:
    st.markdown("<h3>Legend</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><span style='color: green;'>🟢 Positive (> 0.2)</span></li>
        <li><span style='color: orange;'>🟠 Neutral (-0.2 to 0.2)</span></li>
        <li><span style='color: red;'>🔴 Negative (< -0.2)</span></li>
    </ul>
    <p><strong>Total Responses:</strong> {}</p>
    """.format(len(df)), unsafe_allow_html=True)

# Section 2: Themes and Responses
st.markdown("<h2>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)
themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

if selected_theme:
    st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)
    theme_buildings = df[df["Tell us about your classroom"].str.contains(selected_theme, case=False, na=False)]
    summary = theme_buildings.groupby("Buildings Name")["Sentiment"].mean().reset_index()
    summary["Sentiment_Category"] = summary["Sentiment"].apply(
        lambda x: "🟢" if x > 0.2 else "🟠" if -0.2 <= x <= 0.2 else "🔴"
    )
    st.dataframe(summary.rename(columns={"Buildings Name": "Building", "Sentiment": "Avg Sentiment"}))

    st.markdown(f"<h3>Key Responses for '{selected_theme}'</h3>", unsafe_allow_html=True)
    for _, row in theme_buildings.iterrows():
        st.markdown(
            f"{'🟢' if row['Sentiment'] > 0.2 else '🟠' if -0.2 <= row['Sentiment'] <= 0.2 else '🔴'} {row['Tell us about your classroom']} (Building: {row['Buildings Name']})"
        )

# Section 3: Sentiment Classification by Buildings
st.markdown("<h2>Sentiment Classification by Buildings</h2>", unsafe_allow_html=True)

building_summary = df.groupby("Buildings Name").agg(
    Avg_Sentiment=("Sentiment", "mean"),
    Total_Responses=("Sentiment", "size"),
).reset_index()

fig = px.treemap(
    building_summary,
    path=["Buildings Name"],
    values="Total_Responses",
    color="Avg_Sentiment",
    color_continuous_scale="RdYlGn",
    title="Building Sentiment Treemap",
)
st.plotly_chart(fig, use_container_width=True)

# Display Details for Each Building
st.markdown("<h3>Details for All Buildings</h3>", unsafe_allow_html=True)
for _, building in building_summary.iterrows():
    st.markdown(f"### {building['Buildings Name']}")
    st.write(f"**Average Sentiment Score:** {building['Avg_Sentiment']:.2f}")
    st.write(f"**Total Responses:** {building['Total_Responses']}")
    building_responses = df[df["Buildings Name"] == building["Buildings Name"]]
    for _, response in building_responses.head(5).iterrows():  # Limit to 3-5 responses
        st.markdown(
            f"{'🟢' if response['Sentiment'] > 0.2 else '🟠' if -0.2 <= response['Sentiment'] <= 0.2 else '🔴'} {response['Tell us about your classroom']}"
        )
