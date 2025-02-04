import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Load and preprocess data
data_file = 'Merged_Classroom_Data_with_Location_Count.csv'
df = pd.read_csv(data_file)
df = df.dropna(subset=["Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"])
df["Buildings Name"] = df["Buildings Name"].str.strip().str.title()
df["Tell us about your classroom"] = df["Tell us about your classroom"].str.strip().str.lower()
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

# Initialize sentiment analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def calculate_sentiment(comment):
    return sia.polarity_scores(comment)['compound']

df["Sentiment Score"] = df["Tell us about your classroom"].apply(calculate_sentiment)

# Theme extraction
themes = ["spacious", "lighting", "comfort", "accessibility", "collaborative"]

def assign_themes(comment):
    assigned_themes = [theme for theme in themes if theme in comment]
    return ", ".join(assigned_themes)

df["Themes"] = df["Tell us about your classroom"].apply(assign_themes)

# AI-based summarization
summarizer = pipeline("summarization")

def generate_summary(building):
    comments = df[df["Buildings Name"] == building]["Tell us about your classroom"].tolist()
    text = " ".join(comments)[:1024]  # Limit input to avoid model overload
    if text:
        summary = summarizer(text, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
    else:
        summary = "No summary available."
    return summary

df["Summary"] = df["Buildings Name"].apply(generate_summary)

# Streamlit Application Layout
st.title("University of Alabama Huntsville - Classroom Sentiment Analysis")

st.sidebar.header("Filter Options")
selected_building = st.sidebar.selectbox("Select a building", options=df["Buildings Name"].unique())

if selected_building:
    building_data = df[df["Buildings Name"] == selected_building]
    st.subheader(f"Details for {selected_building}")
    st.write(f"Average Sentiment Score: {building_data['Sentiment Score'].mean():.2f}")
    st.write(f"Total Responses: {len(building_data)}")
    st.write(f"Themes Highlighted: {', '.join(building_data['Themes'].unique())}")
    st.write(f"Summary: {generate_summary(selected_building)}")

# Plotting
st.subheader("Classroom Sentiment Heatmap")
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="Sentiment Score",
    hover_name="Buildings Name",
    hover_data={"Sentiment Score": ":.2f", "Themes": True},
    color_continuous_scale="RdYlGn",
    title="Sentiment Scores by Classroom Location",
    zoom=15
)
fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig)

# Display quotes
st.subheader("Student Comments by Building")
for theme in themes:
    with st.expander(f"Theme: {theme.capitalize()}"):
        theme_comments = df[df["Themes"].str.contains(theme, na=False)]
        for _, row in theme_comments.iterrows():
            color = "green" if row["Sentiment Score"] > 0 else "red"
            st.markdown(f"<p style='color:{color}'>{row['Tell us about your classroom']}</p>", unsafe_allow_html=True)
