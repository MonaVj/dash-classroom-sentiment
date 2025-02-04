# Import necessary libraries
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize NLTK
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Predefined themes
predefined_themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]

# Grammar correction function
def correct_grammar(text):
    if isinstance(text, str):
        return str(TextBlob(text).correct())
    return text

# Function to calculate sentiment and assign color
def get_sentiment_label_and_color(score):
    if score > 0.2:
        return "🟢"
    elif score < -0.2:
        return "🔴"
    else:
        return "🟠"

# Function to display balanced key responses
def display_key_responses(building_data):
    st.write("### Key Responses from Students:")

    # Separate positive, negative, and neutral responses
    positive_responses = building_data[building_data["Sentiment Score"] > 0.2]
    negative_responses = building_data[building_data["Sentiment Score"] < -0.2]
    neutral_responses = building_data[(building_data["Sentiment Score"] >= -0.2) & (building_data["Sentiment Score"] <= 0.2)]

    # Order responses by sentiment (positive to negative)
    all_responses = pd.concat([positive_responses, neutral_responses, negative_responses])

    # Limit to 5 responses, prioritize balanced display
    displayed_responses = all_responses.head(5)

    # Display responses with grammar correction
    for _, row in displayed_responses.iterrows():
        sentiment_color = get_sentiment_label_and_color(row["Sentiment Score"])
        st.markdown(f"{sentiment_color} **{row['Corrected Response']}**")

# Main Streamlit app function
def main():
    # Set up page configuration
    st.set_page_config(page_title="Classroom Sentiment Dashboard", layout="wide")

    # Header Section
    st.title("📚 Classroom Sentiment and Theme Dashboard")
    st.markdown("Analyze classroom sentiment and themes. Upload your dataset, explore insights, and discover trends!")

    # File upload
    data_file = st.file_uploader("Upload Classroom Data CSV", type=["csv"])

    if data_file is not None:
        try:
            # Read the file
            df = pd.read_csv(data_file, encoding="ISO-8859-1")

            # Validate required columns
            required_columns = {"Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"}
            if not required_columns.issubset(df.columns):
                st.error("CSV file is missing one or more required columns.")
                st.stop()

            # Preprocess data
            df = df.drop_duplicates()
            df = df.dropna(subset=["Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"])
            df["Buildings Name"] = df["Buildings Name"].str.strip().str.title()
            df["Tell us about your classroom"] = df["Tell us about your classroom"].str.strip()

            # Grammar correction
            df["Corrected Response"] = df["Tell us about your classroom"].apply(correct_grammar)

            # Sentiment analysis
            df["Sentiment Score"] = df["Corrected Response"].apply(
                lambda text: sia.polarity_scores(text)["compound"] if isinstance(text, str) else 0
            )

            # Aggregate data at the building level
            building_sentiments = df.groupby("Buildings Name").agg(
                Latitude=("Latitude", "mean"),
                Longitude=("Longitude", "mean"),
                Average_Sentiment=("Sentiment Score", "mean"),
                Count=("Corrected Response", "count")
            ).reset_index()

            # Overall Sentiment Analysis Map
            st.subheader("🗺️ Overall Sentiment Analysis of Classrooms By Building")
            map_center = [building_sentiments["Latitude"].mean(), building_sentiments["Longitude"].mean()]
            m = folium.Map(location=map_center, zoom_start=16, width="100%", height="75%")

            for _, row in building_sentiments.iterrows():
                sentiment = row["Average_Sentiment"]
                color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
                popup_text = f"""
                <div style="width: 250px; font-size: 14px;">
                    <b>{row['Buildings Name']}</b><br>
                    <b>Avg Sentiment:</b> {row['Average_Sentiment']:.2f}<br>
                    <b>Responses:</b> {row['Count']}
                </div>
                """
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=10,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)

            folium_static(m)

            # Themes Section - Positioned next to the map
            with st.sidebar:
                st.subheader("🎯 Key Themes in Responses")
                theme_selection = st.radio("Select a Theme:", predefined_themes)

                if theme_selection:
                    filtered_df = df[df["Corrected Response"].str.contains(theme_selection, case=False, na=False)]
                    st.write(f"Buildings mentioning '{theme_selection}':")
                    st.write(filtered_df["Buildings Name"].unique())

            # Treemap for sentiment overview
            st.subheader("📊 Sentiment Treemap - Click to Explore")
            treemap_fig = px.treemap(
                building_sentiments,
                path=["Buildings Name"],
                values="Count",
                color="Average_Sentiment",
                color_continuous_scale=["red", "orange", "green"],
                title="Building Sentiment Treemap",
            )
            treemap_fig.update_layout(margin=dict(t=25, l=0, r=0, b=0))
            clicked_building = st.plotly_chart(treemap_fig, use_container_width=True)

            # Display Details on Click
            if clicked_building:
                building_data = df[df["Buildings Name"] == clicked_building]
                st.subheader(f"🏛️ Details for {clicked_building}")
                st.write(f"**Average Sentiment Score:** {building_sentiments[building_sentiments['Buildings Name'] == clicked_building]['Average_Sentiment'].values[0]:.2f}")
                st.write(f"**Total Responses:** {len(building_data)}")

                # Display Key Responses
                display_key_responses(building_data)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
