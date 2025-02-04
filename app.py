# Import necessary libraries
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize NLTK
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Grammar correction function
def correct_grammar(text):
    if isinstance(text, str):
        return str(TextBlob(text).correct())
    return text

# Function to calculate sentiment and assign color
def get_sentiment_label_and_color(score):
    if score > 0.2:
        return "🟢 Positive"
    elif score < -0.2:
        return "🔴 Negative"
    else:
        return "🟠 Neutral"

# Function to display key quotes
def display_key_quotes(building_data):
    st.write("### Key Quotes from Students:")
    sorted_data = building_data.sort_values(by="Sentiment Score", ascending=False)
    for _, row in sorted_data.iterrows():
        sentiment_label = get_sentiment_label_and_color(row["Sentiment Score"])
        st.markdown(f"{sentiment_label} **{row['Corrected Response']}**")

# Main Streamlit app function
def main():
    # Set up page configuration
    st.set_page_config(page_title="Classroom Sentiment Dashboard", layout="wide")

    # Header Section
    st.title("📚 Classroom Sentiment and Theme Dashboard")
    st.markdown("""
        Analyze classroom sentiment and themes. Upload your dataset, explore insights, and discover trends!
    """)

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
            m = folium.Map(location=map_center, zoom_start=15)

            for _, row in building_sentiments.iterrows():
                sentiment = row["Average_Sentiment"]
                color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
                popup_text = f"""
                <b>{row['Buildings Name']}</b><br>
                <b>Avg Sentiment:</b> {row['Average_Sentiment']:.2f} | <b>Responses:</b> {row['Count']}
                """
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=8,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)

            folium_static(m)

            # Sentiment Treemap
            st.subheader("📊 Click on the Building to Learn More")
            treemap_fig = px.treemap(
                building_sentiments,
                path=["Buildings Name"],
                values="Count",
                color="Average_Sentiment",
                color_continuous_scale=["red", "orange", "green"],
                title="Building Sentiment Treemap",
            )
            treemap_fig.update_layout(margin=dict(t=25, l=0, r=0, b=0))
            selected_building = st.selectbox(
                "Select a Building for Details (Click Treemap for Suggestions):",
                building_sentiments["Buildings Name"],
            )
            st.plotly_chart(treemap_fig, use_container_width=True)

            # Display Detailed Building Analysis
            if selected_building:
                building_data = df[df["Buildings Name"] == selected_building]
                st.subheader(f"🏛️ Details for {selected_building}")
                st.write(f"**Average Sentiment Score:** {building_sentiments[building_sentiments['Buildings Name'] == selected_building]['Average_Sentiment'].values[0]:.2f}")
                st.write(f"**Total Responses:** {len(building_data)}")

                # Display Key Quotes
                display_key_quotes(building_data)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
