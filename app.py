import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK data
nltk.download("vader_lexicon")

# Set up page configuration
st.set_page_config(
    page_title="University of Alabama, Huntsville: Engagement Analysis",
    layout="wide",
)

# App Title
st.title("University of Alabama, Huntsville: Engagement Analysis")

# File Upload Section
st.subheader("Upload your dataset (CSV format)")
uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])

# Functions to load and process data
@st.cache_data
def load_data(uploaded_file):
    # Load the CSV file
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    
    # Drop rows with missing latitude or longitude
    df = df.dropna(subset=["Latitude", "Longitude"])
    
    # Ensure numeric conversion for latitude and longitude
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    
    return df

@st.cache_data
def analyze_sentiments(data):
    # Sentiment analysis using VADER
    sia = SentimentIntensityAnalyzer()
    data["Sentiment"] = data["Tell us about your classroom"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    return data

if uploaded_file:
    try:
        # Load and process the data
        df = load_data(uploaded_file)
        df = analyze_sentiments(df)

        # Overall Sentiment Analysis Map
        st.header("Overall Sentiment Analysis of Classroom Spaces by Buildings")
        
        # Map Creation
        sentiment_map = folium.Map(
            location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=15
        )
        for _, row in df.iterrows():
            # Define color based on sentiment score
            color = (
                "green"
                if row["Sentiment"] > 0.2
                else "orange"
                if -0.2 <= row["Sentiment"] <= 0.2
                else "red"
            )
            # Add CircleMarker to the map
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['Buildings Name']}</b><br>Sentiment: {row['Sentiment']:.2f}",
                    max_width=300,
                ),
            ).add_to(sentiment_map)
        
        # Display map and legend side by side
        col1, col2 = st.columns([3, 1])
        with col1:
            folium_static(sentiment_map, width=850, height=500)
        with col2:
            st.markdown("### Legend")
            st.markdown(
                """
                - 🟢 **Positive** (Sentiment > 0.2)  
                - 🟠 **Neutral** (-0.2 ≤ Sentiment ≤ 0.2)  
                - 🔴 **Negative** (Sentiment < -0.2)  
                **Total Responses:** {0}
                """.format(
                    len(df)
                )
            )

        # Theme Selection
        st.header("Explore Emerging Themes and Responses")
        themes = ["Spacious", "Lighting", "Comfort", "Accessibility", "Collaborative"]
        selected_theme = st.radio("Select a Theme to Explore:", themes)

        # Filter and display data based on the selected theme
        if selected_theme:
            st.subheader(f"Buildings Mentioning '{selected_theme}'")
            theme_data = df[df["Tell us about your classroom"].str.contains(selected_theme, case=False, na=False)]
            if not theme_data.empty:
                # Display table with sentiment indicators
                theme_table = theme_data[["Buildings Name", "Sentiment"]]
                theme_table["Sentiment Color"] = theme_table["Sentiment"].apply(
                    lambda x: "🟢 Positive" if x > 0.2 else "🟠 Neutral" if x >= -0.2 else "🔴 Negative"
                )
                st.table(theme_table)

                # Key responses
                st.subheader(f"Key Responses for '{selected_theme}'")
                for i, row in theme_data.iterrows():
                    sentiment_icon = (
                        "🟢" if row["Sentiment"] > 0.2 else "🟠" if row["Sentiment"] >= -0.2 else "🔴"
                    )
                    st.markdown(f"{sentiment_icon} {row['Tell us about your classroom']}")

            else:
                st.warning("No responses found for the selected theme.")

        # Sentiment Classification by Buildings
        st.header("Sentiment Classification by Buildings")
        st.subheader("Building Sentiment Treemap")
        unique_buildings = df["Buildings Name"].unique()

        # Dropdown to select a building
        selected_building = st.selectbox("Select a Building for Details:", unique_buildings)

        # Display details for the selected building
        if selected_building:
            building_data = df[df["Buildings Name"] == selected_building]
            st.subheader(f"Details for {selected_building}")
            st.markdown(f"**Average Sentiment Score:** {building_data['Sentiment'].mean():.2f}")
            st.markdown(f"**Total Responses:** {len(building_data)}")
            st.subheader("Key Responses:")
            for _, row in building_data.iterrows():
                sentiment_icon = (
                    "🟢" if row["Sentiment"] > 0.2 else "🟠" if row["Sentiment"] >= -0.2 else "🔴"
                )
                st.markdown(f"{sentiment_icon} {row['Tell us about your classroom']}")

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload a CSV file to begin.")
