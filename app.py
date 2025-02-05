import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer

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

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    # Ensure required columns exist
    required_columns = ["Latitude", "Longitude", "Buildings Name", "Tell us about your classroom"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
    else:
        # Drop rows with missing coordinates
        df = df.dropna(subset=["Latitude", "Longitude"])

        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        df["Avg_Sentiment"] = df["Tell us about your classroom"].apply(
            lambda x: sia.polarity_scores(x)["compound"] if pd.notnull(x) else 0
        )
        df["Count"] = 1

        # Aggregate data by building
        building_summary = df.groupby("Buildings Name").agg(
            Avg_Sentiment=("Avg_Sentiment", "mean"),
            Latitude=("Latitude", "mean"),
            Longitude=("Longitude", "mean"),
            Count=("Count", "sum"),
        ).reset_index()

        # Section 1: Map and Ranking
        st.subheader("Overall Sentiment Map")
        st.markdown(
            "This map visualizes the sentiment scores for each building based on user feedback. "
            "Each point's color represents the sentiment: green (positive), orange (neutral), or red (negative)."
        )

        map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
        folium_map = folium.Map(location=map_center, zoom_start=15, control_scale=True)

        for _, row in building_summary.iterrows():
            sentiment_color = (
                "green" if row["Avg_Sentiment"] > 0.2 else
                "orange" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "red"
            )
            popup_content = f"""
            <strong>{row['Buildings Name']}</strong><br>
            Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
            Responses: {row['Count']}
            """
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=10,
                color=sentiment_color,
                fill=True,
                fill_color=sentiment_color,
                popup=folium.Popup(popup_content, max_width=250),
            ).add_to(folium_map)

        # Disable interactivity for the map
        folium_map.options["scrollWheelZoom"] = False
        folium_map.options["dragging"] = False
        folium_map.options["zoomControl"] = False

        col1, col2 = st.columns([4, 1])
        with col1:
            folium_static(folium_map)
        with col2:
            st.markdown("**Legend**")
            st.markdown("🟢 Positive (> 0.2)")
            st.markdown("🟠 Neutral (-0.2 to 0.2)")
            st.markdown("🔴 Negative (< -0.2)")
            st.markdown(f"**Total Responses:** {len(df)}")

            st.markdown("<h4>Building Rankings:</h4>", unsafe_allow_html=True)
            ranked_buildings = building_summary.sort_values("Avg_Sentiment", ascending=False)
            for _, row in ranked_buildings.iterrows():
                st.markdown(f"- {row['Buildings Name']}: **{row['Avg_Sentiment']:.2f}**")

        # Section 2: Explore Themes
        st.markdown("<h2 style='margin-top: 30px;'>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)
        theme_keywords = {
            "Spacious": ["spacious", "roomy", "open space", "ample", "not cramped"],
            "Lighting": ["bright", "natural light", "well-lit", "dark", "dim"],
            "Comfort": ["comfortable", "seating", "chairs", "desk", "cozy"],
            "Accessibility": ["accessible", "ramp", "wheelchair", "disability", "parking"],
            "Collaborative": ["collaborative", "group", "discussion", "teamwork"],
        }
        themes = list(theme_keywords.keys())
        selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

        if selected_theme:
            st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)
            keywords = theme_keywords[selected_theme]
            theme_data = df[df["Tell us about your classroom"].str.contains('|'.join(keywords), case=False, na=False)]

            grouped_theme_data = theme_data.groupby("Buildings Name").agg({
                "Tell us about your classroom": list,
                "Avg_Sentiment": "mean",
                "Count": "sum"
            }).reset_index()

            grouped_theme_data_display = grouped_theme_data[["Buildings Name", "Avg_Sentiment", "Count"]]
            grouped_theme_data_display.rename(
                columns={"Buildings Name": "Building", "Avg_Sentiment": "Average Score", "Count": "Response Count"},
                inplace=True
            )
            st.dataframe(grouped_theme_data_display, use_container_width=True)

            st.markdown(f"<h3>Key Responses for '{selected_theme}'</h3>", unsafe_allow_html=True)
            for _, row in theme_data.iterrows():
                building_name = row["Buildings Name"]
                response_text = row["Tell us about your classroom"]
                st.markdown(f"- **{building_name}**: {response_text}")

        # Section 3: Sentiment Classification by Buildings
        st.markdown("<h2 style='margin-top: 30px;'>Sentiment Classification by Buildings</h2>", unsafe_allow_html=True)
        fig = px.treemap(
            building_summary,
            path=["Buildings Name"],
            values="Count",
            color="Avg_Sentiment",
            color_continuous_scale="RdYlGn",
            title="Building Sentiment Treemap",
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_building = st.selectbox("Select a Building for Details:", building_summary["Buildings Name"])
        if selected_building:
            st.markdown(f"<h3>Details for {selected_building}</h3>", unsafe_allow_html=True)
            building_data = building_summary[building_summary["Buildings Name"] == selected_building]
            avg_sentiment = building_data["Avg_Sentiment"].values[0]
            count = building_data["Count"].values[0]
            st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
            st.write(f"**Total Responses:** {count}")

            st.markdown("<h4>Key Responses:</h4>", unsafe_allow_html=True)
            building_responses = df[df["Buildings Name"] == selected_building]
            for _, row in building_responses.head(3).iterrows():  # Limit to 3 responses
                response_text = row["Tell us about your classroom"]
                st.markdown(f"- {response_text}")

            # Custom Recommendations
            st.markdown("<h4>Design Recommendations:</h4>", unsafe_allow_html=True)
            if avg_sentiment > 0.2:
                recommendations = [
                    "Maintain strong areas like natural light and spacious layouts.",
                    "Add flexible furniture for group activities.",
                    "Improve acoustic systems to enhance learning experiences.",
                ]
            elif -0.2 <= avg_sentiment <= 0.2:
                recommendations = [
                    "Improve accessibility features like wider hallways and ramps.",
                    "Introduce ergonomic seating for better comfort.",
                    "Reassess lighting and ventilation to enhance the atmosphere."
                ]
            else:
                recommendations = [
                    "Address outdated facilities with a full redesign.",
                    "Incorporate modern technology for interactive learning.",
                    "Prioritize comfort through improved seating and space optimization."
                ]
            for rec in recommendations:
                st.markdown(f"- {rec}")
