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

    # 📌 **New Feature: Show Full Scrollable Preview of Uploaded File**
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df, height=400, width=1000)  # Enables scrolling with a fixed height

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

        # Section 1: Map
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

        col1, col2 = st.columns([4, 1])
        with col1:
            folium_static(folium_map)
        with col2:
            st.markdown("**Legend**")
            st.markdown("🟢 Positive (> 0.2)")
            st.markdown("🟠 Neutral (-0.2 to 0.2)")
            st.markdown("🔴 Negative (< -0.2)")
            st.markdown(f"**Total Responses:** {len(df)}")

        # Section 2: Sentiment Classification by Buildings
        st.markdown("<h2>Sentiment Classification by Buildings</h2>", unsafe_allow_html=True)
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

            # 📌 Section 3: Key Responses
            st.markdown("<h4>Key Responses:</h4>", unsafe_allow_html=True)
            building_responses = df[df["Buildings Name"] == selected_building]
            responses = []
            for _, row in building_responses.iterrows():
                sentiment = (
                    "🟢" if row["Avg_Sentiment"] > 0.2 else
                    "🟠" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "🔴"
                )
                if len(row["Tell us about your classroom"].split()) > 5:
                    responses.append({
                        "response": row["Tell us about your classroom"],
                        "sentiment": sentiment,
                        "score": row["Avg_Sentiment"]
                    })
            positive = [r for r in responses if r["score"] > 0.2]
            neutral = [r for r in responses if -0.2 <= r["score"] <= 0.2]
            negative = [r for r in responses if r["score"] < -0.2]

            balanced_responses = positive[:2] + neutral[:2] + negative[:2]
            for res in balanced_responses:
                st.markdown(f"{res['sentiment']} {res['response']}")

            # 📌 Section 4: Design Recommendations
            st.markdown("<h4>Design Recommendations:</h4>", unsafe_allow_html=True)
            if avg_sentiment > 0.2:
                recommendations = [
                    "Enhance existing strengths such as lighting and seating comfort.",
                    "Introduce advanced collaboration tools for group work.",
                    "Add flexible layouts to adapt to modern teaching needs."
                ]
            elif -0.2 <= avg_sentiment <= 0.2:
                recommendations = [
                    "Address lighting and acoustics to create a balanced environment.",
                    "Incorporate ergonomic furniture for long sessions.",
                    "Expand accessibility features such as ramps and elevators."
                ]
            else:
                recommendations = [
                    "Prioritize redesigning outdated and cramped spaces.",
                    "Incorporate modern AV systems for better teaching experiences.",
                    "Add more natural light and ventilation to improve comfort."
                ]
            for rec in recommendations:
                st.markdown(f"- {rec}")
