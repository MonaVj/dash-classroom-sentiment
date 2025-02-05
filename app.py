import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

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
    try:
        # Load Data
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        # Check for required columns
        required_columns = ["Latitude", "Longitude", "Buildings Name", "Tell us about your classroom"]
        if not all(col in df.columns for col in required_columns):
            st.error("The uploaded file is missing required columns. Please check your data.")
        else:
            # Sentiment Analysis
            sia = SentimentIntensityAnalyzer()
            df["Avg_Sentiment"] = df["Tell us about your classroom"].apply(
                lambda x: sia.polarity_scores(x)["compound"] if pd.notnull(x) else 0
            )

            # Section 1: Map Visualization
            st.markdown("<h2>Overall Sentiment Analysis of Classroom Spaces</h2>", unsafe_allow_html=True)
            map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
            folium_map = folium.Map(location=map_center, zoom_start=15, scrollWheelZoom=False)

            building_summary = df.groupby("Buildings Name").agg(
                Avg_Sentiment=("Avg_Sentiment", "mean"),
                Latitude=("Latitude", "mean"),
                Longitude=("Longitude", "mean"),
                Count=("Tell us about your classroom", "count"),
            ).reset_index()

            for _, row in building_summary.iterrows():
                sentiment_color = (
                    "green" if row["Avg_Sentiment"] > 0.2 else "red" if row["Avg_Sentiment"] < -0.2 else "orange"
                )
                popup_content = f"""
                <strong>{row['Buildings Name']}</strong><br>
                Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
                Total Responses: {row['Count']}
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

            # Section 2: Explore Emerging Themes
            st.markdown("<h2>Explore Emerging Themes and Responses</h2>", unsafe_allow_html=True)

            # Predefined Keywords for Themes
            theme_keywords = {
                "Spacious": ["spacious", "roomy", "ample", "open space"],
                "Lighting": ["bright", "natural light", "well-lit", "dim"],
                "Comfort": ["comfortable", "seating", "chairs", "cozy"],
                "Accessibility": ["accessible", "ramp", "wheelchair", "disability"],
                "Collaborative": ["collaborative", "group", "discussion", "teamwork"],
            }

            themes = list(theme_keywords.keys())
            selected_theme = st.radio("Select a Theme to Explore:", themes, index=0)

            if selected_theme:
                st.markdown(f"<h3>Buildings Mentioning '{selected_theme}'</h3>", unsafe_allow_html=True)
                keywords = theme_keywords[selected_theme]
                theme_data = df[df["Tell us about your classroom"].str.contains('|'.join(keywords), case=False, na=False)]

                grouped_theme_data = theme_data.groupby("Buildings Name").agg(
                    Avg_Sentiment=("Avg_Sentiment", "mean"),
                    Count=("Tell us about your classroom", "count"),
                ).reset_index()

                grouped_theme_data["Sentiment"] = grouped_theme_data["Avg_Sentiment"].apply(
                    lambda x: "🟢" if x > 0.2 else "🟠" if -0.2 <= x <= 0.2 else "🔴"
                )

                st.dataframe(grouped_theme_data.rename(
                    columns={"Buildings Name": "Building", "Avg_Sentiment": "Average Score", "Count": "Response Count"}
                ))

                st.markdown(f"<h3>Key Responses for '{selected_theme}'</h3>", unsafe_allow_html=True)
                responses = []
                for _, row in theme_data.iterrows():
                    sentiment_icon = (
                        "🟢" if row["Avg_Sentiment"] > 0.2 else "🟠" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "🔴"
                    )
                    responses.append({
                        "response": f"*In {row['Buildings Name']}, {row['Tell us about your classroom']}*",
                        "sentiment": sentiment_icon,
                        "score": row["Avg_Sentiment"],
                    })

                positive = [r for r in responses if r["score"] > 0.2]
                neutral = [r for r in responses if -0.2 <= r["score"] <= 0.2]
                negative = [r for r in responses if r["score"] < -0.2]

                balanced_responses = positive[:2] + neutral[:2] + negative[:2]
                for res in balanced_responses:
                    st.markdown(f"{res['sentiment']} {res['response']}")

            # Section 3: Sentiment Classification by Buildings
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

                st.markdown("<h4>Key Responses:</h4>", unsafe_allow_html=True)
                building_responses = df[df["Buildings Name"] == selected_building]
                responses = []
                for _, row in building_responses.iterrows():
                    sentiment = (
                        "🟢" if row["Avg_Sentiment"] > 0.2 else "🟠" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "🔴"
                    )
                    responses.append({
                        "response": f"*{row['Tell us about your classroom']}*",
                        "sentiment": sentiment,
                        "score": row["Avg_Sentiment"],
                    })

                positive = [r for r in responses if r["score"] > 0.2]
                neutral = [r for r in responses if -0.2 <= r["score"] <= 0.2]
                negative = [r for r in responses if r["score"] < -0.2]

                balanced_responses = positive[:2] + neutral[:2] + negative[:2]
                for res in balanced_responses:
                    st.markdown(f"{res['sentiment']} {res['response']}")

                st.markdown("<h4>Design Recommendations:</h4>", unsafe_allow_html=True)
                recommendations = []
                if avg_sentiment > 0.2:
                    recommendations = [
                        "Maintain current strengths like lighting and seating.",
                        "Add collaborative spaces to enhance engagement.",
                        "Upgrade AV systems for modern teaching needs.",
                    ]
                elif -0.2 <= avg_sentiment <= 0.2:
                    recommendations = [
                        "Improve lighting and seating arrangements.",
                        "Focus on creating functional layouts for different teaching styles.",
                        "Consider better soundproofing for quieter learning environments.",
                    ]
                else:
                    recommendations = [
                        "Redesign outdated and cramped spaces.",
                        "Improve accessibility for students with disabilities.",
                        "Incorporate more natural light and modern furniture.",
                    ]

                for rec in recommendations:
                    st.markdown(f"- {rec}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin the analysis.")
