import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from folium.plugins import LocateControl

# Ensure required NLTK resource is available
nltk.download('vader_lexicon')

# Page Configuration
st.set_page_config(page_title="UAH Engagement Analysis", layout="wide")

# Apply Custom CSS Styling
st.markdown("""
    <style>
        body { background-color: #f7f9fc; }
        h1 { color: #4A90E2; font-size: 36px; text-align: center; font-weight: bold; }
        h2 { color: #2A3D66; font-size: 28px; margin-top: 30px; font-weight: bold; }
        .stButton>button { background-color: #4A90E2; color: white; border-radius: 10px; padding: 10px 20px; }
        .sidebar .sidebar-content { background-color: #e3e7ee; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cb/UAH_Logo.png/250px-UAH_Logo.png")
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    st.markdown("---")
    st.markdown("### About the Tool")
    st.info("This tool analyzes sentiment data from classroom feedback and visualizes engagement trends at UAH.")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    
    # 📊 Show Scrollable Preview of Uploaded Data
    st.subheader("📌 Data Preview")
    st.dataframe(df, height=300, width=1000)
    
    # Ensure required columns exist
    required_columns = ["Latitude", "Longitude", "Buildings Name", "Tell us about your classroom"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
    else:
        df = df.dropna(subset=["Latitude", "Longitude"])
        sia = SentimentIntensityAnalyzer()
        df["Avg_Sentiment"] = df["Tell us about your classroom"].apply(lambda x: sia.polarity_scores(x)["compound"] if pd.notnull(x) else 0)
        df["Count"] = 1

        # Aggregate Data
        building_summary = df.groupby("Buildings Name").agg(
            Avg_Sentiment=("Avg_Sentiment", "mean"),
            Latitude=("Latitude", "mean"),
            Longitude=("Longitude", "mean"),
            Count=("Count", "sum"),
        ).reset_index()
        
        # 🌍 Sentiment Map
        st.subheader("📍 Sentiment Analysis Map")
        map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
        folium_map = folium.Map(location=map_center, zoom_start=15, control_scale=True)
        LocateControl().add_to(folium_map)

        for _, row in building_summary.iterrows():
            sentiment_color = "green" if row["Avg_Sentiment"] > 0.2 else "orange" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "red"
            popup_content = f"""
            <strong>{row['Buildings Name']}</strong><br>
            Average Sentiment: {row['Avg_Sentiment']:.2f}<br>
            Responses: {row['Count']}
            """
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                icon=folium.Icon(color=sentiment_color, icon="info-sign"),
                popup=folium.Popup(popup_content, max_width=250),
            ).add_to(folium_map)

        folium_static(folium_map)

        # 📊 Sentiment Breakdown
        st.subheader("📊 Sentiment Distribution")
        fig = px.bar(
            building_summary, x="Buildings Name", y="Avg_Sentiment", 
            color="Avg_Sentiment", color_continuous_scale="RdYlGn", 
            title="Sentiment Scores by Building",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 🏛️ Sentiment Classification Table
        st.subheader("🏛️ Sentiment Classification by Building")
        building_summary["Sentiment"] = building_summary["Avg_Sentiment"].apply(
            lambda x: "🟢 Positive" if x > 0.2 else "🟠 Neutral" if -0.2 <= x <= 0.2 else "🔴 Negative"
        )
        st.dataframe(building_summary[["Buildings Name", "Avg_Sentiment", "Count", "Sentiment"]], use_container_width=True)

        # 🎭 Thematic Analysis
        st.subheader("🎭 Explore Themes in Feedback")
        theme_keywords = {
            "Spacious": ["spacious", "roomy", "open space", "ample"],
            "Lighting": ["bright", "natural light", "well-lit", "dim"],
            "Comfort": ["comfortable", "seating", "cozy"],
            "Accessibility": ["accessible", "ramp", "disability"],
            "Collaboration": ["collaborative", "group", "teamwork"],
        }
        themes = list(theme_keywords.keys())
        selected_theme = st.selectbox("Select a Theme to Explore:", themes)
        theme_data = df[df["Tell us about your classroom"].str.contains('|'.join(theme_keywords[selected_theme]), case=False, na=False)]
        
        # 📌 Key Responses
        st.subheader(f"📌 Key Responses for '{selected_theme}'")
        for _, row in theme_data.iterrows():
            sentiment_icon = "🟢" if row["Avg_Sentiment"] > 0.2 else "🟠" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "🔴"
            st.markdown(f"{sentiment_icon} **{row['Buildings Name']}**: {row['Tell us about your classroom']}")

        # 📌 Summary & Next Steps
        st.markdown("---")
        st.markdown("### 📌 Summary & Recommendations")
        st.info("Use this analysis to improve classroom experiences based on student feedback and sentiment trends.")
