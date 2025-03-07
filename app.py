import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from folium.plugins import LocateControl
import itertools

# Ensure required NLTK resource is available
nltk.download('vader_lexicon')

# Page Configuration
st.set_page_config(page_title="UAH Engagement Analysis", layout="wide")

# Apply Custom CSS Styling
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background-color: #f7f9fc; }
        h1 { color: #4A90E2; font-size: 36px; text-align: center; font-weight: bold; }
        h2 { color: #2A3D66; font-size: 28px; margin-top: 30px; font-weight: bold; }
        .stButton>button { background-color: #4A90E2; color: white; border-radius: 10px; padding: 10px 20px; }
        .sidebar .sidebar-content { background-color: #e3e7ee; }
        .tooltip-container { position: relative; display: inline-block; cursor: pointer; }
        .tooltip-text { visibility: hidden; width: 200px; background-color: black; color: #fff;
                        text-align: center; border-radius: 5px; padding: 5px;
                        position: absolute; z-index: 1; bottom: 125%; left: 50%;
                        transform: translateX(-50%); opacity: 0; transition: opacity 0.3s; }
        .tooltip-container:hover .tooltip-text { visibility: visible; opacity: 1; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cb/UAH_Logo.png/250px-UAH_Logo.png")
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    
    required_columns = ["Latitude", "Longitude", "Buildings Name", "Tell us about your classroom"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
    else:
        df = df.dropna(subset=["Latitude", "Longitude"])
        
        @st.cache_data
        def compute_sentiment(data):
            sia = SentimentIntensityAnalyzer()
            data["Avg_Sentiment"] = data["Tell us about your classroom"].apply(
                lambda x: sia.polarity_scores(x)["compound"] if pd.notnull(x) else 0
            )
            return data
        
        df = compute_sentiment(df)
        df["Count"] = 1
        
        building_summary = df.groupby("Buildings Name").agg(
            Avg_Sentiment=("Avg_Sentiment", "mean"),
            Latitude=("Latitude", "mean"),
            Longitude=("Longitude", "mean"),
            Count=("Count", "sum"),
        ).reset_index()
        
        # 🌍 Interactive Sentiment Map
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
                tooltip=f"{row['Buildings Name']} - Click for details"
            ).add_to(folium_map)
        
        folium_static(folium_map)
        
        # 📌 Thematic Feedback with Pagination
        st.subheader("🏛️ Thematic Feedback")
        theme_keywords = {
            "Spacious": ["spacious", "roomy", "open space", "ample"],
            "Lighting": ["bright", "natural light", "well-lit", "dim"],
            "Comfort": ["comfortable", "seating", "cozy"],
            "Accessibility": ["accessible", "ramp", "disability"],
            "Collaboration": ["collaborative", "group", "teamwork"],
        }
        
        # Paginate comments
        comments_per_page = 5
        feedback_pages = list(itertools.zip_longest(*[iter(df.iterrows())] * comments_per_page))
        page_num = st.number_input("Page", min_value=1, max_value=len(feedback_pages), step=1)
        for _, row in feedback_pages[page_num - 1]:
            sentiment_icon = "🟢" if row["Avg_Sentiment"] > 0.2 else "🟠" if -0.2 <= row["Avg_Sentiment"] <= 0.2 else "🔴"
            with st.expander(f"{sentiment_icon} **{row['Buildings Name']}**:"):
                st.write(row["Tell us about your classroom"])

        # 📊 Sentiment Breakdown for Selected Building
        selected_building = st.selectbox("Select a Building:", ["All"] + sorted(df["Buildings Name"].unique().tolist()))
        if selected_building != "All":
            st.subheader(f"📊 Sentiment Analysis for {selected_building}")
            building_data = building_summary[building_summary["Buildings Name"] == selected_building]
            fig = px.bar(
                building_data, x="Buildings Name", y="Avg_Sentiment", 
                color="Avg_Sentiment", color_continuous_scale="RdYlGn", 
                title="Sentiment Scores",
            )
            st.plotly_chart(fig, use_container_width=True)
            avg_sentiment = building_data["Avg_Sentiment"].values[0]
            st.subheader("🔍 Recommendations")
            recommendations = ["Maintain high-quality environments."] if avg_sentiment > 0.2 else ["Enhance collaborative spaces."] if -0.2 <= avg_sentiment <= 0.2 else ["Redesign and improve comfort."]
            for rec in recommendations:
                st.markdown(f"- {rec}")
