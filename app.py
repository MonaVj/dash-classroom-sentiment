import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import nltk
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

# Load Hugging Face sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Page Configuration
st.set_page_config(page_title="UAH Engagement Analysis", layout="wide")

# Apply Custom CSS Styling with Bootstrap
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background-color: #f7f9fc; }
        h1 { color: #4A90E2; font-size: 36px; text-align: center; font-weight: bold; }
        h2 { color: #2A3D66; font-size: 28px; font-weight: bold; }
        .stButton>button { background-color: #4A90E2; color: white; border-radius: 10px; padding: 10px 15px; }
        .sidebar .sidebar-content { background-color: #e3e7ee; }
        .card { margin: 10px; padding: 15px; box-shadow: 2px 2px 10px #ccc; }
        .dropdown .dropdown-menu { font-size: 0.85rem; }
        .dropdown.no-arrow .dropdown-toggle::after { display: none; }
        .card-header[data-toggle="collapse"] {
            position: relative;
            padding: 0.75rem 3.25rem 0.75rem 1.25rem;
            font-weight: bold;
        }
        .card-header[data-toggle="collapse"]::after {
            position: absolute;
            right: 0;
            top: 0;
            padding-right: 1.725rem;
            line-height: 51px;
            font-weight: 900;
            content: '\f107';
            font-family: 'Font Awesome 5 Free';
            color: #5a5c69;
        }
        .card-header.collapsed::after { content: '\f105'; }
    </style>
""", unsafe_allow_html=True)

# Navbar Integration from Bootstrap Template
st.markdown("""
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <a class="navbar-brand" href="#">Sentiment Dashboard</a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
    </nav>
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
        st.stop()
    
    df = df.dropna(subset=["Latitude", "Longitude"])
    
    @st.cache_data
    def compute_sentiment(data):
        def get_sentiment(text):
            result = sentiment_model(text[:512])[0]  # Limit text length
            return result["score"] if result["label"] == "POSITIVE" else -result["score"]
        
        data["Avg_Sentiment"] = data["Tell us about your classroom"].apply(lambda x: get_sentiment(str(x)) if pd.notnull(x) else 0)
        return data
    
    df = compute_sentiment(df)
    df["Count"] = 1
    
    building_summary = df.groupby("Buildings Name").agg(
        Avg_Sentiment=("Avg_Sentiment", "mean"),
        Latitude=("Latitude", "mean"),
        Longitude=("Longitude", "mean"),
        Count=("Count", "sum"),
    ).reset_index()
    
    # 🌍 KeplerGL Interactive Map on the Left
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("📍 Sentiment Analysis Map")
        map_config = KeplerGl(height=500)
        map_config.add_data(data=building_summary, name="Sentiment Map")
        keplergl_static(map_config)
    
    # 📊 Thematic Analysis on the Right with Bootstrap Cards
    with col2:
        st.subheader("🎭 Explore Thematic Analysis")
        theme_keywords = {
            "Spacious": ["spacious", "roomy", "open space", "ample"],
            "Lighting": ["bright", "natural light", "well-lit", "dim"],
            "Comfort": ["comfortable", "seating", "cozy"],
            "Accessibility": ["accessible", "ramp", "disability"],
            "Collaboration": ["collaborative", "group", "teamwork"],
        }
        themes = list(theme_keywords.keys())
        selected_theme = st.selectbox("Select a Theme to Explore:", themes, index=0)

        if selected_theme:
            st.markdown(f"**Buildings Related to {selected_theme}**")
            keywords = theme_keywords[selected_theme]
            theme_data = df[df["Tell us about your classroom"].str.contains('|'.join(keywords), case=False, na=False)]
            grouped_theme_data = theme_data.groupby("Buildings Name").agg(
                Avg_Sentiment=("Avg_Sentiment", "mean"),
                Count=("Count", "sum")
            ).reset_index()
            
            fig = px.bar(
                grouped_theme_data, x="Buildings Name", y="Avg_Sentiment", 
                color="Avg_Sentiment", color_continuous_scale="RdYlGn", 
                title="Sentiment Scores by Theme"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 📊 Building Selection & Breakdown
    st.subheader("🏛 Sentiment Breakdown by Building")
    building_options = ["All"] + sorted(df["Buildings Name"].dropna().unique().tolist())
    selected_building = st.selectbox("Select a Building:", building_options)
    
    if selected_building != "All":
        building_data = building_summary[building_summary["Buildings Name"] == selected_building]
        if not building_data.empty:
            fig = px.treemap(
                building_data,
                path=["Buildings Name"],
                values="Count",
                color="Avg_Sentiment",
                color_continuous_scale="RdYlGn",
                title="Building Sentiment Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            avg_sentiment = building_data["Avg_Sentiment"].values[0]
            st.subheader("🔍 Recommendations")
            recommendations = ["Maintain high-quality environments."] if avg_sentiment > 0.2 else ["Enhance collaborative spaces."] if -0.2 <= avg_sentiment <= 0.2 else ["Redesign and improve comfort."]
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.warning("No data available for this building.")
