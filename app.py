import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
nltk.download('vader_lexicon')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="University of Alabama, Huntsville: Engagement Analysis")

# Load the dataset
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file, encoding="ISO-8859-1")

# Analyze Sentiment
def analyze_sentiment(text):
    if pd.notna(text):
        return sia.polarity_scores(text)['compound']
    return 0

def classify_sentiment(score):
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"

# File Upload Section
st.title("University of Alabama, Huntsville: Engagement Analysis")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)

    # Ensure columns match the dataset
    required_columns = ['Tell us about your classroom', 'Latitude', 'Longitude', 'Buildings Name']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The dataset must contain the following columns: {required_columns}")
        st.stop()

    # Process Data
    df['Sentiment'] = df['Tell us about your classroom'].apply(analyze_sentiment)
    df['Sentiment Category'] = df['Sentiment'].apply(classify_sentiment)

    # Section 1: Sentiment Map
    st.header("Overall Sentiment Analysis of Classroom Spaces by Buildings")
    col1, col2 = st.columns([3, 1])

    with col1:
        # Create Folium Map
        sentiment_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=15)
        for _, row in df.iterrows():
            color = 'green' if row['Sentiment'] > 0.2 else 'orange' if -0.2 <= row['Sentiment'] <= 0.2 else 'red'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{row['Buildings Name']}<br>Sentiment: {row['Sentiment']:.2f}"
            ).add_to(sentiment_map)
        folium_static(sentiment_map)

    with col2:
        # Add legend
        st.markdown("### Legend")
        st.markdown("""
        - 🟢 **Positive** (> 0.2)
        - 🟠 **Neutral** (-0.2 to 0.2)
        - 🔴 **Negative** (< -0.2)
        """)
        st.markdown(f"**Total Responses:** {len(df)}")

    # Section 2: Explore Themes
    st.header("Explore Emerging Themes and Responses")
    themes = ['Spacious', 'Lighting', 'Comfort', 'Accessibility', 'Collaborative']
    theme_selection = st.radio("Select a Theme to Explore:", themes)

    # Filter responses by theme
    if theme_selection:
        theme_responses = df[df['Tell us about your classroom'].str.contains(theme_selection, case=False, na=False)]
        st.subheader(f"Buildings Mentioning '{theme_selection}'")
        theme_summary = theme_responses.groupby('Buildings Name').agg({'Sentiment': 'mean'}).reset_index()
        theme_summary['Sentiment Category'] = theme_summary['Sentiment'].apply(classify_sentiment)

        st.write(theme_summary[['Buildings Name', 'Sentiment', 'Sentiment Category']])

        st.subheader(f"Key Responses for '{theme_selection}'")
        for _, row in theme_responses.iterrows():
            sentiment_color = '🟢' if row['Sentiment'] > 0.2 else '🟠' if -0.2 <= row['Sentiment'] <= 0.2 else '🔴'
            st.markdown(f"{sentiment_color} {row['Tell us about your classroom']} ({row['Buildings Name']})")

    # Section 3: Sentiment Classification by Buildings
    st.header("Sentiment Classification by Buildings")
    st.subheader("Building Sentiment Treemap")
    building_summary = df.groupby('Buildings Name').agg({'Sentiment': 'mean', 'Tell us about your classroom': 'count'}).reset_index()
    building_summary.rename(columns={'Tell us about your classroom': 'Response Count'}, inplace=True)
    fig = px.treemap(
        building_summary,
        path=['Buildings Name'],
        values='Response Count',
        color='Sentiment',
        color_continuous_scale='RdYlGn',
        title="Building Sentiment Treemap"
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_building = st.selectbox("Select a Building for Details:", building_summary['Buildings Name'])
    if selected_building:
        st.subheader(f"Details for {selected_building}")
        building_data = df[df['Buildings Name'] == selected_building]
        avg_sentiment = building_data['Sentiment'].mean()
        response_count = building_data.shape[0]

        st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
        st.write(f"**Total Responses:** {response_count}")
        st.write("**Key Responses:**")
        for _, row in building_data.iterrows():
            sentiment_color = '🟢' if row['Sentiment'] > 0.2 else '🟠' if -0.2 <= row['Sentiment'] <= 0.2 else '🔴'
            st.markdown(f"{sentiment_color} {row['Tell us about your classroom']}")
