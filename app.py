import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from transformers import pipeline

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Classroom Sentiment Analysis"

# Load and preprocess data
data_file = 'Merged_Classroom_Data_with_Location_Count.csv'

df = pd.read_csv(data_file)
df = df.dropna(subset=["Tell us about your classroom", "Latitude", "Longitude", "Buildings Name"])
df["Buildings Name"] = df["Buildings Name"].str.strip().str.title()
df["Tell us about your classroom"] = df["Tell us about your classroom"].str.strip().str.lower()
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

# Initialize sentiment analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def calculate_sentiment(comment):
    return sia.polarity_scores(comment)['compound']

df["Sentiment Score"] = df["Tell us about your classroom"].apply(calculate_sentiment)

# Theme extraction
themes = ["spacious", "lighting", "comfort", "accessibility", "collaborative"]

def assign_themes(comment):
    assigned_themes = []
    for theme in themes:
        if theme in comment:
            assigned_themes.append(theme)
    return ", ".join(assigned_themes)

df["Themes"] = df["Tell us about your classroom"].apply(assign_themes)

# AI-based summarization
summarizer = pipeline("summarization")

def generate_summary(building):
    comments = df[df["Buildings Name"] == building]["Tell us about your classroom"].tolist()
    text = " ".join(comments)[:1024]  # Limit input to avoid model overload
    if text:
        summary = summarizer(text, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
    else:
        summary = "No summary available."
    return summary

df["Summary"] = df["Buildings Name"].apply(generate_summary)

# Aggregate building data
building_summary = df.groupby("Buildings Name").agg({
    "Sentiment Score": "mean",
    "Tell us about your classroom": "count",
    "Themes": lambda x: ', '.join(set(", ".join(x).split(', '))),
    "Summary": "first"
}).reset_index()

building_summary.rename(columns={"Tell us about your classroom": "Count"}, inplace=True)

# Layout
dropdown_options = [{'label': b, 'value': b} for b in building_summary["Buildings Name"].unique()]

app.layout = html.Div([
    html.H1("University of Alabama Huntsville - Classroom Sentiment Analysis", style={'textAlign': 'center'}),
    
    dcc.Graph(id='map'),
    
    dcc.Dropdown(id='building-dropdown', options=dropdown_options, placeholder="Select a building"),
    
    html.Div(id='building-details', style={"marginTop": "20px"}),
    
    dcc.Graph(id='treemap', style={"marginTop": "20px"}),
    
    html.Div([
        html.H3("Filter by Theme:"),
        html.Div([
            html.Button(theme.capitalize(), id=f"theme-{theme}", n_clicks=0, style={"marginRight": "10px"})
            for theme in themes
        ], style={"marginBottom": "20px"}),
        html.Div(id="theme-filter-results")
    ])
])

# Callbacks
@app.callback(
    Output('map', 'figure'),
    [Input('building-dropdown', 'value')]
)
def update_map(selected_building):
    filtered_df = df if not selected_building else df[df["Buildings Name"] == selected_building]
    map_fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="Sentiment Score",
        hover_name="Buildings Name",
        hover_data={"Sentiment Score": ":.2f", "Themes": True},
        color_continuous_scale="RdYlGn",
        title="Sentiment Scores by Classroom Location",
        zoom=15
    )
    map_fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return map_fig

@app.callback(
    Output('building-details', 'children'),
    [Input('building-dropdown', 'value')]
)
def update_building_details(selected_building):
    if not selected_building:
        return "Select a building to see details."
    
    building_data = building_summary[building_summary["Buildings Name"] == selected_building]
    if building_data.empty:
        return "No data available for the selected building."
    
    building_info = building_data.iloc[0]
    return html.Div([
        html.H3(f"Details for {selected_building}"),
        html.P(f"Average Sentiment Score: {building_info['Sentiment Score']:.2f}"),
        html.P(f"Total Responses: {building_info['Count']}"),
        html.P(f"Themes Highlighted: {building_info['Themes']}"),
        html.P(f"Summary: {building_info['Summary']}")
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
