from pathlib import Path 

import pandas as pd 
import streamlit as st 

# PROJECT PATHS 

PROJECT_ROOT = Path(
     r"C:\Users\Sanu\OneDrive\Desktop\Spotify-Global-Hit-Intelligence"
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

CLEAN_DATA_PATH = (
    DATA_DIR / "spotify_global_chart_totals_lyrics_cleaned.csv"
)

NLP_DATA_PATH = (
    DATA_DIR / "spotify_lyrics_nlp_features.csv"
)

CHART_ANALYTICS_PATH = (
    DATA_DIR / "spotify_chart_performance_analytics.csv"
)

# DATA LOADING 

@st.cache_data
def load_clean_data():
    """Load the cleaned Spotify dataset."""
    return pd.read_csv(
        CLEAN_DATA_PATH,
        low_memory=False
    )

@st.cache_data
def load_song_explorer_data():
    """
    Load cleaned Spotify chart data and merge NLP features.

    NLP features are aligned using record_id.
    Sentiment category is recreated from sentiment polarity
    to make the application independent of the CSV's category column.
    """

    df = pd.read_csv(
        CLEAN_DATA_PATH,
        low_memory=False
    )

    nlp = pd.read_csv(
        NLP_DATA_PATH,
        low_memory=False
    )

    # Recreate stable row-level IDs
    df["record_id"] = range(len(df))
    nlp["record_id"] = range(len(nlp))

    nlp_columns = [
        "record_id",
        "lyrics_unique_word_count",
        "meaningful_word_count",
        "lexical_diversity",
        "avg_word_length",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "dominant_topic",
        "topic_probability",
        "lyric_complexity_group"
    ]

    df = df.merge(
        nlp[nlp_columns],
        on="record_id",
        how="left",
        validate="one_to_one"
    )

    # Recreate sentiment category 

    df["sentiment_category"] = "Neutral"

    df.loc[
        df["sentiment_polarity"] > 0.05,
        "sentiment_category"
    ] = "Positive"

    df.loc[
        df["sentiment_polarity"] < -0.05,
        "sentiment_category"
    ] = "Negative"

    return df 

@st.cache_data
def load_nlp_data():
    """Load lyrics NLP features."""
    return pd.read_csv(
        NLP_DATA_PATH,
        low_memory=False
    )

#st.cache_data
def load_chart_analytics():
    """Load chart-performance analytics."""
    return pd.read_csv(
        CHART_ANALYTICS_PATH,
        low_memory=False 
    )


# FORMATTING FUNCTIONS

def format_number(value):
    """Format large numbers for dashboard display."""

    if pd.isna(value):
        return "N/A"


    value = float(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_percentage(value):
    """Format decimal percenatge."""

    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def format_weeks(value):
    """Format chart weeks."""

    if pd.isna(value):
        return "N/A"

    return f"{int(value):,}"


# KPI CALCULATIONS

def calculate_kpis(df):
    """Calculate executive dashboard KPIs."""

    kpis = {
        "total_songs": df["song name"].nunique(),

        "unique_artists": df["artist name"].nunique(),

        "lyrics_available": (
            df["lyrics_available"].sum()
            if "lyrics_available" in df.columns
            else 0
        ),

        "top10_songs": (
            (df["t10"] > 0).sum()
        ),

        "max_weeks": df["wks"].max(),

        "max_total_streams": df["total"].max(),

        "median_weeks": df["wks"].median(),

        "median_total_streams": df["total"].median(),
    }

    return kpis 


# TOP SONGS

def get_top_songs(
    df,
    metric="total",
    n=10
):

    """Return top songs based on selected metric."""

    columns= [
        "artist name",
        "song name",
        "wks",
        "t10",
        "pk",
        "PkStreams",
        "total"
    ]

    available_columns = [
        col for col in columns
        if col in df.columns 
    ]

    return (
        df[available_columns]
        .sort_values(metric, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# LONGEVITY SEGMENTS

def create_longevity_segments(df):
    """Create chart longevity categories."""

    df = df.copy()

    bins = [
        0,
        4,
        12,
        26,
        52,
        float("inf")
    ]

    labels = [
        "Short",
        "Moderate",
        "Long",
        "Very Long",
        "Exceptional"
    ]

    df["longevity_group"] = pd.cut(
        df["wks"],
        bins=bins,
        labels=labels,
        include_lowest=True 
    )

    return df