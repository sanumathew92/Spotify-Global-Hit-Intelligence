import streamlit as st 
import plotly.express as px 
import pandas as pd 

from utils import (
    load_clean_data,
    format_number,
    format_weeks
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Artist Intelligence",
    page_icon="👤",
    layout="wide"
)

# LOAD DATA 

df = load_clean_data()

# PAGE HEADER 

st.title("👤 Artist Intelligence")

st.markdown(
    """
    Analyze artist-level chart performance, catalog depth, longevity, Top-10 consistency
    and streaming performance.

    **Important:** Artist performance is evaluated across multiple dimensions rather than using
    a single overall score.
    """
)


# CREATE ARTIST-LEVEL DATASET

artist_performance = (
    df.groupby("artist name")
    .agg(
        charting_songs=("song name", "nunique"),
        total_chart_streams=("total", "sum"),
        mean_total_streams=("total", "sum"),
        median_total_streams=("total", "median"),
        mean_weeks= ("wks", "mean"),
        median_weeks=("wks", "median"),
        max_weeks=("wks", "max"),
        mean_peak_position=("pk","mean"),
        median_peak_position=("pk","median"),
        best_peak_position=("pk","min"),
        total_top10_weeks=("t10","sum"),
        mean_top10_weeks=("t10","mean"),
        top10_songs=("t10", lambda x: (x > 0).sum()),
        mean_peak_streams=("PkStreams","mean"),
        median_peak_streams=("PkStreams","median"),
        max_peak_streams=("PkStreams","max")
    )
    .reset_index()
)

# DERIVED ARTIST METRICS 

artist_performance["top10_rate"] = (
    artist_performance["top10_songs"]
    / artist_performance["charting_songs"]
)

artist_performance["top10_rate_pct"] = (
    artist_performance["top10_rate"] * 100
)

artist_performance["mean_median_ratio"] = (
    artist_performance["mean_total_streams"]
    / artist_performance["median_total_streams"]
)

artist_performance["stream_consistency_cv"] = (
    artist_performance["mean_total_streams"]
    / artist_performance["median_total_streams"]
)

# SIDEBAR 

st.sidebar.header("🔎 Artist Filters")


# Minimum catalog size 

min_catalog = st.sidebar.slider(
    "Minimum Charting Songs",
    min_value=1,
    max_value=int(
        artist_performance["charting_songs"].max()
    ),
    value=5 
)

filtered_artists = artist_performance[
    artist_performance["charting_songs"] >= min_catalog
].copy()

# Artist selector 

artist_options = sorted(
    filtered_artists["artist name"]
    .dropna()
    .astype(str)
    .tolist()
)


selected_artist = st.sidebar.selectbox(
    "Select Artist",
    ["All Artists"] + artist_options
)

# ARTIST OVERVIEW 

st.divider()

st.subheader("📊 Artist Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Artists",
        f"{len(filtered_artists):,}"
    )

with col2:

    st.metric(
        "Total Charing Songs",
        f"{filtered_artists['charting_songs'].sum():,}"
    )

with col3:

    st.metric(
        "Artists with Top-10 Songs",
        f"{(filtered_artists['top10_songs'] > 0).sum():,}"
    )

# SELECTED ARTIST PROFILE 


if selected_artist != "All Artists":

    artist_row = artist_performance[
        artist_performance["artist name"] == selected_artist
    ].iloc[0]

    st.divider()

    st.header(f"👤{selected_artist}")


    # Primary metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Charting Songs",
            f"{int(artist_row['charting_songs']):,}"
        )

    with col2:

        st.metric(
            "Median Streams",
            format_number(
                artist_row["median_total_streams"]
            )
        )


    with col3:

        st.metric(
            "Median Chart Weeks",
            f"{artist_row['median_weeks']:.1f}"
        )


    with col4:

        st.metric(
            "Top-10 Rate",
            f"{artist_row['top10_rate_pct']:.1f}%"
        )


    # Secondary metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Best Peak Position",
            f"#{int(artist_row['best_peak_position'])}"
        )


    with col2:

        st.metric(
            "Longest Chart Run",
            f"{int(artist_row["best_peak_position"])}"
        )

    with col3:

        st.metric(
            "Total Top-10 Weeks",
            f"{int(artist_row['total_top10_weeks']):,}"
        )

    with col4:

        st.metric(
            "Max Peak Streams",
            format_number(
                artist_row["max_peak_streams"]
            )
        )


    # ARTIST SONG CATALOG

    st.divider()

    st.subheader("🎵 Artist Song Catalog")

    artist_songs= df[
        df["artist name"] == selected_artist
    ].copy()

    artist_songs = artist_songs.sort_values(
        "total",
        ascending=False
    )

    catalog_display = artist_songs[
        [
            "song name",
            "wks",
            "t10",
            "pk",
            "PkStreams",
            "total"
        ]
    ].copy()


    catalog_display["total"] = (
        catalog_display["total"]
        .apply(format_number)
    )


    catalog_display["PkStreams"] = (
        catalog_display["PkStreams"]
        .apply(format_number)
    )

    catalog_display.columns = [
        "Song",
        "Chart Weeks",
        "Top-10 Weeks",
        "Peak Position",
        "Peak Streams",
        "Total Streams"
    ]

    st.dataframe(
        catalog_display,
        use_container_width=True,
        hide_index=True 
    )

    # ARTIST SONG PERFORMANCE 

    st.subheader("📈 Song Performance")

    fig = px.scatter(
        artist_songs,
        x="wks",
        y="total",
        size="PkStreams",
        hover_name="song name",
        hover_data=[
            "t10",
            "pk", 
        ],
        log_y=True,
        title=f"{selected_artist} - Chart Longevity vs Total Streams",
        labels={
            "wks": "Chart Weeks",
            "total": "Total Streams"
        }
    )

    fig.update_layout(
        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # TOP SONGS BY STREAMS

    st.subheader("🔥 Top Songs by Total Streams")

    top_artist_songs = (
        artist_songs
        .sort_values("total", ascending=False)
        .head(10)
    )


    fig = px.bar(
        top_artist_songs.sort_values("total"),
        x="total",
        y="song name",
        orientation="h",
        title=f"Top Songs - {selected_artist}",
        labels = {
            "total": "Total Streams",
            "song name": "Song"
        }
    )


    fig.update_layout(
        height=500,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )

# ALL-ARTIST ANALYSIS 

else:

    st.divider()

    st.subheader("🏆 Artist Catalog Leaders")


    catalog_leaders = (
        filtered_artists
        .sort_values(
            "charting_songs",
            ascending=False
        )
        .head(15)
    )


    fig = px.bar(
        catalog_leaders.sort_values(
            "charting_songs"
        ),
        x="charting_songs",
        y="artist name",
        orientation="h",
        title="Artists with the Most Charting Songs",
        labels={
            "charting_songs": "Number of Charting Songs",
            "artist name": "Artist"
        }
    )


    fig.update_layout(
        height=600,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # MEDIAN PERFORMANCE

    st.subheader("💎 Artists by Median Song Streams")

    median_leaders = (
        filtered_artists
        .sort_values(
            "median_total_streams",
            ascending=False
        )
        .head(15)
    )


    fig = px.bar(
        median_leaders.sort_values(
            "median_total_streams"
        ),
        x="median_total_streams",
        y="artist name",
        orientation="h",
        title="Highest Median Song Streams",
        labels={
            "median_total_streams": "Median Total Streams",
            "artist name": "Artist"
        }
    )

    fig.update_layout(
        height=600,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # TOP-10 CONSISTENCY 

    st.subheader("🏆 Top-10 Consistency")


    top10_leaders = (
        filtered_artists[
            filtered_artists["charting_songs"] >= 5
        ]
        .sort_values(
            "top10_rate",
            ascending=False
        )
        .head(15)
    )

    fig.update_xaxes(
        tickformat=".0%"
    )

    fig.update_layout(
        height=600,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True 
    )


    # CATALOG SIZE VS MEDIAN PERFORMANCE 

    st.subheader("📊 Catalog Size vs Median Performance")

    fig = px.scatter(
        filtered_artists,
        x="charting_songs",
        y="median_total_streams",
        size="top10_songs",
        hover_name="artist name",
        log_y=True,
        title="Catalog Size vs Median Song Streams",
        labels={
            "charting_songs": "Number of Charting Songs",
            "median_total_streams": "Median Total Streams",
            "top10_songs": "Top-10 Songs"
        }
    )

    fig.update_layout(
        height=550
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ARTIST PERFORMANCE TABLE

    st.divider()

    st.subheader("📋 Artist Performance Table")

    table = filtered_artists[
        [
            "artist name",
            "charting_songs",
            "median_total_streams",
            "median_weeks",
            "best_peak_position",
            "top10_songs",
            "top10_rate_pct"
        ]
    ].copy()

    table = table.sort_values(
        "median_total_streams",
        ascending=False
    )

    table["median_total_streams"] = (
        table["median_total_streams"]
        .apply(format_number)
    )

    table["median_weeks"] = (
        table["median_weeks"]
        .round(1)
    )

    table["best_peak_position"] = (
        table["best_peak_position"]
        .astype(int)
    )

    table["top10_rate_pct"] = (
        table["top10_rate_pct"]
        .round(1)
    )

    table.columns = [
        "Artist",
        "Charting Songs",
        "Median Streams",
        "Median Weeks",
        "Best Peak",
        "Top-10 Songs",
        "Top-10 Rate (%)"
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True 
    )



    # METHODOLOGY NOTE 

    st.divider()

    st.info(
    """

    **Interpretation note**

    Artist catalog size and total catalog streams are influenced by the number of songs an artist has charted. Median Song
    performance, Top-10 rate and longevity provide additional measures of consistency.

    These metrics describe historical chart performance and should not be interpreted as causal measures of artist quality.
    """
    )
    # FOOTER

    st.divider()

    st.caption(
        "Spotify Global Hit Intelligence | Artist Intelligence"
    )

