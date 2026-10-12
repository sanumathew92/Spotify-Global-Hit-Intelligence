"""
Hit Segmentation Dashboard
1. Cluster Overview
Number of songs per segment
Percentage of dataset
Median chart weeks
Median total streams
Median peak position
Median peak streams
Top-10 performance

2. Segment comparison
Cluster performance bar charts
Longevity comparison
Total-stream comparison
Peak-position comparison

3. PCA Visualization
Interactive 2D PCA scatter
Cluster coloring
Song/artist hover information

4. Segment profiles
Detailed statistics for each cluster
Clear business interpretation

5. Segment song explorer
Select a cluster
Show its top songs
Sort by streams, longevity or peak performance

6. Important Interpretation
We'll keep the language descriptive:
"short-lived"
"moderate"
"exceptional/high-performing"

rather than claiming one cluster is inherently "better".
"""

import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 

from utils import (
    load_clean_data,
    format_number
)


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Hit Segmentation",
    page_icon="🎯",
    layout="wide"
)

# PROJECT PATH 

from pathlib import Path 

PROJECT_ROOT = Path(
    r"C:\Users\Sanu\OneDrive\Desktop\Spotify-Global-Hit-Intelligence"
)

SEGMENTATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spotify_hit_segments.csv"
)


# PAGE HEADER 

st.title("🎯 Hit Segmentation")

st.markdown(
    """
    Discover distinct song-performance segments using unsupervised machine learning.

    The segmentation group songs according to their historical chart longevity, streaming performance adn chart-position
    characteristics.
    """
)

st.info(
    """
    **Interpretation note:** These segments are descriptive groups identified from historical chart-performance patterns.
    They should not be interpreted as causal categories or as a ranking of song quality.
    """
)

# LOAD DATA

@st.cache_data
def load_segmentation_data():

    if not SEGMENTATION_PATH.exists():

        return None 

    return pd.read_csv(
        SEGMENTATION_PATH,
        low_memory=False
    )

df = load_clean_data()
segmentation_df = load_segmentation_data()

# FALLBACK 

if segmentation_df is None:

    st.error(
        f"""
        Segmentation file was not found.

        Expected file:

        `{SEGMENTATION_PATH}`

        Please save the final Notebook 08 segmentation dataset using this filename before running Page 5.
        """
    )

    st.stop()


# PREPARE DATA 

segmentation_df.columns = (
    segmentation_df.columns
    .str.strip()
)


# Detect cluster column

cluster_candidates = [
    "cluster",
    "Cluster",
    "cluster_label",
    "Cluster_Label",
    "segment",
    "segment_label"
]

cluster_column = next(
    (
        col 
        for col in cluster_candidates
        if col in segmentation_df.columns
    ),
    None
)


if cluster_column is None:

    st.error(
        """
        Could not identify the cluster column.

        Expected one of:

        `cluster`, `Cluster`, `cluster_label`, 
        `Cluster_Label`, `segment`, `segment_label`
        """
    )


    st.stop()

# Standardize cluster column
segmentation_df["cluster"] = segmentation_df[
    cluster_column
]


# MERGE WITH CLEAN DATA 

if "record_id" not in df.columns:
    df["record_id"] = np.arange(len(df))

if "record_id" in segmentation_df.columns:

    # Use the detected column name rather than assuming "cluster".
    cluster_lookup = segmentation_df[
        ["record_id", "cluster_column"]
    ].copy()

    cluster_lookup = cluster_lookup.rename(
        columns={cluster_column: "cluster"}
    )

    segmentation_data = df.merge(
        cluster_lookup,
        on="record_id",
        how="inner",
        validate="one_to_one"
    )

elif len(segmentation_df) == len(df):

    # Align by row order when record_id is unavailable.
    segmentation_data = df.copy()

    segmentation_data["cluster"] = (
        segmentation_df[cluster_column].to_numpy()
    )

else:

    st.error(
        """
        The segmentation output cannot be aligned safely with the cleaned dataset.

        It must contain a record_id column or have the same number of rows in the original row order.
        """
    )

    st.stop()


# CREATE SEGMENT LABELS

cluster_labels = {
    0: "Moderate Longevity / Mid-Level Performance",
    1: "Short-Lived / Low Chart Penetration",
    2: "Exceptional Longevity / High Performance"
}


segmentation_data["segment"] = (
    segmentation_data["cluster"]
    .map(cluster_labels)
    .fillna(
        "Segment "
        + segmentation_data["cluster"].astype(str)
    )
)


# ADD PCA FEATURES IF AVAILABLE

# Normalize column names for PCA detection.
pca_name_map = {
    str(col).strip().lower(): col 
    for col in segmentation_df.columns 
}

pc1_source = next(
    (
        pca_name_map[name]
        for name in ["pc1", "pca1", "pca_component_1"]
        if name in pca_name_map
    ),
    None
)

pc2_source = next(
    (
        pca_name_map[name]
        for name in ["pc2", "pca2", "pca_component_2"]
        if name in pca_name_map
    ),
    None 
)

pca_columns = []

if pc1_source is not None and pc2_source is not None:

    # Avoid merging duplicate PCA column names.
    if(
        "record_id" in segmentation_df.columns
        and "record_id" in segmentation_data.columns
    ):

        pca_lookup = segmentation_df[
            ["record_id", pc1_source. pc2_source]
        ].copy()

        pca_lookup = pca_lookup.rename(
            columns={
                pc1_source: "PC1",
                pc2_source: "PC2"
            }
        )

        # Keep only PCA coordinates not already in the main data.
        columns_to_add = [
            col for col in ["PC1", "PC2"]
            if col not in segmentation_data.columns
        ]

        if columns_to_add:
            pca_lookup = pca_lookup[
                ["record_id"] + columns_to_add
            ]

            segmentation_data = segmentation_data.merge(
                pca_lookup,
                on="record_id",
                how="left",
                validate="one_to_one"
            )

            pca_columns= (
                ["PC1", "PC2"]
                if all(
                    col in segmentation_data.columns
                    for col in ["PC1", "PC2"]
                )
                else []
            )

        elif len(segmentation_df) == len(df):

            # Align by row order if record_id is unavailable
            segmentation_data["PC1"] = (
                segmentation_data["pc1_source"].to_numpy()
        )

            segmentation_data["PC2"] = (
                segmentation_df[pc2_source].to_numpy()
        )

            pca_columns = ["PC1", "PC2"]

# If PCA columns are absent or cannot be aligned.
# skip the PCA chart without interrupting the page.


# SIDEBAR 

st.sidebar.header("🔎 Segment Filters")

segment_options = [
    "All Segments"
] + sorted(
    segmentation_data["segment"]
    .dropna()
    .unique()
    .tolist()
)


selected_segment = st.sidebar.selectbox(
    "Select Segment",
    segment_options
)


if selected_segment != "All Segments":

    filtered_df = segmentation_data[
        segmentation_data["segment"] == selected_segment
    ].copy()

else:

    filtered_df = segmentation_data.copy()


# OVERVIEW KPIs

st.divider()

st.subheader("📊 Segmentation Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Songs",
        f"{len(segmentation_data):,}"
    )


with col2:

    st.metric(
        "Segments",
        f"{segmentation_data['cluster'].nunique():,}"
    )

with col3:

    st.metric(
        "Median Chart Weeks",
        f"{segmentation_data['wks'].median():.0f}"
    )

with col4:

    st.metric(
        "Median Total Streams",
        format_number(
            segmentation_data["total"].median()
        )
    )
    

# SEGMENT DISTRIBUTION

st.divider()

st.subheader("🎯 Segment Distribution")

segment_summary = (
    segmentation_data
    .groupby(
        ["cluster", "segment"],
        as_index = False
    )
    .agg(
        songs=("song name", "count"),
        median_weeks = ("wks", "median"),
        median_total_streams = ("total", "median"),
        median_peak_streams = ("PkStreams", "median"),
        median_peak_position = ("pk", "median"),
        median_top10_weeks = ("t10", "median")
    )
)

segment_summary["percentage"] = (
    segment_summary["songs"]
    / len(segmentation_data)
    * 100
)

with col1:

    fig = px.bar(
        segment_summary,
        x="segment",
        y="songs",
        text="songs",
        title="Songs per Segment",
        labels={
            "segment": "Segment",
            "songs": "Number of Songs"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-20
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.pie(
        segment_summary,
        names="segment",
        values="songs",
        title="Segment Share"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# SEGMENT PROFILE TABLE 

st.subheader("📋 Segment Profile")

profile_display = segment_summary.copy()

profile_display["percentage"] = (
    profile_display["percentage"]
    .round(2)
)

profile_display["median_total_streams"] = (
    profile_display["median_total_streams"]
    .apply(format_number)
)

profile_display["median_peak_streams"] = (
    profile_display["median_peak_streams"]
    .apply(format_number)
)

profile_display["median_weeks"] = (
    profile_display["median_weeks"]
    .round(1)
)

profile_display["median_peak_position"] = (
    profile_display["median_peak_position"]
    .round(1)
)


profile_display["median_top10_weeks"] = (
    profile_display["median_top10_weeks"]
    .round(1)
)

profile_display.columns = [
    "Cluster",
    "Segment",
    "Songs",
    "Median Weeks",
    "Median Streams",
    "Median Peak Streams",
    "Median Peak Position",
    "Median Top-10 Weeks",
    "Share (%)"
]

st.dataframe(
    profile_display,
    use_container_width=True,
    hide_index=True
)


# PERFORMANCE COMPARISON

st.divider()

st.subheader("📈 Segment Performance Comparison")


metric_choice = st.selectbox(
    "Select Performance Metric",
    [

        "Median Chart Weeks",
        "Median Total Streams",
        "Median Peak Streams",
        "Median Peak Posiiton",
        "Median Top-10 Weeks"
    ]
)

metric_mapping = {
    "Median Chart Weeks": "median_weeks",
    "Median Total Streams": "median_total_streams",
    "Median Peak Streams": "median_peak_streams",
    "Median Peak Position": "median_peak_position",
    "Median Top-10 Weeks": "median_top10_weeks"
}


selected_metric = metric_mapping[
    metric_choice
]

fig = px.bar(
    segment_summary,
    x="segment",
    y=selected_metric,
    text=selected_metric,
    title=metric_choice,
    labels={
        "segment": "Segment",
        selected_metric: metric_choice
    }
)

fig.update_traces(
    textposition="outside"
)


fig.update_layout(
    height=550,
    xaxis_tickangle=-20
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# LONGEVITY DISTRIBUTION

st.divider()

st.subheader("⏳ Chart Longevity by Segment")

fig = px.box(
    segmentation_data,
    x="segment",
    y="wks",
    color="segment",
    points=False,
    log_y=True,
    title="Chart Weeks Distribution by Segment",
    labels={
        "segment": "Segment",
        "wks": "Chart Weeks",
    }
)

fig.update_layout(
    height=550,
    showlegend=False,
    xaxis_tickangle=-20
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# STREAM PERFORMANCE

st.subheader("💰 Total Streams by Segment")


fig = px.box(
    segmentation_data,
    x="segment",
    y="total",
    color="segment",
    points=False,
    log_y=True,
    title="Total Streams Distribution by segment",
    labels={
        "segment": "Segment",
        "total": "Total Streams"
    }
)

fig.update_layout(
    height=550,
    showlegend=False,
    xaxis_tickangle=-20
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# PEAK POSITION

st.subheader("🏆 Peak Chart Position by Segment")

fig = px.box(
    segmentation_data,
    x="segment",
    y="pk",
    color="segment",
    points=False,
    title="Peak Position Distribution",
    labels={
        "segment": "Segment",
        "pk" : "Peak Position"
    }
)

fig.update_layout(
    height=500,
    showlegend=False,
    xaxis_tickangle=-20
)


st.plotly_chart(
    fig,
    use_container_width=True
)

# PCA VISUALIZATION

if len(pca_columns) >= 2:

    st.divider()

    st.subheader("🧬 PCA Segment Visualization")

    pc1 = pca_columns[0]
    pc2 = pca_columns[1]

    pca_plot = segmentation_data.dropna(
        subset=[pc1, pc2]
    ).copy()

    fig = px.scatter(
        pca_plot,
        x=pc1,
        y=pc2,
        color="segment",
        hover_name="song name",
        hover_data=[
            "artist name",
            "wks",
            "t10",
            "pk",
            "total"
        ],
        title="Song Segments in PCA Space",
        labels={
            pc1: "Principal Component 1",
            pc2: "Principal Component 2"
        }
    )

    fig.update_layout(
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "PC1 + PC2 explain approximately 81.61% of the variance "
        "in the segmentation feature space."
    )


else:

    st.info(
        """
        PCA coordinates were not found in the saved segmentation
        file. The remaining segment analysis is still available.
        """
    )


# SELECTED SEGMENT SONG EXPLORER

st.divider()

st.subheader("🎵 Segment Song Explorer")

sort_option = st.selectbox(
    "Sort Songs By",
    [
        "Total Streams",
        "Chart Weeks",
        "Peak Streams",
        "Peak Position"
    ]
)


sort_mapping = {
    "Total Streams": "total",
    "Chart Weeks": "wks",
    "Peak Streams": "PkStreams",
    "Peak Position": "pk"
}

sort_column = sort_mapping[
    sort_option
]

ascending = (
    sort_option == "Peak Position"
)

segment_songs = (
    filtered_df
    .sort_values(
        sort_column,
        ascending=ascending
    )
    .head(20)
    .copy()
)


song_display = segment_songs[
    [
        "artist name",
        "song name",
        "wks",
        "t10",
        "pk",
        "PkStreams",
        "total",
        "segment"
    ]
].copy()

song_display["PkStreams"] = (
    song_display["PkStreams"]
    .apply(format_number)
)

song_display["total"]= (
    song_display["total"]
    .apply(format_number)
)


song_display.columns= [
    "Artist",
    "Song",
    "Weeks",
    "Top-10 Weeks",
    "Peak",
    "Peak Streams",
    "Total Streams",
    "Segment"
]


st.dataframe(
    song_display,
    use_container_width=True,
    hide_index=True
)

# SEGMENT INTERPRETATION

st.divider()

st.subheader("💡 Segment Interpretation")


interpretation = {
    0: """
    **Moderate Longevity / Mid-Level Performance**

    Songs in this segment typically show meaningful chart longevity and moderate streaming 
    performance, but generally do not reach the extreme performance levels of the exceptional segment.
    """,

    1: """
    **Short-Lived / Low Chart Penetration**

    Songs in this segment generally have short chart runs,
    lower total streaming accumulation and weaker peak positions.
    """,

    2: """
    **Exceptional Longevity / High Performance**

    Songs in the segment show substantially longer chart runs, stronger streaming performance and 
    much stronger peak positions. This is the smallest segment and represents a relatively concentrated 
    group of highly persistent chart performance.
    """
}

if selected_segment != "All Segments":

    selected_cluster= (
        filtered_df["cluster"].iloc[0]
    )

    st.markdown(
        interpretation.get(
            int(selected_cluster),
            "Custom segment."
        )
    )

else:

    st.markdown(
        """

        ### Three broad performance patterns emerge

        **Short-lived songs** form the largest segment and tend to have limited chart penetration.

        **Moderate performers** form another large segment with substantially lonegvity and streaming accumulation.

        **Exceptional performers** form a much smaller group with unusually long chart runs, stronger streaming performance
        and stronger peak positions.

        These segments are descriptive profiles rather than a causal explanation of why songs become successful.
        """

    )

# METHODOLOGY 

st.divider()

st.info(
    """
    **Methodology**

    The segmentation was developed using unsupervised clustering in Notebook 08. K values from 2 through 8 were
    evaluated using inertia and silhouette score, with **3 clusters selected**.

    PCA was then used to visulaize the resulting segments in a two-dimensional space. The first two principal components
    explain approximately **81.61%** of the variance.

    Segment labels are descriptive interpretations of the observed cluster profiles and are not predictive or 
    causal classifications.
    """
)

# FOOTER

st.divider()

st.caption(
    "Spotify Global Hit Intelligence | Hit Segmentation"
)