"""
SHAP Explainability

1. Global Feature Importance - Show which feature contribute most to predicting Top-10 achievement, high-stream status and 
long-running songs.
2. Feature Impact Analysis - Explain whether feature values tend to puch model predictions higher or lower, using the saved 
SHAP results.
3. Feature Relationships - Explore saved dependence plots for lyrics word count, lexical diversity and sentiment 
4. Individual Song Explanation - Display a saved waterfall plot to explain how features contribute to an 
individual prediction.
"""

from pathlib import Path 

import pandas as pd 
import streamlit as st 

# PATHS 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "shap"
SHAP_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "shap"

# PAGE CONFIGURATION

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SHAP Explainability")
st.markdown(
    """

    Understand how machine learning models use song and lyric features to predict historical Spotify chart outcomes.

    SHAP explains model predictions; it does not establish that a feature causes a song to succeed.
    """
)

# DATA LOADING 

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()


    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def show_table(path, title):
    data = load_csv(str(path))

    st.subheader(title)


    if data.empty:
        st.info(f"No readable data available: {path_name}")
        return 

    st.caption(
        f"{len(data):,} rows . {len(data.columns)} columns"
    )
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True 
    )

def show_figure(path, caption):
    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True
        )
        return True 

    st.info("Figure not found: {path_name}")
    return False 


# AVAILABLE SHAP ASSETS

st.header("1. SHAP Overview")

importance_files = {
    "High Total Streams":
        "shap_feature_importance_high_total_streams.csv",
    "Long-Running Songs":
        "shap_feature_imprtance_long_running_songs.csv",
    "Top-10 Achievement":
        "shap_feature_importance_top10_reached.csv",
}


summary_figures = {
    "High Total Streams":
        "shap_summary_high_total_streams.png",
    "Long-Running Songs":
        "shap_summary_long_running_songs.png",
    "Top-10 Achievement":
        "shap_summary_top10_reached.png",
}

feature_group_files = {
    "General feature groups":
        "shap_feature_group_importance.csv",
    "Top-10 feature groups":
        "shap_fetaure_group_importance_t10.csv",
}


available_tasks = [
    task for task, filename in importance_files.items()
    if (SHAP_TABLE_DIR / filename).exists()
]


if available_tasks:
    selected_task = st.selectbox(
        "Select prediction task",
        available_tasks 
    )

    left, right = st.columns(2)

    with left:
        st.metric("Prediction task", selected_task)

    with right:
        importance_path = (
            SHAP_TABLE_DIR / importance_files[selected_task]
        )
        importance_data = load_csv(str(importance_path))

        st.metric(
            "Saved importance rows",
            f"{len(importance_data):,}"
        )

else:
    st.warning(
        "No task-specific SHAP importance tables were found. "
        "Check outputs/tables/shap."
    )

    selected_task = None 


# GLOBAL FEATURE IMPORTANCE 

st.divider()
st.header("2. Global Feature Importance")

if selected_task is not None:
    show_table(
        SHAP_TABLE_DIR / importance_files[selected_task],
        f"Feature Importance - {selected_task}"
    )

    figure_path = (
        SHAP_FIGURE_DIR / summary_figures[selected_task]
    )

    if figure_path.exists():
        show_figure(
            figure_path,
            f"SHAP Summary - {selected_task}"
        )

    else:
        st.info(
            "The matching summary plot is unavailable. see the dependence plots and save tables below."
        )


# FEATURE GROUP IMPORTANCE

st.divider()
st.header("3. Feature Group Importance")

available_groups = [
    label for label, filename in feature_group_files.items()
    if (SHAP_TABLE_DIR / filename).exists()
]

if available_groups:
    selected_group_file = st.selectbox(
        "Choose feature group table",
        available_groups
    )

    show_table(
        SHAP_TABLE_DIR / feature_group_files[selected_group_file],
        selected_group_file
    )

else:
    st.info("No feature group importance tables were found.")


# FEATURE DEPENDENCE

st.divider()
st.header("4. Feature Impact and Dependence")

dependence_figures = {
    "Lexical diversity":
        "shap_dependence_lexical_diversity.png",
    "Lyrics word count":
        "shap_dependence_lyrics_word_count.png",
    "Sentiment":
        "shap_dependence_sentiment.png"
}


available_dependence = [
    label for label, filename in dependence_figures.items()
    if (SHAP_FIGURE_DIR / filename).exists()
]

if available_dependence:
    selected_dependence = st.selectbox(
        "Select feature",
        available_dependence
    )

    show_figure(
        SHAP_FIGURE_DIR / dependence_figures[selected_dependence],
        f"SHAP Dependence - {selected_dependence}"
    )

else:
    st.info("No standard dependence plots were found.")


# INDIVIDUAL SONG EXPLANATIONS

st.divider()
st.header("5. Individual Song Explanation")

waterfall_figures = {
    "General individual-song explanation":
        "individual_song_shap_waterfall.png",
    "Long-running song prediction":
        "individual_song_shap_waterfall.png",
    "Top-10 prediction":
        "individual_song_waterfall_t10.png",
}

available_waterfalls = [
    label for label, filename in waterfall_figures.items()
    if (SHAP_FIGURE_DIR / filename).exists()  
]

if available_waterfalls:
    selected_waterfall = st.selectbox(
        "Choose individual explanation",
        available_waterfalls
    )

    show_figure(
        SHAP_FIGURE_DIR / waterfall_figures[selected_waterfall],
        selected_waterfall
    )

    st.caption(
        "A waterfall plot breaks down a single prediction into feature contributuins"
        "relative to the model's baseline."
    )

else:
    st.info("No saved individual-song waterfall plots were found.")


# INTERACTION ANALYSIS

st.divider()
st.header("6. Feature Interaction Analysis")

interaction_figures = {
    "General SHAP interactions":
        "shap_interaction_summary.png",
    "Top-10 SHAP interactions":
        "shap_interaction_summary_t10.png",
}

avialable_interactions = [
    label for label, filename in interaction_figures.items()
    if (SHAP_FIGURE_DIR / filename).exists()
]

if avialable_interactions:
    selected_interaction = st.selectbox(
        "Select interaction visualization",
        avialable_interactions
    )

    show_figure(
        SHAP_FIGURE_DIR / interaction_figures[selected_interaction],
        selected_interaction
    )

else:
    st.info("No Saved interaction plots were found.")



# SHAP VALUES

st.divider()
st.header("7. SHAP Values Explorer")


shap_value_files = {
    "High Total Streams":
        "shap_values_high_total_streams_test.csv",
    "Long-Running Songs":
        "shap_values_long_running_songs_test.csv",
    "Top-10 Achievement":
        "shap_values_top10_reached_test.csv"
}

available_shap_tables = [
    task for task, filename in shap_value_files.items()
    if (SHAP_TABLE_DIR / filename).exists()
]

if available_shap_tables:
    selected_shap_table = st.selectbox(
        "Select SHAP values dataset",
        available_shap_tables
    )

    shap_values = load_csv(
        str(SHAP_TABLE_DIR / shap_value_files[selected_shap_table])
    )

    if not shap_values.empty:
        st.caption(
            f"{len(shap_values):,} rows . "
            f"{len(shap_values.columns)} columns"
        )

        st.dataframe(
            shap_values.head(500),
            use_container_width=True,
            hide_index=True
        )

        if len(shap_values) > 500:
            st.caption(
                "Showing the first 500 rows for readability."

            )

else:
    st.info("No saved SHAP values tables were found.")


# INTERPRETATION AND LIMITATIONS

st.divider()
st.header("8. How to Interpret SHAP.")

st.markdown(
    """
    **Global importance:** Features with larger average absolute SHAP values generally have greater u=influence on 
    model outputs. **Positive and negative contributions:** Their meaning depends on the target, model output scale,
    and class being explained. **Dependence plots:** Reveal how feature value relate to SHAP contributions and may suggest 
    interactions.
    **Individual explanations:** Explain one prediction, not every song or the general population.
    **Causality:** SHAP explains a model's behaviour, it does not prove that changing a lyric feature would change chart
    success. **Data Leakage:** If chart outcome variables were included as predictors for related targets, explanations 
    can reflect leakage rather than useful advance prediction.
    """
)

st.caption(
    "Spotify Global Hit Intelligence - Explainable AI Module"
)