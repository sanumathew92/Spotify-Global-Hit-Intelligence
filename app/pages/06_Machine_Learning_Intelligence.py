from pathlib import Path 
import pandas as pd 
import plotly.express as px 
import streamlit as st 

# PATHS 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
ML_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "machine_learning"
COMPARISON_DIR = TABLE_DIR / "model_comparison"
SHAP_DIR = TABLE_DIR / "model_comparison"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "machine_learning"
MODEL_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "model_comparison"
TOP10_THRESHOLD_RESULTS_PATH = (ML_TABLE_DIR / 
    "top10_threshold_test_results.csv")

TOP10_THRESHOLD_VALIDATION_PATH = (
    ML_TABLE_DIR / "top10_threshold_validation_results.csv"
)

TOP10_THRESHOLD_CM_PATH = (
    ML_TABLE_DIR / "top10_threshold_confusion_matrix.csv"
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Machine Learning Intelligence",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Intelligence")
st.markdown(
    """
    Explore model performance, cross-valdiation results and test predictions for Spotify chart success.

    **Prediction tasks:** Top-10 achievement, high-stream classification and chart longevity.
    """
)

# DATA LOADER 

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except(OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()

def show_csv_table(path, title):
    data = load_csv(str(path))

    st.subheader(title)

    if data.empty:
        st.info(f"No readable data found at: {path_name}")
        return 

    st.caption(f"{len(data):,} rows . {len(data.columns)} columns")
    st.dataframe(data, use_container_width=True, hide_index=True)

def show_saved_figure(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def find_column(df, candidates):
    """Find a column using case-insensitive matching."""
    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]

    return None 

def show_numeric_comparison(df, title):
    """Plot a selectable numeric metric when suitable columns exist."""
    if df.empty:
        return 

    model_col = find_column(
        df,
        ["model","model_name","classifier", "algorithm"]
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if model_col is None or not numeric_cols:
        st.caption(
            "A metric chart is unavailable because a model-name"
            "column or numeric metric columns were not detected."
        )
        return 


    metric = st.selectbox(
        "Choose evaluation metric",
        options=numeric_cols,
        key=f"metric_{title}"
    )

    chart_data = df[[model_col, metric]].copy()
    chart_data[metric] = pd.to_numeric(
        chart_data[metric], errors="coerce"
    )
    chart_data = chart_data.dropna(subset=[metric])

    if chart_data.empty:
        st.info("No valid values are available for this metric.")
        return 

    chart_data[model_col] = chart_data[model_col].astype(str)

    fig = px.bar(
        chart_data,
        x=model_col,
        y=metric,
        color=model_col,
        title=f"{title}: {metric}",
        text_auto=".3f"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Model",
        yaxis_title=metric,
        xaxis_tickangle=-25
    )

    st.plotly_chart(fig, use_container_width=True)


# LOAD MODEL RESULTS

model_results = load_csv(
    str(ML_TABLE_DIR / "model_results.csv")
)

cv_results = load_csv(
    str(ML_TABLE_DIR / "cross_validation_results.csv")
)

complete_comparison = load_csv(
    str(COMPARISON_DIR / "complete_model_comparison.csv")
)


task_comparison = load_csv(
    str(COMPARISON_DIR / "model_comparison_all_tasks.csv")
)

best_models = load_csv(
    str(COMPARISON_DIR / "best_f1_model_by_task.csv")
)


# OVERVIEW

st.header("1. Model Evaluation Overview")

if not model_results.empty:
    st.success(
        f"Loaded saved model results: "
        f"{len(model_results):,} rows and "
        f"{len(model_results.columns)} columns."
    )

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True 
    )

else:

    st.warning(
        "The model_results.csv file is missing or empty, "
        "Check the outputs/tables/machine_learning folder."
    )


# MODEL COMPARISON

st.divider()
st.header("2. Model Comparison")

comparison_options = {
    "Complete model comparison": complete_comparison,
    "Comparison across prediction tasks": task_comparison,
    "Best model by F1 score": best_models,
}

available_comparisons = [
    name for name, data in comparison_options.items()
    if not data.empty
]

if available_comparisons:
    selected_comparison = st.selectbox(
        "Select comparison dataset",
        available_comparisons
    )

    selected_data = comparison_options[selected_comparison]

    st.dataframe(
        selected_data,
        use_container_width=True,
        hide_index=True
    )


    show_numeric_comparison(
        selected_data,
        selected_comparison
    )
else:
    st.info("No model comparison tables were found.")


# MODEL PERFORMANCE FIGURES

st.divider()
st.header("3. Saved Performance Visualizations")

figure_options = {
    "Top-10 confusion matrix":
        FIGURE_DIR / "topic_confusion_matrix.png",
    "High-stream confusion matrix":
        FIGURE_DIR / "high_stream_confusion_matrix.png",
    "Longevity confusion matrix":
        FIGURE_DIR / "longevity_confusion_matrix.png",
    "Top-10 ROC curves":
        FIGURE_DIR / "top10_roc_curves.png",
    "High-stream ROC curves":
        FIGURE_DIR / "high_streams_roc_curve.png",
    "Longevity ROC curves":
        FIGURE_DIR / "longevity_roc_curves.png",
    "Top-10 precision-recall curves":
        FIGURE_DIR / "top10_precision_recall_curves.png",
    "Model comparison by F1":
        MODEL_FIGURE_DIR / "model_comparison_f1.png",
    "Model comparison by PR-AUC":
        MODEL_FIGURE_DIR / "model_comparison_pr_auc.png",
    "Model comparison by ROC-AUC":
        MODEL_FIGURE_DIR / "model_comparison_roc_auc.png",
    "Model training time":
        MODEL_FIGURE_DIR / "model_training_time.png"
}


available_figures = [
    name for name, path in figure_options.items()
    if path.exists()
]


if available_figures:
    selected_figure = st.selectbox(
        "Choose a saved performance visualization",
        available_figures
    )

    show_saved_figure(
        figure_options[selected_figure],
        selected_figure
    )

else:
    st.info("No saved model performance figures were found.")


# TOP-10 VALIDATION-SELECTED MODEL

st.divider()

st.header("🎯 Top-10 Classification - Final Evaluation")

st.markdown(
    """
    This section presents the model selected using validation F1-score optimization. The classification threshold is 
    selected on validation data and then applied to the final test probabilities.
    """
)

if TOP10_THRESHOLD_RESULTS_PATH.exists():

    top10_results = pd.read_csv(TOP10_THRESHOLD_RESULTS_PATH)

    if not top10_results.empty:

        result = top10_results.iloc[0]

        selected_model = str(result["Selected_Model"])
        selected_threshold = float(result["Selected_Threshold"])
        precision= float(result["Test_Precision"])
        recall = float(result["Test_Recall"])
        f1 = float(result["Test_F1"])

        # Model and threshold
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Selected Model", selected_model)

        with col2:
            st.metric("Selected Threshold",
            f"{selected_threshold:.2f}")

        # Performance metrics
        metric1, metric2, metric3 = st.columns(3)


        with metric1:
            st.metric("Test Precision", f"{precision:.2%}")

        with metric2:
            st.metric("Test Recall", f"{recall:.2%}")

        with metric3:
            st.metric("Test F1-score", f"{f1:.4f}")

        st.caption(
            "Metrics are from the saved final test evaluation. "
            "Precision and recall are shown as percentages."
        )

    else:
        st.warning("The Top-10 test results file is empty.")

else:
    st.warning(
        "Top-10 threshold results were not found. "
        "Run Notebook 09's final evaluation and save-results cells first."
    )


# TOP-10 CONFUSION MATRIX 

st.subheader("📊 Top-10 Confusion Matrix")

if TOP10_THRESHOLD_CM_PATH.exists():

    cm_df = pd.read_csv(
        TOP10_THRESHOLD_CM_PATH,
        index_col=0
    )

    st.dataframe(
        cm_df,
        use_container_width=True 
    )

    
    st.caption(
        "Rows represent actual classes; columns represent predicted classes. "
       
    )

else:
    st.warning(
        "Confusion matrix CSV not found. "
        "Check the output folder and filename."
    )

# VALIDATION THRESHOLD COMPARISON

st.subheader("🔎 Validation Threshold Comparison")

if TOP10_THRESHOLD_VALIDATION_PATH.exists():

    validation_df = pd.read_csv(
        TOP10_THRESHOLD_VALIDATION_PATH
    )

    if not validation_df.empty:

        validation_df = validation_df.sort_values(
            "Validation_F1",
            ascending=False
        )

        st.dataframe(
            validation_df.round(4),
            use_container_width=True,
            hide_index=True 
        )

        st.caption(
            "The validation table compares candidate models and their selected thresholds. The final test metrics"
            "above are reported separately"
        )

else:
    st.info(
        "Run Notebook 09 to generate the validation threshold comparison."
    )


# CROSS-VALIDATION

st.divider()
st.header("4. Cross-Validation Results")

if not cv_results.empty:
    st.markdown(
        """
        Cross-validation helps assess how consistently a model performs across different data splits. Compare the 
        mean scores with their variation where those values are available.
        """
    )

    st.dataframe(
        cv_results,
        use_container_width=True,
        hide_index=True
    )

    show_numeric_comparison(
        cv_results,
        "Cross-valdiation"
    )
else:

    st.info("No cross-validation results were found.")

# TEST PREDICTIONS

st.divider()
st.header("5. Test Set Predictions")

prediction_files = {
    "Top-10 predictions":
        ML_TABLE_DIR / "top_10_test_predictions.csv",
    "High-stream predictions":
        ML_TABLE_DIR / "high_stream_test_predictions.csv",
    "Longevity predictions":
        ML_TABLE_DIR / "longevity_test_predictions.csv",
}

available_prediction_files = [
    name for name, path in prediction_files.items()
    if path.exists()
]

if available_prediction_files:
    selected_prediction = st.selectbox(
        "Select prediction task",
        available_prediction_files
    )

    predictions = load_csv(
        str(prediction_files[selected_prediction])
    )

    if not predictions.empty:
        st.caption(
            f"{len(predictions):,} prediction rows available."
        )

        st.dataframe(
            predictions.head(500),
            use_container_width=True,
            hide_index=True
        )

        if len(predictions) > 500:
            st.caption(
                "Showing the first 500 rows for display performance."
            )

        numeric_prediction_cols = predictions.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_prediction_cols) >= 2:
            with st.expander("Explore prediction columns"):
                x_col = st.selectbox(
                    "X-axis",
                    numeric_prediction_cols,
                    key="prediction_x"
                )

                y_col = st.selectbox(
                    "Y-axis",
                    [
                        col for col in numeric_prediction_cols
                        if col != x_col
                    ],
                    key= "prediction_y"
                )


                fig = px.scatter(
                    predictions.head(3000),
                    x=x_col,
                    y=y_col,
                    title="Relationship between prediction variables",
                    opacity=0.6
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

else:
    st.info("No test prediction files were found.")


# METHODOLOGY AND LIMITATIONS

st.divider()
st.header("6. Interpretation and Limitations")

st.markdown(
    """

    **Classification metrics:** Accuracy alone may be misleading when the target classes are imbalanced. 
    Review precision, recall, F1-score, ROC-AUC and PR-AUC where available. 
    **Cross-valdiation:** Compare mean scores and variability, not only a single train-test split.
    **Prediction vs. explanation:** Predictive performances does not establish that a feature causes a song 
    to succeed.
    **Data leakage:** Confirm that every feature would genuinely be available at prediction must not be used to predict
    that same performance.
    **Scope:** These saved models analyze historical chart outcomes, They should not be interpreted as proof that lyrics 
    alone cause streaming or chart success.
    """
)

st.caption(
    "Spotify Global Hit Intelligence . Machine Learning Module"
)
