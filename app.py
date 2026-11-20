import io
import json

import pandas as pd
import streamlit as st

from racing_model import load_csv, build_features, train_and_predict, choose_selections

st.set_page_config(
    page_title="SA Racing Predictor",
    page_icon="🏇",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🏇 SA Racing Predictor")
st.caption("Probable Winner • Value Bet • Danger Horse")

HISTORICAL_FILE = "data/sa_historical_results.csv"
TODAY_FILE = "data/todays_runners.csv"
VALUE_EDGE_THRESHOLD = 0.03
MIN_VALUE_BET_ODDS = 2.0

@st.cache_data(show_spinner="Loading racing data...")
def load_data():
    history = load_csv(HISTORICAL_FILE)
    today = load_csv(TODAY_FILE)
    return history, today

@st.cache_resource(show_spinner="Training model...")
def get_predictions(history, today):
    history = build_features(history)
    today = build_features(today)
    predictions, metrics = train_and_predict(
        history=history,
        today=today,
        model_type="lightgbm",
        value_edge_threshold=VALUE_EDGE_THRESHOLD,
        min_value_bet_odds=MIN_VALUE_BET_ODDS,
    )
    selections = choose_selections(predictions)
    return predictions, selections, metrics

history, today = load_data()
predictions, selections, metrics = get_predictions(history, today)

selection_df = pd.DataFrame([s.__dict__ for s in selections])

with st.sidebar:
    st.header("Filters")
    courses = ["All courses"] + sorted(selection_df["course"].unique().tolist())
    selected_course = st.selectbox("Course", courses)

    race_ids = selection_df["race_id"].unique().tolist()
    selected_race = st.selectbox("Race", race_ids)

    st.divider()
    st.subheader("Model health")
    st.metric("Historical races", f"{metrics['historical_rows']:,}")
    st.metric("Log loss", f"{metrics['out_of_fold_log_loss']:.4f}")
    st.metric("Brier score", f"{metrics['out_of_fold_brier_score']:.4f}")

if selected_course != "All courses":
    selection_df = selection_df[selection_df["course"] == selected_course]

race_selections = selection_df[selection_df["race_id"] == selected_race].copy()

if race_selections.empty:
    st.warning("No selections available for this race.")
    st.stop()

race_info = predictions[predictions["race_id"] == selected_race].iloc[0]

st.subheader(
    f"Race {int(race_info['race_number'])} — {race_info['course']}"
)
st.write(
    f"{int(race_info['distance_m'])}m • {race_info['going']} • "
    f"{race_info['race_type']}"
)

best = race_selections[race_selections["is_best_selection"]].iloc[0]

st.success(
    f"**★ Best selection: {best['horse_name']}**\n\n"
    f"**{best['selection_type']}** • "
    f"Model win probability: **{best['model_win_probability']:.1%}** • "
    f"Odds: **{best['market_odds']:.2f}**"
)

st.divider()

for _, row in race_selections.iterrows():
    highlight = "🥇" if row["is_best_selection"] else "•"

    with st.container(border=True):
        st.markdown(
            f"### {highlight} {row['selection_type']}\n"
            f"**{row['horse_name']}**\n\n"
            f"Win probability: **{row['model_win_probability']:.1%}**  \n"
            f"Market odds: **{row['market_odds']:.2f}**  \n"
            f"Model edge: **{row['model_edge']:+.1%}**"
        )

st.divider()

st.subheader("Full race analysis")

display_columns = [
    "horse_name", "trainer_name", "jockey_name", "draw", "market_odds",
    "model_win_probability", "model_edge", "market_rank"
]

race_runners = predictions[
    predictions["race_id"] == selected_race
][display_columns].sort_values(
    "model_win_probability", ascending=False
)

st.dataframe(
    race_runners,
    use_container_width=True,
    hide_index=True,
    column_config={
        "model_win_probability": st.column_config.ProgressColumn(
            "Win probability",
            min_value=0.0,
            max_value=1.0,
            format="%.1f%%",
        ),
        "model_edge": st.column_config.NumberColumn(
            "Edge",
            format="%+.1f%%",
        ),
    },
)

csv_buffer = io.StringIO()
race_selections.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download selections",
    data=csv_buffer.getvalue(),
    file_name=f"{selected_race}_selections.csv",
    mime="text/csv",
)

st.info(
    "Use this as research support, not a guaranteed tipping service. "
    "Horse racing outcomes remain highly uncertain."
)
