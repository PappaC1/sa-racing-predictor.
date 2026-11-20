def train_and_predict(
    history: pd.DataFrame,
    today: pd.DataFrame,
    model_type: str,
    value_edge_threshold: float = 0.03,
    min_value_bet_odds: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    ...
