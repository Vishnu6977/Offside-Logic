import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
NUM_ESTIMATORS = 400
MIN_SAMPLES_SPLIT = 30

def load_data(file_path: str) -> pd.DataFrame:
    """Load the matches.csv file."""
    matches = pd.read_csv(file_path)
    return matches

def preprocess_data(matches: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    matches["Date"] = pd.to_datetime(matches["Date"], errors='coerce')
    matches.dropna(subset=["Date"], inplace=True)
    matches["Venue"] = matches["Venue"].astype("category").cat.codes
    matches["Opponent"] = matches["Opponent"].astype("category")
    matches["Opponent_code"] = matches["Opponent"].cat.codes
    matches["Hour"] = pd.to_datetime(matches["Time"], format='%H:%M', errors='coerce').dt.hour.fillna(0).astype(int)
    matches["day_code"] = matches["Date"].dt.dayofweek
    matches["target"] = matches["Result"].apply(lambda x: 2 if x == 'W' else (1 if x == 'D' else 0)).astype(int)
    return matches

def create_rolling_features(matches: pd.DataFrame, feature_columns: list[str], rolling_window: int = 5) -> pd.DataFrame:
    rolling_features = []
    
    for team in matches["Team"].unique():
        team_matches = matches[matches["Team"] == team].sort_values("Date")
        team_rolling = team_matches[feature_columns].rolling(rolling_window, closed='left').mean()
        team_rolling["Team"] = team
        team_rolling["Date"] = team_matches["Date"].values
        rolling_features.append(team_rolling)
    rolling_data = pd.concat(rolling_features, ignore_index=True)
    rolling_data = rolling_data.dropna(subset=feature_columns)
    matches = matches.merge(rolling_data, on=["Team", "Date"], suffixes=("", "_rolling"))
    return matches

def train_random_forest_model(data: pd.DataFrame, predictors: list[str], target_column: str = "target") -> pd.DataFrame:
    """Train a random forest model."""
    train = data[data["Date"] < '2024-03-18']
    test = data[data["Date"] >= '2024-03-18']
    rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, min_samples_split=MIN_SAMPLES_SPLIT, random_state=1)
    rf.fit(train[predictors], train[target_column])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame({"actual": test[target_column], "predicted": preds}, index=test.index)
    return combined

def evaluate_model(combined: pd.DataFrame) -> tuple[float, float, float, float]:
    """Evaluate the model using various metrics."""
    accuracy = accuracy_score(combined["actual"], combined["predicted"])
    return accuracy

def main():
    file_path = "matches.csv"
    matches = load_data(file_path)
    matches = preprocess_data(matches)
    feature_columns = ["xG", "xGA", "Poss", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
    matches = create_rolling_features(matches, feature_columns)
    rolling_feature_columns = [f"{c}_rolling" for c in feature_columns]
    predictors = feature_columns + rolling_feature_columns
    combined = train_random_forest_model(matches, predictors)
    accuracy, precision, recall, f1 = evaluate_model(combined)
    print("Accuracy:", accuracy)
    print("Micro Precision:", precision)
    print("Micro Recall:", recall)
    print("Micro F1 Score:", f1)

if __name__ == "__main__":
    main()
