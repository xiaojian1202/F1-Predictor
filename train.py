import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# ... (Insert your data loading and preprocessing logic here to get train_df, train_labels) ...
DATA_PATH = 'data/'  # Folder where the Kaggle dataset was stored
START_YEAR = 2000    # Focus on the modern era for data consistency
# --- Data Loading ---
def load_data():
    print("Loading Vopani Dataset CSVs...")
    # Load core files with '\N' handling for nulls
    races = pd.read_csv(f'{DATA_PATH}races.csv', na_values='\\N')
    results = pd.read_csv(f'{DATA_PATH}results.csv', na_values='\\N')
    status = pd.read_csv(f'{DATA_PATH}status.csv', na_values='\\N')
    qualifying = pd.read_csv(f'{DATA_PATH}qualifying.csv', na_values='\\N')
    standings = pd.read_csv(f'{DATA_PATH}driver_standings.csv', na_values='\\N')
    circuits = pd.read_csv(f'{DATA_PATH}circuits.csv', na_values='\\N')
    lap_times = pd.read_csv(f'{DATA_PATH}lap_times.csv', na_values='\\N')
    weather = pd.read_csv(f'{DATA_PATH}weather.csv', na_values='\\N')
    print("Datasets are loaded in...")

    return races, results, status, qualifying, standings, circuits, lap_times, weather

# Load the raw dataframes
races_df, results_df, status_df, qual_df, stand_df, circuits_df, laps_df, weather_df = load_data()

# --- Cleaning & Weather Merging ---
def clean_and_merge_weather(races, weather):
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    weather['year'] = weather['datetime'].dt.year

    merged_races = pd.merge(
        races, 
        weather[['year', 'round', 'temperature', 'precipitation']], 
        on=['year', 'round'], 
        how='left'
    )
    
    merged_races['precipitation'] = merged_races['precipitation'].fillna(0)
    merged_races['temperature'] = merged_races['temperature'].fillna(20.0)

    drop_cols = ['url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time']
    cols_to_drop = [c for c in drop_cols if c in merged_races.columns]
    
    return merged_races.drop(columns=cols_to_drop)

main_races = clean_and_merge_weather(races_df, weather_df)

# --- Generate Ground Truth Labels (is_exciting) ---
def generate_labels(races, results, status, laps):
    target_race_ids = races['raceId'].unique()
    
    # A. Chaos Score (DNFs)
    results_filtered = results[results['raceId'].isin(target_race_ids)].copy()
    normal_finish_ids = [1] + list(range(11, 20))
    results_filtered['is_dnf'] = ~results_filtered['statusId'].isin(normal_finish_ids)
    chaos_df = results_filtered.groupby('raceId')['is_dnf'].sum().reset_index()
    chaos_df.rename(columns={'is_dnf': 'chaos_score'}, inplace=True)
    
    # B. Action Score (Overtakes)
    laps_filtered = laps[laps['raceId'].isin(target_race_ids)].copy()
    laps_filtered.sort_values(['raceId', 'driverId', 'lap'], inplace=True)
    laps_filtered['pos_change'] = laps_filtered.groupby(['raceId', 'driverId'])['position'].diff().abs()
    action_df = laps_filtered.groupby('raceId')['pos_change'].sum().reset_index()
    action_df.rename(columns={'pos_change': 'action_score'}, inplace=True)
    
    # C. Combine into Final Label
    metrics = races[['raceId']].merge(chaos_df, on='raceId', how='left')
    metrics = metrics.merge(action_df, on='raceId', how='left').fillna(0)
    
    chaos_thresh = metrics['chaos_score'].quantile(0.85)
    action_thresh = metrics['action_score'].quantile(0.85)
    
    metrics['is_exciting'] = (
        (metrics['chaos_score'] > chaos_thresh) | 
        (metrics['action_score'] > action_thresh)
    ).astype(int)
    
    return metrics[['raceId', 'is_exciting']]

labels_df = generate_labels(main_races, results_df, status_df, laps_df)

# --- Feature Engineering ---

# A. Grid Shakeup
# Grid shake-up measures how far drivers qualified from their usual season performance. 
# Bigger shake-up means the grid is more unusual. These out-of-position starts typically 
# lead to more overtakes and more unpredictable race behavior, making a key predictor of excitement.
qual_df = qual_df.merge(races_df[['raceId', 'year', 'round']], on='raceId', how='left')
qual_df = qual_df.sort_values(['year', 'round', 'driverId'])

season_avg = (
    qual_df.groupby(['year', 'driverId'])['position']
    .mean()
    .reset_index()
    .rename(columns={'position': 'season_avg_position'})
)

qual_pos = qual_df.merge(season_avg, on=["year", "driverId"], how="left")
qual_pos["shake"] = (qual_pos["position"] - qual_pos["season_avg_position"]).abs()

grid_shakeup = (
    qual_pos.groupby("raceId")["shake"]
    .mean()
    .reset_index()
    .rename(columns={"shake": "grid_shakeup"})
)

# B. Title Tension
title_tension = (
    stand_df.groupby("raceId")
    .apply(lambda df: df.sort_values("points", ascending=False).head(3)["points"].std(), include_groups=False)
    .reset_index(name="title_tension")
)

# --- Merge Datasets and Split ---
eda_df = (
    main_races
    .merge(labels_df, on="raceId", how="left")
    .merge(grid_shakeup, on="raceId", how="left")
    .merge(title_tension, on="raceId", how="left")
)
eda_df.dropna(inplace=True)

# Filter down to the exact 3 features the API expects + the label
features = ["grid_shakeup", "title_tension", "precipitation"]
feature_df = eda_df[features + ["is_exciting"]]

# Shuffle and create train_df and train_labels
feature_df = feature_df.sample(frac=1, random_state=0)
size = len(feature_df)
train_df = feature_df.iloc[:int(size * 0.8)].copy()
train_labels = train_df.pop("is_exciting")

# (Optional: test_df is not strictly needed for the training script since we only want to save the final model artifacts, but it's good for evaluating).
test_df = feature_df.iloc[int(size * 0.8):].copy()
test_labels = test_df.pop("is_exciting")

# Execute Cleaning
main_races = clean_and_merge_weather(races_df, weather_df)

# 1. Scale (normalize) the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)

# 2. Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0, max_depth=1)
model.fit(train_scaled, train_labels)

# 3. Export the artifacts
# joblib saves the scaler and trained model as files instead of in memory

# Scaling rules applied to training data
joblib.dump(scaler, "scaler.joblib")

# Trained Random Forest model
joblib.dump(model, "f1_rf_model.joblib")
print("Model and scaler saved successfully!")