from sklearn.ensemble import RandomForestClassifier


classifier = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)