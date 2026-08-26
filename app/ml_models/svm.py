from sklearn.svm import SVC


classifier = SVC(
    kernel='linear',
    max_iter=1000,
    break_ties=True,
    random_state=42
)