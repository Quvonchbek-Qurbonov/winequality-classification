from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    min_samples_split=5,
    random_state=42
)