from sklearn.neighbors import KNeighborsClassifier


classifier = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='auto',
    metric='euclidean',
    n_jobs=-1
)