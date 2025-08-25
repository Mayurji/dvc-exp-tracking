from dvclive import Live
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

n_estimators = 100
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

with Live() as live:

    live.log_param("n_estimators", n_estimators)

    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)

    live.log_metric("train/f1", f1_score(y_train, y_train_pred, average="weighted"), plot=False)
    live.log_sklearn_plot(
        "confusion_matrix", y_train, y_train_pred, name="train/confusion_matrix",
        title="Train Confusion Matrix")

    y_test_pred = clf.predict(X_test)

    live.log_metric("test/f1", f1_score(y_test, y_test_pred, average="weighted"), plot=False)
    live.log_sklearn_plot(
        "confusion_matrix", y_test, y_test_pred, name="test/confusion_matrix",
        title="Test Confusion Matrix")