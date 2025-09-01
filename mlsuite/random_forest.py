"""
mlsuite/random_forest.py
A random forest model to rank feature importance
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class RandomForest:
    def __init__(
        self, X, y, test_size=0.2, n_estimators=200, random_state=42, top_n=20
    ):
        sns.set_theme()
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.top_n = top_n
        self.n_estimators = n_estimators

    def split_data(self):
        """perform a simple train-test-split in preparation for rf training"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

    def train(self):
        """train RF model on provided data"""
        self.split_data()
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.rf.fit(self.X_train, self.y_train)
        self.y_pred = self.rf.predict(self.X_test)

    def plot_n_features(self):
        """plot the top N features in the data provided as ranked by the RF model"""
        # print classification report
        print(classification_report(self.y_test, self.y_pred))

        # ... then figure out the importances of each feature
        importances = self.rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot top 20 features
        # top_n = 20
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.top_n), importances[indices[: self.top_n]], align="center")
        plt.xticks(
            range(self.top_n), self.X.columns[indices[: self.top_n]], rotation=90
        )
        plt.title(f"Top N TSFEL Features by RF Importance ({self.top_n = })")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()
