from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd


class FitnessEvaluator:
    def __init__(self, X_train, y_train, classifier_type='knn'):
        self.X_train = X_train
        self.classifier_type = classifier_type.lower()
        self.accuracy_cache = {}

        if hasattr(y_train, 'iloc'):
            self.y_train = y_train.iloc[:, 0] if len(y_train.shape) > 1 else y_train
        else:
            self.y_train = pd.Series(y_train)

        self.total_good = (self.y_train == 1).sum()
        self.total_bad = (self.y_train == 0).sum()
        self.iv_scores = self._calculate_all_iv()

    def _calculate_all_iv(self):
        iv_list = []
        for col in self.X_train.columns:
            unique_vals = self.X_train[col].nunique()

            if unique_vals <= 10:
                binned_vals = self.X_train[col]
            else:
                binned_vals = pd.qcut(self.X_train[col], q=10, duplicates='drop')

            # Użycie .values zapobiega błędom wyrównania indeksów
            df = pd.DataFrame({
                'val': binned_vals.values,
                'target': self.y_train.values
            })

            stats = df.groupby('val', observed=False)['target'].agg(['count', 'sum'])
            stats.columns = ['count', 'good']
            stats['bad'] = stats['count'] - stats['good']
            stats['good_dist'] = stats['good'] / self.total_good
            stats['bad_dist'] = stats['bad'] / self.total_bad

            # Metoda wygładzania (epsilon) zapobiegająca log(0) i dzieleniu przez 0
            epsilon = 1e-6
            stats['good_dist'] = np.maximum(stats['good_dist'], epsilon)
            stats['bad_dist'] = np.maximum(stats['bad_dist'], epsilon)

            # Liczymy IV dla wszystkich koszyków bez odrzucania jakichkolwiek wartości
            woe = np.log(stats['good_dist'] / stats['bad_dist'])
            iv = (stats['good_dist'] - stats['bad_dist']) * woe

            iv_list.append(float(iv.sum()))

        return np.array(iv_list)

    def evaluate(self, individual):
        mask = individual.features == 1
        num_features = int(np.sum(mask))

        if num_features == 0:
            individual.objectives = np.array([float(len(individual.features)), -0.0])
            individual.accuracy = 0.0
            return

        total_iv = np.sum(self.iv_scores[mask])
        individual.objectives = np.array([float(num_features), -total_iv])

        # Sprawdzenie cache przed kosztowną ewaluacją
        cache_key = tuple(individual.features)
        if cache_key in self.accuracy_cache:
            individual.accuracy = self.accuracy_cache[cache_key]
            return

        X_subset = self.X_train.iloc[:, mask]

        if self.classifier_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5, p=1, weights='uniform')
            cv_folds = 5
        elif self.classifier_type == 'svm':
            model = SVC(kernel='rbf', C=1.0, gamma=0.1, class_weight='balanced', random_state=42)
            cv_folds = 10
        elif self.classifier_type == 'ann':
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20,
                                  learning_rate_init=0.5, momentum=0.3,
                                  solver='sgd', random_state=42)
            cv_folds = 10
        else:
            raise ValueError(
                f"Nieznany classifier_type: '{self.classifier_type}'. Dostępne: 'knn', 'svm', 'ann'.")

        scores = cross_val_score(model, X_subset, self.y_train, cv=cv_folds, scoring='accuracy')
        individual.accuracy = scores.mean()
        self.accuracy_cache[cache_key] = individual.accuracy