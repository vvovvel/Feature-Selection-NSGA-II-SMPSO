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
            # Używamy qcut do binowania, aby obliczyć IV dla każdej cechy
            binned_vals = pd.qcut(self.X_train[col], q=10, duplicates='drop')
            df = pd.DataFrame({'val': binned_vals, 'target': self.y_train})

            stats = df.groupby('val', observed=False)['target'].agg(['count', 'sum'])
            stats.columns = ['count', 'good']
            stats['bad'] = stats['count'] - stats['good']
            stats['good_dist'] = stats['good'] / self.total_good
            stats['bad_dist'] = stats['bad'] / self.total_bad

            mask = (stats['good_dist'] > 0) & (stats['bad_dist'] > 0)
            valid_stats = stats[mask].copy()

            if not valid_stats.empty:
                woe = np.log(valid_stats['good_dist'] / valid_stats['bad_dist'])
                iv = (valid_stats['good_dist'] - valid_stats['bad_dist']) * woe
                iv_list.append(float(iv.sum()))
            else:
                iv_list.append(0.0)
        return np.array(iv_list)

    def evaluate(self, individual):
        mask = individual.features == 1
        num_features = np.sum(mask)

        # Obsługa pustego zbioru cech
        if num_features == 0:
            individual.objectives = np.array([float(len(individual.features)), 0.0])
            individual.accuracy = 0.0
            return

        # f1 = liczba cech (minimalizacja)
        # f2 = suma IV (maksymalizacja, więc -suma IV)
        total_iv = np.sum(self.iv_scores[mask])
        individual.objectives = np.array([float(num_features), -total_iv])

        # OBLICZANIE ACCURACY (Jako metryka jakości rozwiązania)
        X_subset = self.X_train.iloc[:, mask]

        # Konfiguracja klasyfikatora zgodnie z potrzebami projektu
        if self.classifier_type == 'knn':
            # Zmieniono p=2 (Euklidesowa)
            model = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform')
        elif self.classifier_type == 'svm':
            model = SVC(kernel='rbf', C=1.0, gamma=0.1, class_weight='balanced', random_state=42)
        elif self.classifier_type == 'ann':
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20,
                                  learning_rate_init=0.5, momentum=0.3,
                                  solver='sgd', random_state=42)

        # Walidacja krzyżowa - przypisujemy ją tylko do atrybutu individual.accuracy
        scores = cross_val_score(model, X_subset, self.y_train, cv=10, scoring='balanced_accuracy')
        individual.accuracy = scores.mean()