import numpy as np
import pandas as pd


class FitnessEvaluator:
    def __init__(self, X_raw, X_train, y_train, classifier_type='knn'):
        """
        X_raw: surowe dane przed skalowaniem i kodowaniem (do IV)
        X_train: przetworzone dane po MinMaxScaler i LabelEncoder (do KNN/SVM/ANN)
        """
        self.X_raw = X_raw
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
        for col in self.X_raw.columns:
            unique_vals = self.X_raw[
                col].nunique()  # Liczy liczbę unikalnych wartości w danej kolumnie, aby zdecydować o sposobie binningu

            if unique_vals <= 15:  # Jeśli cecha ma mało unikalnych wartości, nie wymaga grupowania
                binned_vals = self.X_raw[col]
            else:  # Jeśli cecha jest ciągła, dzielimy ją na 15 buckets
                binned_vals = pd.qcut(self.X_raw[col], q=15,
                                      duplicates='drop')  # Dzieli dane na 15 kawalkow, kazda zmienna jest zastapiona numerem swojego bucketa

            df = pd.DataFrame({  # Tworzy tymczasową ramkę danych do obliczeń
                'val': binned_vals.values,  # Wartości (lub przypisane kubełki)
                'target': self.y_train.values  # Odpowiadające im etykiety celu (0 lub 1)
            })

            stats = df.groupby('val', observed=False)['target'].agg(['count',
                                                                     'sum']) #całkowita suma wartości w obrębie bucketa
            stats.columns = ['count', 'good']  # Nazywa kolumny statystyk
            stats['bad'] = stats['count'] - stats['good']  # Oblicza liczbę "złych" przypadków (pozostałe)
            stats['good_dist'] = stats[
                                     'good'] / self.total_good  # Oblicza udział "dobrych" w danym kubełku względem wszystkich "dobrych"
            stats['bad_dist'] = stats[
                                    'bad'] / self.total_bad

            epsilon = 1e-6  #eps dla stabilności numerycznej
            stats['good_dist'] = np.maximum(stats['good_dist'],
                                            epsilon)  #unikamy dzielenia przez zero
            stats['bad_dist'] = np.maximum(stats['bad_dist'], epsilon)

            woe = np.log(stats['good_dist'] / stats[
                'bad_dist'])  #WoE wzór
            iv = (stats['good_dist'] - stats['bad_dist']) * woe  # IV wzór
            iv_list.append(float(iv.sum()))  # sumuje iv dla wszystkich bucketów w tej cesze i dodaje je do listy

        return np.array(iv_list)  # zwraca tablice IV dla wszystkich cech po kolei

    def evaluate(self, individual):
        """
        Ocena osobnika na podstawie dwóch funkcji celu:
        1. Liczba wybranych cech (minimalizacja).
        2. Sumaryczna wartość IV (maksymalizacja).
        """
        mask = individual.features == 1
        num_features = np.sum(mask)

        # Jeśli nie wybrano żadnej cechy, total_iv przyjmuje 0.0
        if num_features == 0:
            total_iv = 0.0
        else:
            total_iv = np.sum(self.iv_scores[mask])

        individual.objectives = np.array([float(num_features), float(total_iv)])