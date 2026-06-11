import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import pandas as pd

from src.data_loader import load_german_data
from src.fitness import FitnessEvaluator
from src.individual import Individual


class CreditRiskFeatureSelection(ElementwiseProblem):
    def __init__(self, evaluator, num_features):
        super().__init__(n_var=num_features,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=0, xu=1, vtype=int)
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        x_bin = np.round(x).astype(int)
        ind = Individual(x_bin)
        self.evaluator.evaluate(ind)

        out["F"] = [ind.objectives[0], -ind.objectives[1]]


def main():
    print("Pobieranie i przetwarzanie zbioru danych...")
    X_raw_train, X_train, X_test, y_train, y_test = load_german_data()
    NUM_FEATURES = X_train.shape[1]

    POP_SIZE = 100
    MAX_GEN = 500
    NUM_RUNS = 30
    results_summary = []

    evaluator = FitnessEvaluator(X_raw_train, X_train, y_train, classifier_type='knn')
    problem = CreditRiskFeatureSelection(evaluator, NUM_FEATURES)

    for run in range(NUM_RUNS):
        seed = run + 42
        print(f"\n--- Uruchomienie {run + 1}/{NUM_RUNS} (seed={seed}) ---")

        algorithm = NSGA2(
            pop_size=POP_SIZE,
            sampling=BinaryRandomSampling(),
            crossover=PointCrossover(n_points=1, prob=0.9),
            mutation=PolynomialMutation(prob=1.0),
            eliminate_duplicates=True
        )

        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8, cvtol=1e-6, ftol=0.0025, period=10, n_max_gen=MAX_GEN
        )

        res = minimize(problem, algorithm, termination=termination, seed=seed, verbose=False)

        if res.X is None:
            continue

        run_solutions = []
        for x in res.X:
            x_bin = np.round(x).astype(int)
            mask = x_bin == 1
            if np.sum(mask) == 0: continue

            ind = Individual(x_bin)
            evaluator.evaluate(ind)

            model = KNeighborsClassifier(n_neighbors=5, p=1, weights='uniform')
            cv_scores = cross_val_score(model, X_train.iloc[:, mask], y_train, cv=10, scoring='accuracy')
            ind.accuracy = cv_scores.mean()
            run_solutions.append(ind)

        best_run_ind = max(run_solutions, key=lambda ind: ind.accuracy)
        best_mask = best_run_ind.features == 1

        final_model = KNeighborsClassifier(n_neighbors=5, p=1, weights='uniform')
        final_model.fit(X_train.iloc[:, best_mask], y_train)
        test_acc = final_model.score(X_test.iloc[:, best_mask], y_test)
        test_f1 = f1_score(y_test, final_model.predict(X_test.iloc[:, best_mask]))

        results_summary.append({
            'run': run + 1,
            'train_acc_cv': best_run_ind.accuracy,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'num_features': int(best_run_ind.objectives[0]),
            'mask': best_mask,
            'iv_sum': best_run_ind.objectives[1]
        })

    df = pd.DataFrame(results_summary)
    print("\n" + "=" * 30)
    print(f"WYNIKI ZBIORCZE ({NUM_RUNS} uruchomień)")
    print("=" * 30)
    print(df.describe().loc[['mean', 'std', 'min', 'max']])

    # --- PODSUMOWANIE NAJLEPSZEGO MODELU ---
    best_idx = df['test_acc'].idxmax()
    best_row = df.loc[best_idx]

    print("\n" + "=" * 30)
    print("NAJLEPSZY MODEL (WG TEST_ACC)")
    print("=" * 30)
    print(f"Uruchomienie nr: {int(best_row['run'])}")
    print(f"Test Accuracy:   {best_row['test_acc']:.4f}")
    print(f"Test F1-Score:   {best_row['test_f1']:.4f}")
    print(f"Liczba cech:     {int(best_row['num_features'])}")
    print(f"Suma IV:         {best_row['iv_sum']:.4f}")

    selected_features = X_train.columns[best_row['mask']].tolist()
    print(f"Wybrane cechy:   {selected_features}")


if __name__ == "__main__":
    main()