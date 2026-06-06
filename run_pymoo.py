import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.callback import Callback
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import load_german_data
from src.fitness import FitnessEvaluator
from src.individual import Individual


# --- KROK 1 i 2: Definicja problemu ---
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
        out["F"] = ind.objectives


# --- PODGLĄD NA ŻYWO (Callback) ---
class AccuracyMonitorCallback(Callback):
    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator

    def notify(self, algorithm):
        X = algorithm.opt.get("X")
        if X is not None and len(X) > 0:
            mid_idx = len(X) // 2
            x_sample = np.round(X[mid_idx]).astype(int)
            cache_key = tuple(x_sample)
            cached_acc = self.evaluator.accuracy_cache.get(cache_key, None)

            nf = int(np.sum(x_sample))
            if cached_acc is not None:
                print(
                    f" ---> [Gen {algorithm.n_gen}] Front: {len(X)} rozwiązań | "
                    f"Próbka: {nf} cech, Accuracy = {cached_acc:.4f}")
            else:
                print(
                    f" ---> [Gen {algorithm.n_gen}] Front: {len(X)} rozwiązań | "
                    f"Próbka: {nf} cech, Accuracy = (brak w cache)")


# --- KROK 3, 4 i 5: Wykonanie ---
def main():
    print("Pobieranie i przetwarzanie zbioru danych...")
    X_train, X_test, y_train, y_test = load_german_data()
    NUM_FEATURES = X_train.shape[1]
    POP_SIZE = 100
    MAX_GEN = 100

    evaluator = FitnessEvaluator(X_train, y_train, classifier_type='knn')
    problem = CreditRiskFeatureSelection(evaluator, NUM_FEATURES)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=BinaryRandomSampling(),
        crossover=PointCrossover(n_points=1, prob=0.9),
        mutation=BitflipMutation(prob=0.1),
        eliminate_duplicates=True
    )

    # Inteligentny warunek stopu — do testów (przedwczesne zatrzymanie gdy brak postępu)
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=10,
        n_max_gen=MAX_GEN
    )

    monitor = AccuracyMonitorCallback(evaluator)

    print("NSGA-II: Rozpoczęcie optymalizacji przez pymoo...")
    res = minimize(
        problem,
        algorithm,
        termination=termination,
        callback=monitor,
        seed=1,
        verbose=True
    )

    print("\n--- Optymalizacja zakończona ---")

    if res.X is None:
        print("BŁĄD: Algorytm nie znalazł żadnych rozwiązań.")
        return

    print("Obliczanie Accuracy dla wszystkich rozwiązań z ostatecznego frontu Pareto...")

    best_solutions = []
    for x in res.X:
        ind = Individual(np.round(x).astype(int))
        evaluator.evaluate(ind)
        best_solutions.append(ind)

    sorted_front = sorted(best_solutions, key=lambda ind: ind.accuracy, reverse=True)

    print("\nNajlepsze rozwiązania NSGA-II (posortowane po accuracy CV):")
    for i, ind in enumerate(sorted_front[:5]):
        nf = ind.objectives[0]
        iv = -ind.objectives[1]
        print(f" Rozwiązanie {i + 1}: Accuracy CV = {ind.accuracy:.4f}, "
              f"IV = {iv:.4f}, Liczba cech = {int(nf)}")

    # Ewaluacja najlepszego rozwiązania na zbiorze testowym
    print("\nEwaluacja najlepszego rozwiązania na zbiorze testowym:")
    best_ind = sorted_front[0]
    best_mask = best_ind.features == 1

    model = KNeighborsClassifier(n_neighbors=5, p=1, weights='uniform')
    model.fit(X_train.iloc[:, best_mask], y_train)
    test_accuracy = model.score(X_test.iloc[:, best_mask], y_test)
    print(f" Test Accuracy = {test_accuracy:.4f} | "
          f"Liczba cech = {int(best_ind.objectives[0])} | "
          f"IV = {-best_ind.objectives[1]:.4f}")


if __name__ == "__main__":
    main()