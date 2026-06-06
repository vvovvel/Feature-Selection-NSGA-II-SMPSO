import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from src.data_loader import load_german_data
from src.fitness import FitnessEvaluator
from src.individual import Individual


# --- KROK 1 i 2: Definicja problemu ---
class CreditRiskFeatureSelection(ElementwiseProblem):
    def __init__(self, evaluator, num_features):
        super().__init__(n_var=num_features,
                         n_obj=2,  # Minimalizacja NF i -IV
                         n_ieq_constr=0,
                         xl=0, xu=1, vtype=int)
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        # x z pymoo to wektor, np. [0, 1, 1, 0...]
        # Tworzymy Twój obiekt Individual
        x_bin = np.round(x).astype(int)
        ind = Individual(x_bin)

        # Oceniamy osobnika
        self.evaluator.evaluate(ind)

        # Przekazujemy cele z powrotem do pymoo
        out["F"] = ind.objectives


# --- KROK 3, 4 i 5: Wykonanie ---
def main():
    print("Pobieranie i przetwarzanie zbioru danych...")
    X_train, X_test, y_train, y_test = load_german_data()
    NUM_FEATURES = X_train.shape[1]
    POP_SIZE = 100
    MAX_GEN = 100

    evaluator = FitnessEvaluator(X_train, y_train, classifier_type='knn')
    problem = CreditRiskFeatureSelection(evaluator, NUM_FEATURES)

    # Inicjalizacja NSGA-II pod problem binarny
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=BinaryRandomSampling(),
        crossover=PointCrossover(n_points=1, prob=0.9),
        mutation=BitflipMutation(prob=1.0 / NUM_FEATURES),
        eliminate_duplicates=True
    )

    print("NSGA-II: Rozpoczęcie optymalizacji przez pymoo...")
    res = minimize(
        problem,
        algorithm,
        ('n_gen', MAX_GEN),
        seed=1,
        verbose=True  # Pokaże logi z kolejnych generacji
    )

    print("\n--- Optymalizacja zakończona ---")
    print("Obliczanie Accuracy dla rozwiązań z frontu Pareto...")

    best_solutions = []
    # res.X zawiera znalezione zbiory cech, res.F zawiera cele
    for x in res.X:
        ind = Individual(np.round(x).astype(int))
        evaluator.evaluate(ind)  # Wywołujemy, aby zaktualizować ind.accuracy
        best_solutions.append(ind)

    # Sortowanie ostatecznych wyników po dokładności
    sorted_front = sorted(best_solutions, key=lambda ind: ind.accuracy, reverse=True)

    print("\nNajlepsze rozwiązania NSGA-II (posortowane po accuracy):")
    for i, ind in enumerate(sorted_front[:5]):
        nf = ind.objectives[0]
        iv = -ind.objectives[1]
        print(f" Rozwiązanie {i + 1}: Accuracy = {ind.accuracy:.4f}, IV = {iv:.4f}, Liczba cech = {int(nf)}")


if __name__ == "__main__":
    main()