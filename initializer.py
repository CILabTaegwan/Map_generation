import random

def initialize():
    """Initialize 100 individuals, each of which consists of 10000 bits"""
    population = []

    weights = [0.75, 0.05, 0.05, 0.05, 0.05, 0.05]

    for _ in range(100):
        individual = ""
        for _ in range(35):
            individual += str(random.choices(population=list(range(0, 6)), weights=weights, k=1)[0])
        population.append(individual)

    return population