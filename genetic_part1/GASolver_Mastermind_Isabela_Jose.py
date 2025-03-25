import mastermind as mm
import random

class Individual:
    """Represents an Individual for a genetic algorithm"""

    def __init__(self, chromosome: list, fitness: float):
        """Initializes an Individual for a genetic algorithm

        Args:
            chromosome (list): a list representing the individual's chromosome
            fitness (float): the individual's fitness (the higher, the better the fitness)
        """
        self.chromosome = chromosome
        self.fitness = fitness

    def __lt__(self, other):
        """Implementation of the less_than comparator operator"""
        return self.fitness < other.fitness

    def __repr__(self):
        """Representation of the object for print calls"""
        return f'Indiv({self.fitness:.1f},{self.chromosome})'

class GASolver:
    def __init__(self, selection_rate=0.5, mutation_rate=0.1):
        """Initializes an instance of a GA solver for a given problem

        Args:
            selection_rate (float, optional): Selection rate between 0 and 1.0. Defaults to 0.5.
            mutation_rate (float, optional): Mutation rate between 0 and 1.0. Defaults to 0.1.
        """
        self._selection_rate = selection_rate
        self._mutation_rate = mutation_rate
        self._population = []

    def reset_population(self, pop_size=50):
        """Initialize the population with pop_size random Individuals"""
        self._population = []
        for _ in range(pop_size):
            chromosome = MATCH.generate_random_guess()
            fitness = MATCH.rate_guess(chromosome)
            new_individual = Individual(chromosome, fitness)
            self._population.append(new_individual)

    def evolve_for_one_generation(self):
        """Apply the process for one generation:
        - Sort the population (Descending order)
        - Selection: Keep top fraction of the population
        - Reproduction: Recreate the same quantity by crossing surviving individuals
        - Mutation: Mutate individuals with probability mutation_rate
        """
        # Sort the population in descending order of fitness
        self._population.sort(reverse=True)

        # Selection: Keep top fraction
        survivors = int(self._selection_rate * len(self._population))
        parents = self._population[:survivors]

        # Reproduction: Create new children
        new_population = parents.copy()
        while len(new_population) < len(self._population):
            a, b = random.sample(parents, 2)  # Select two random parents
            x_point = random.randrange(0, len(a.chromosome))
            new_chromosome = a.chromosome[:x_point] + b.chromosome[x_point:]
            fitness = MATCH.rate_guess(new_chromosome)
            new_individual = Individual(new_chromosome, fitness)
            new_population.append(new_individual)

        # Mutation
        for individual in new_population[survivors:]:  # Avoid mutating parents
            if random.random() < self._mutation_rate:
                pos = random.randrange(0, len(individual.chromosome))
                valid_colors = mm.get_possible_colors()
                new_gene = random.choice(valid_colors)
                new_chromosome = (
                    individual.chromosome[:pos]
                    + [new_gene]
                    + individual.chromosome[pos + 1:]
                )
                individual.chromosome = new_chromosome
                individual.fitness = MATCH.rate_guess(new_chromosome)

        self._population = new_population

    def evolve_until(self, max_nb_of_generations=500, threshold_fitness=None):
        """Evolve the population until a condition is met:
        - Max number of generations is reached, or
        - A sufficiently high fitness value is achieved
        """
        for generation in range(max_nb_of_generations):
            self.evolve_for_one_generation()
            best_individual = self.get_best_individual()
            print(f"Generation {generation + 1}: Best fitness = {best_individual.fitness:.2f}")

            if threshold_fitness is not None and best_individual.fitness >= threshold_fitness:
                break

    def show_generation_summary(self):
        """Print some debug information on the current state of the population"""
        best = self.get_best_individual()
        print(f"Best individual: {best}")

    def get_best_individual(self):
        """Return the best Individual of the population"""
        return max(self._population, key=lambda ind: ind.fitness)

# Main code to solve the Mastermind problem
MATCH = mm.MastermindMatch(secret_size=4)
solver = GASolver()
solver.reset_population()
solver.evolve_until(threshold_fitness=MATCH.max_score())

best = solver.get_best_individual()
print(f"Best guess: {best.chromosome}")
print(f"Problem solved? {MATCH.is_correct(best.chromosome)}")