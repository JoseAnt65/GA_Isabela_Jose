import cities
import random

class Individual:
    """Represents an Individual for a genetic algorithm"""

    def __init__(self, chromosome: list, fitness: float):
        """Initializes an Individual for a genetic algorithm

        Args:
            chromosome (list): a list representing the individual's chromosome
            fitness (float): the individual's fitness (higher is better)
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
    def __init__(self, city_dict, selection_rate=0.5, mutation_rate=0.1):
        """Initializes an instance of a GA solver for the TSP problem

        Args:
            city_dict (dict): A dictionary of cities with coordinates
            selection_rate (float, optional): Selection rate between 0 and 1.0. Defaults to 0.5.
            mutation_rate (float, optional): Mutation rate between 0 and 1.0. Defaults to 0.1.
        """
        self.city_dict = city_dict
        self._selection_rate = selection_rate
        self._mutation_rate = mutation_rate
        self._population = []

    def reset_population(self, pop_size=50):
        """Initialize the population with pop_size random Individuals"""
        self._population = []
        for _ in range(pop_size):
            chromosome = cities.default_road(self.city_dict)
            random.shuffle(chromosome)  # Shuffle for randomness
            fitness = -cities.road_length(self.city_dict, chromosome)  # Negative length as fitness
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

            # Reproduce with crossover
            x_point = random.randint(1, len(a.chromosome) - 1)  # Avoid empty splits
            child_chromosome = a.chromosome[:x_point]
            child_chromosome += [city for city in b.chromosome if city not in child_chromosome]

            fitness = -cities.road_length(self.city_dict, child_chromosome)
            new_individual = Individual(child_chromosome, fitness)
            new_population.append(new_individual)

        # Mutation
        for individual in new_population[survivors:]:  # Avoid mutating parents
            if random.random() < self._mutation_rate:
                i, j = random.sample(range(len(individual.chromosome)), 2)
                individual.chromosome[i], individual.chromosome[j] = (
                    individual.chromosome[j],
                    individual.chromosome[i],
                )
                individual.fitness = -cities.road_length(self.city_dict, individual.chromosome)

        self._population = new_population

    def evolve_until(self, max_nb_of_generations=500):
        """Evolve the population for a set number of generations"""
        for generation in range(max_nb_of_generations):
            self.evolve_for_one_generation()
            best_individual = self.get_best_individual()
            print(f"Generation {generation + 1}: Best fitness = {best_individual.fitness:.2f}")

    def show_generation_summary(self):
        """Print some debug information on the current state of the population"""
        best = self.get_best_individual()
        print(f"Best individual: {best}")

    def get_best_individual(self):
        """Return the best Individual of the population"""
        return max(self._population, key=lambda ind: ind.fitness)

# Main code to solve the TSP problem
city_dict = cities.load_cities("cities.txt")
solver = GASolver(city_dict)
solver.reset_population()
solver.evolve_until(max_nb_of_generations=500)

best = solver.get_best_individual()
print(f"Best road: {best.chromosome}")
print(f"Road length: {-best.fitness:.2f}")

# Visualize the result
cities.draw_cities(city_dict, best.chromosome)
