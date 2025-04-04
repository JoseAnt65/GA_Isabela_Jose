# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:24:15 2022

@author: agademer & tdrumond

Template for exercise 1
(genetic algorithm module specification)
"""

#import package
import mastermind as mm
import random

#so the code stays the same
MATCH = mm.MastermindMatch(secret_size=4)
solver = GASolver()
solver.reset_population()
solver.evolve_until(threshold_fitness=MATCH.max_score())

class Individual:
    """Represents an Individual for a genetic algorithm"""

    def __init__(self, chromosome: list, fitness: float):
        """Initializes an Individual for a genetic algorithm 

        Args:
            chromosome (list[]): a list representing the individual's chromosome
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
        """Initializes an instance of a ga_solver for a given GAProblem

        Args:
            selection_rate (float, optional): Selection rate between 0 and 1.0. Defaults to 0.5.
            mutation_rate (float, optional): mutation_rate between 0 and 1.0. Defaults to 0.1.
        """
        self._selection_rate = selection_rate
        self._mutation_rate = mutation_rate
        self._population = self.population.append(new_individual, fitness, chromosome)


    def reset_population(self, pop_size=50):
        """ Initialize the population with pop_size random Individuals """
        #replace your code (PASS)
        self._population = [] #Ensure population is reset
        for _ in range(pop_size):
            chromosome = MATCH.generate_random_guess()
            fitness = MATCH.rate_guess(chromosome)
            new_individual = Individual(chromosome, fitness)
            self._population.append(new_individual)


    def evolve_for_one_generation(self):
        """ Apply the process for one generation : 
            -	Sort the population (Descending order)
            -	Selection: Remove x% of population (less adapted)
            -   Reproduction: Recreate the same quantity by crossing the 
                surviving ones 
            -	Mutation: For each new Individual, mutate with probability 
                mutation_rate i.e., mutate it if a random value is below   
                mutation_rate
        """
        #replace your code (PASS)
        #we are sorting the population and organizing it in decreasing order
        self._population.sort(reverse=True)

        #Selection: keep top fraction
        survivors = int(self._selection_rate * len(self._population))
        parents = self._population[:survivors]

        #Reproduction: create the new children
        new_population = parents.copy()
        while len(new_population) < len(self._population):
            a, b = random.sample(parents, 2) #select two random parents
        number = random.randrange(start,stop)
        x_point = random.randrange(0, len(a.chromosome))
        y_point = random.randrange(0, len(b.chromosome))
        new_chrom = a.chromosome[0:x_point] + b.chromosome[x_point: ]
        new_individual = Individual(new_chrom, MATCH.rate_guess(new_chrom))
        fitness = MATCH.rate_guess(new_chrom)
        number = random.random()
        valid_colors = mm.getPossibleColors()
        new_gene = random.choice(valid_colors)
        new_chrom = a.chromosome[0:pos] + [new_gene] + a.chromosome[pos+1:]
        new_individual = Individual(new_chrom, fitness)
        fitness = MATCH.rate_guess(new_chrom)
        new_population.append(new_individual)
    
    #Mutation
    for individual in new_population[survivors:]: #avoid mutating parents
        if random.random() < self._mutation_rate:
            pos = random.randrange(0,len(individual.chromosome))
            valid_colors = mm.get_possible_colors()
            

    def show_generation_summary(self):
        """ Print some debug information on the current state of the population """
        pass  # REPLACE WITH YOUR CODE

    def get_best_individual(self):
        """ Return the best Individual of the population """
        #pass REPLACE WITH YOUR CODE
        get_best_individual = self.population [0]


    def evolve_until(self, max_nb_of_generations=500, threshold_fitness=None):
        """ Launch the evolve_for_one_generation function until one of the two condition is achieved : 
            - Max nb of generation is achieved
            - The fitness of the best Individual is greater than or equal to
              threshold_fitness
        """
        #pass  # REPLACE WITH YOUR CODE
        
    best = solver.get_best_individual()
    print(f"Best guess {best.chromosome}")
    print(f"Problem solved? {MATCH.is_correct(best.chromosome)}")