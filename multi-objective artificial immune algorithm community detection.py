#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics.cluster import adjusted_mutual_info_score
from deap import base, creator, tools, algorithms

# Create the Karate Club graph
G = nx.karate_club_graph()
num_nodes = G.number_of_nodes()
adj_matrix = nx.to_numpy_array(G, dtype=np.int64)

# Define the objective functions
def objective_function_1(x):
    # Maximizing the number of internal edges within communities
    internal_edges = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if x[i] == x[j] and adj_matrix[i][j] == 1:
                internal_edges += 1
    return internal_edges,

def objective_function_2(x):
    # Minimizing the number of communities
    unique_communities = len(np.unique(x))
    return unique_communities,

# Create types for fitness and individuals
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_nodes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda x: (objective_function_1(x), objective_function_2(x)))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# Parameters
population_size = 50
max_generations = 100

# Create initial population
population = toolbox.population(n=population_size)

pareto_front = []
for generation in range(max_generations):
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select the next generation individuals
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    pareto_front.extend(offspring)

    # Select individuals for the next generation
    population = toolbox.select(population + offspring, k=population_size)

# Extract Pareto front objectives
pareto_objectives = [ind.fitness.values for ind in pareto_front]

# Display the results
plt.figure(figsize=(12, 5))

# Plot the communities of the graph
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G)
colors = [population[i] for i in range(population_size)]
nx.draw(G, pos, node_color=colors, with_labels=True, cmap='viridis')
plt.title('Graph Communities')

# Pareto Plot
plt.subplot(1, 2, 2)
pareto_objectives = np.array(pareto_objectives)
plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1])
plt.xlabel('Objective 1 (Number of Internal Edges within Communities)')
plt.ylabel('Objective 2 (Number of Communities)')
plt.title('Pareto Plot')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate modularity and NMI for the obtained communities
best_individual = tools.selBest(population, k=1)[0]
communities = best_individual
labels = ['']  # Replace with actual community labels if available
modularity_value = modularity(G, communities)
nmi_value = adjusted_mutual_info_score(labels, communities)

print("Modularity Value:", modularity_value)
print("NMI Value:", nmi_value)
