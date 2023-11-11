#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics.cluster import adjusted_mutual_info_score

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
    return internal_edges

def objective_function_2(x):
    # Minimizing the number of communities
    unique_communities = len(np.unique(x))
    return unique_communities

# Evaluate the population's objectives
def evaluate_population(population):
    objectives = np.zeros((len(population), 2))
    for i, krill in enumerate(population):
        objectives[i] = krill.objective_values
    return objectives

class Krill:
    def __init__(self, position):
        self.position = position
        self.objective_values = [objective_function_1(position),
                                 objective_function_2(position)]
        self.step_size = random.random()  # Random initialization of step size

    def update_step_size(self, c):
        self.step_size = min(max(self.step_size + random.uniform(-c, c), 0), 1)

    def update_position(self, best_position, a, c):
        distance = np.linalg.norm(np.array(best_position) - np.array(self.position))
        randomness = np.random.normal(0, 1)
        self.position = (self.position + self.step_size * (best_position.astype(np.int64) - self.position) / (1 + a * distance) + c * randomness).astype(np.int64)

        # Bound the position within the search space if necessary
        self.position = np.clip(self.position, lower_bound, upper_bound).astype(np.int64)

        # Update the objective values
        self.objective_values = [objective_function_1(self.position),
                                 objective_function_2(self.position)]

# Parameters
population_size = 50
max_iterations = 100
a = 1  # Attraction coefficient
c = 1  # Randomness coefficient
lower_bound = 0  # Lower bound of the search space
upper_bound = 10  # Upper bound of the search space

# Initialize the population
population = []
for _ in range(population_size):
    position = np.random.randint(2, size=num_nodes, dtype=np.int64)  # Generate a random position
    krill = Krill(position)
    population.append(krill)

pareto_front = []
for iteration in range(max_iterations):
    # Update the step size for each krill
    for krill in population:
        krill.update_step_size(c)

    # Find the best krill based on the objectives
    population.sort(key=lambda x: (x.objective_values[0], -x.objective_values[1]))  # Sort based on the first objective in ascending order and the second objective in descending order
    best_position = population[0].position.astype(np.int64)  # Convert to int64

    # Update the position of each krill
    for krill in population:
        krill.update_position(best_position, a, c)

    # Evaluate the population
    objectives = evaluate_population(population)
    pareto_front.append(objectives)

# Display the results
plt.figure(figsize=(12, 5))

# Plot the communities of the graph
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G)
colors = [population[i].position for i in range(population_size)]
c = np.random.rand(num_nodes)
fig, ax = plt.subplots()
pos = nx.spring_layout(G)
x = [pos[i][0] for i in range(num_nodes)]
y = [pos[i][1] for i in range(num_nodes)]

ax.scatter(x, y, c=c, cmap='viridis')

plt.title('Graph Communities')

# Pareto Plot
plt.subplot(1, 2, 2)
pareto_front = np.array(pareto_front)
plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
plt.xlabel('Objective 1 (Number of Internal Edges within Communities)')
plt.ylabel('Objective 2 (Number of Communities)')
plt.title('Pareto Plot')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate modularity and NMI for the obtained communities
communities = [population[i].position for i in range(population_size)]
labels = ['']  # Replace with actual community labels if available
modularity_value = modularity(G, communities)
nmi_value = adjusted_mutual_info_score(labels, communities)

print("Modularity Value:", modularity_value)
print("NMI Value:", nmi_value)
