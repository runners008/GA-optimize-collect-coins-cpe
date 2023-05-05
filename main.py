import random
import math
import numpy as np
from collections import Counter
import time
import statistics

maze = np.array( [[ 3,  2, -1,  2, -1, -1, -3,  0,  0, -2,  0, -3, -3,  2,  0],
 [-2, -1, -3,  3,  2,  0, -3, -3, -1, -2, -3, -2,  1,  1,  2],
 [-2,  1, -3, -2, -1,  3,  1,  0, -3, -3, -1, -3,  1, -1,  2],
 [ 2,  2, -1, -1,  0, -1, -2,  0, -1,  1, -3, -1,  2,  0,  0],
 [-1, -1,  3, -1,  0,  0,  0, -1, -1,  0, -2, -3, -2, -1,  0],
 [ 1, -3,  0, -3, -3,  3, -2, -3,  1,  1,  3,  3, -3, -2,  0],
 [-2,  0,  3,  0,  1,  1,  0, -1,  3,  2, -2, -1, -3, -3,  3],
 [-2, -3, -2, -1, -1, -2, -3,  0,  2,  1, -1, -3,  1, -2, -1],
 [ 0,  1, -3,  3,  3, -2,  1,  2, -3, -2,  0, -3,  1, -1,  3],
 [ 3, -3,  3, -2,  3,  1,  2,  0, -1,  3,  0, -3,  3,  2,  0],
 [ 2, -1,  2,  0,  2, -2,  2,  3,  3, -1,  1, -2,  0, -1,  0],
 [-3,  3, -2,  2, -1,  3, -2, -3,  2,  3, -1, -3, -3,  0,  0],
 [ 3,  0,  3, -1,  0,  2,  1,  1,  1,  3,  1,  0,  3,  1, -3],
 [ 3, -2,  3,  3, -3,  1,  3,  2,  1, -3, -2,  2, -2, -3,  2],
 [ 3,  2, -2,  3, -2,  2, -3,  0, -1,  1,  3,  2,  3,  1,  0]])


# Runn GA Section

gene_range = range(4)  # 0, 1, 2, 3
chromosome_length = 16
population_size = 60

range_coins = 34

generation_limit = 100000
not_improved_limit = 5000

current_generation = []
best_fitness = []
maze_size = len(maze[0])

cross_over_rate = 0.7
mutation_rate = 0.15
elitism_rate = 0.2
alien_rate = 0.1

separator = "-"


# 1. Initial Population

def random_population(chromosome_length: int, population_size: int) -> list:
    return [[random.choice(gene_range) for _ in range(chromosome_length)] for _ in range(population_size)]


# 2. Evaluate (assigned fitness value)
def assigned_fitness(input_maze, gene):
    func_maze = np.array(input_maze)

    current_coins = 0
    distance = len(gene)
    energy = 0

    row, col = maze_size // 2, maze_size // 2

    prev_action = 5

    for action in gene:
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < maze_size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < maze_size - 1:
            col += 1
        else:
            distance -= 1

        if prev_action == action:
            energy += 0.5
        else:
            energy += 1

        if current_coins + func_maze[row, col] >= 0:
            current_coins += func_maze[row, col]
        func_maze[row][col] = 0

        prev_action = action

    chro_fitness = weighted_sum(current_coins, 1, distance, 0, energy, 0, len(gene))
    return chro_fitness, current_coins, distance, energy


# 2.2 Weighted-sum method
# p1 coins collect (0 - 41) เปลี่ยนทีหลังได้
# p2 distance travel (1+len(maze[0]/2) - len(gene))
# p3 energy used (0.5+len(gene)/2 - len(gene))
def weighted_sum(p1, w1, p2, w2, p3, w3, gene_len) -> float:

    # range_coins = 60
    range_dist = gene_len - (1 + len(maze[0]) / 2)
    range_en = gene_len - (0.5 + gene_len / 2)

    coin_score = w1 * (p1 / range_coins)
    dist_score = w2 * ((p2 - 1 - (gene_len / 2)) / range_dist)
    en_score = w3 * ((p3 - 0.5 - (gene_len / 2)) / range_en)

    return coin_score + dist_score - en_score


# 3. Conditioned Met or not?
def check_last(lst, n) -> bool:
    return all(elem == lst[-1] for elem in lst[-n:])


def round_exceed_or_not_improved(round: int, limit: int, round_limit: int, fitness) -> bool:
    if round > round_limit:
        return True
    elif round > limit and check_last(best_fitness, limit):
        return True
    elif fitness == 1.00:
        return True
    else:
        return False


# 3.1 Check duplicate
def is_too_many_duplicate(generation, dup_limit):
    num_lists = len(generation)
    unique_lists = Counter(tuple(lst) for lst in generation)
    most_common_list, most_common_count = unique_lists.most_common(1)[0]

    if most_common_count >= num_lists * dup_limit:
        return True
    else:
        return False


# 4. Parent Selection (Roulette wheel)
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    r = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for i in range(len(population)):
        cumulative_fitness += fitness_scores[i]
        if cumulative_fitness > r:
            return population[i]
    return population[-1]


# 4. (tournament selection)

# 4.1 (เหมือน report) 5->4
# def tournament_selection(population, fitness_scores, tournament_size):
#     tournament = random.sample(range(len(population)), tournament_size)
#     winner_index = tournament[0]
#     for i in tournament:
#         if fitness_scores[i] > fitness_scores[winner_index]:
#             winner_index = i
#     return population[winner_index]

# def tournament(population, fitness_scores, size):
#   pop = population
#   groups = [current_generation[i:i+4] for i in range(0, len(current_generation), 4)]
#   winners = [tournament_selection(group, fitness_scores, 4) for group in groups]
#   champion = tournament_selection(winners, [assigned_fitness(maze, gene) for gene in winners], len(winners))
#   return champion

# 4.2 random 10% then pick the best
def tournament_selection(population, fitness_scores):
    tournament = random.sample(range(len(population)), int(len(population) * 0.05))
    winner_index = tournament[0]
    for i in tournament:
        if fitness_scores[i] > fitness_scores[winner_index]:
            winner_index = i
    return population[winner_index]


# 5. Cross-over
def cross_over(parent_a, parent_b, func_mutation_rate):
    length = len(parent_a)
    crossover_point = random.randint(0, length - 1)
    child = parent_a[:crossover_point] + parent_b[crossover_point:]
    if random.uniform(0, 1) < func_mutation_rate:
        mutation_point = random.randint(0, length - 1)
        mutate_gene = random.randint(0, 3)
        child[mutation_point] = mutate_gene

    return child


# Elitism
def select_max_indices(lst, n):
    max_indices = []
    for i in range(n):
        max_index = max(range(len(lst)), key=lst.__getitem__)
        max_indices.append(max_index)
        lst[max_index] = float('-inf')
    return max_indices


# Main method
def ga():
    current_generation = random_population(chromosome_length, population_size)
    generation = 0
    max_gene = ""
    max_fitness = 0
    while not round_exceed_or_not_improved(generation, not_improved_limit, generation_limit, max_fitness):
        fitness_scores = []
        fitness_coins = []
        fitness_dist = []
        fitness_energy = []
        new_generation = []

        for gene in current_generation:
            fn_score, fn_coin, fn_dist, fn_en = assigned_fitness(maze, gene)
            fitness_scores.append(fn_score)
            fitness_coins.append(fn_coin)
            fitness_dist.append(fn_dist)
            fitness_energy.append(fn_en)

        # Announce the best gene fitness of the current generation
        index = fitness_scores.index(max(fitness_scores))
        best_in_gen_coins = fitness_coins[index]
        best_in_gen_distance = fitness_dist[index]
        best_in_gen_en = fitness_energy[index]

        format_gene = separator.join(str(i) for i in current_generation[index])
        best_fitness.append(max(fitness_scores))
        # if generation % 2000 == 0:
        # print('--- GEN',  "{:6,}".format(generation), '---')
        # for chro in current_generation:
        #   print("genes", chro)
        if len(best_fitness) > 2 and best_fitness[-1] != best_fitness[-2]:
            print('gen', "{:6,}".format(generation),
                  'fn:', "{:0>3.3f}".format(max(fitness_scores)),
                  'av:', "{:0>4.3f}".format(sum(fitness_scores)/len(fitness_scores)),
                  'coin:', "{:2.0f}".format(best_in_gen_coins),
                  'dist:', "{:3.0f}".format(best_in_gen_distance),
                  'ener:', "{:3.1f}".format(best_in_gen_en),
                  'gene:', format_gene
                  )
            max_gene = format_gene
            max_fitness = max(fitness_scores)
        # 80% - Cross over
        for i in range(int(population_size * cross_over_rate)):
            parent_a = roulette_wheel_selection(current_generation, fitness_scores)
            parent_b = tournament_selection(current_generation, fitness_scores)
            new_generation.append(cross_over(parent_a, parent_b, mutation_rate))

        if is_too_many_duplicate(new_generation, 0.6):
            # print("TOO DUP at", generation)
            n = int(len(new_generation) * 0.6)
            # need to remove duplicate chromosome instead of random out.
            new_generation = random.sample(new_generation, len(new_generation) - n)
            for _ in range(n):
                alien = random_population(chromosome_length, 1)
                new_generation.append(alien[0])

        # Elitism
        elite_index = select_max_indices(fitness_scores, int(population_size * elitism_rate))
        for index in elite_index:
            new_generation.append(current_generation[index])

        # Alien
        for _ in range(int(population_size * alien_rate)):
            alien = random_population(chromosome_length, 1)
            new_generation.append(alien[0])

        current_generation = new_generation
        generation += 1
    return max(fitness_scores), max(fitness_coins)


# Hill-climbing
def evaluate_fitness_hill(input_maze, gene):
    func_maze = np.array(input_maze)
    distance = len(gene)
    energy = 0

    prev_action = 5

    current_coins = 0
    row, col = maze_size // 2, maze_size // 2

    for action in gene:
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < maze_size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < maze_size - 1:
            col += 1
        else:
            distance -= 1

        if prev_action == action:
            energy += 0.5
        else:
            energy += 1

        if current_coins + func_maze[row, col] >= 0:
            current_coins += func_maze[row, col]
        func_maze[row][col] = 0

        prev_action = action

    chro_fitness = weighted_sum(current_coins, 1.0, distance, 0, energy, 0, len(gene))
    return chro_fitness, current_coins, distance, energy

# def assigned_fitness(input_maze, gene):
#     func_maze = np.array(input_maze)
#
#     current_coins = 0
#     distance = len(gene)
#     energy = 0
#
#     row, col = maze_size // 2, maze_size // 2
#
#     prev_action = 5
#
#     for action in gene:
#         if action == 0 and row > 0:
#             row -= 1
#         elif action == 1 and row < maze_size - 1:
#             row += 1
#         elif action == 2 and col > 0:
#             col -= 1
#         elif action == 3 and col < maze_size - 1:
#             col += 1
#         else:
#             distance -= 1
#
#         if prev_action == action:
#             energy += 0.5
#         else:
#             energy += 1
#
#         if current_coins + func_maze[row, col] >= 0:
#             current_coins += func_maze[row, col]
#         func_maze[row][col] = 0
#
#         prev_action = action
#
#     chro_fitness = weighted_sum(current_coins, 0.6, distance, 0.1, energy, 0.3, len(gene))
#     return chro_fitness, current_coins, distance, energy


def hill_climbing(func_maze):
    actions = [0, 1, 2, 3]
    current_solution = [random.choice(actions) for i in range(20)]
    current_fitness, _, _, _ = evaluate_fitness_hill(func_maze, current_solution)
    count = 0

    while True:
        neighbors = []
        for i in range(len(current_solution)):
            for a in actions:
                neighbor = current_solution[:]
                neighbor[i] = a
                neighbors.append(neighbor)

        best_neighbor = max(neighbors, key=lambda x: evaluate_fitness_hill(func_maze, x))
        max_best_fitness, _, _, _ = evaluate_fitness_hill(func_maze, best_neighbor)

        if max_best_fitness > current_fitness:
            count = 0
            current_solution = best_neighbor
            current_fitness = max_best_fitness
        elif count < 10:
            count += 1
        else:
            break

    return current_solution, current_fitness


# hill_result = {}
# for exp in range(200):
#     sol, fitness = hill_climbing(maze)
#     coin = int(fitness ** 2)
#     hill_result[coin] = hill_result.get(coin, 0) + 1
#
# for key in sorted(hill_result.keys()):
#     print(key, 'coins:', hill_result[key], 'exp')


result = {}
result_fitness = []
experiment_times = []
exp_times = 100
for exp in range(exp_times):
    start_time = time.perf_counter()
    exp_fitness, exp_coins = ga()
    # solution, exp_fitness = hill_climbing(maze)
    end_time = time.perf_counter()
    experiment_times.append(end_time - start_time)

    print("exp", exp)
    # result_fitness.append(exp_fitness)
    # result[exp_fitness] = result.get(exp_fitness, 0) + 1
    result_fitness.append(exp_fitness)
    result[exp_coins] = result.get(exp_coins, 0) + 1

average_fitness = statistics.mean(result_fitness)
average_time = statistics.mean(experiment_times)
min_time = min(experiment_times)
max_time = max(experiment_times)

for key in sorted(result.keys()):
    print("{:2.0f}".format(key), 'coins:', result[key], 'exp')

print("average fitness:", average_fitness)
print("average time:", average_time)
print("min time:", min_time)
print("max time:", max_time)

# print("--- CONCLUSION ---")
# print('Best gene is:', max_gene)
# print('fitness:', max_fitness, 'coins:', "{:2.0f}".format((max_fitness) ** 2))
# print("--- END ---")


