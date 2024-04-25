import os

import numpy as np

from utils import compute_length


def random_method(instance_):
    n = int(instance_.nPoints)
    solution = np.random.choice(np.arange(n), size=n, replace=False)
    return solution


def nearest_neighbor(instance_, starting_node=0):
    dist_matrix = np.copy(instance_.dist_matrix)
    n = int(instance_.nPoints)
    node = starting_node
    tour = [node]
    for _ in range(n - 1):
        for new_node in np.argsort(dist_matrix[node]):
            if new_node not in tour:
                tour.append(new_node)
                node = new_node
                break
    # tour.append(starting_node)
    return np.array(tour)


def best_nearest_neighbor(instance_):
    solutions, lengths = [], []
    for start in range(instance_.nPoints):
        new_solution = nearest_neighbor(instance_, starting_node=start)
        solutions.append(new_solution)
        lengths.append(compute_length(new_solution, instance_.dist_matrix))

    if lengths is []:
        return None
    else:
        solution = solutions[np.argmin(lengths)]
        return solution


def multi_fragment_check_if_available(n1, n2, sol):
    if len(sol[str(n1)]) < 2 and len(sol[str(n2)]) < 2:
        return True
    else:
        return False


def multi_fragment_check_if_not_close(edge_to_append, sol):
    n1, n2 = edge_to_append
    from_city = n2
    if len(sol[str(from_city)]) == 0:
        return True
    partial_tour = [from_city]
    end = False
    iteration = 0
    while not end:
        if len(sol[str(from_city)]) == 1:
            if from_city == n1:
                return_value = False
                end = True
            elif iteration > 1:
                # print(f"iterazione {iteration}, elementi dentro partial {len(partial_tour)}",
                #       f"from city {from_city}")
                return_value = True
                end = True
            else:
                from_city = sol[str(from_city)][0]
                partial_tour.append(from_city)
                iteration += 1
        else:
            # print(from_city, partial_tour, sol[str(from_city)])
            for node_connected in sol[str(from_city)]:
                # print(node_connected)
                if node_connected not in partial_tour:
                    from_city = node_connected
                    partial_tour.append(node_connected)
                    # print(node_connected, sol[str(from_city)])
                    iteration += 1
    return return_value


def multi_fragment_create_solution(start_sol, sol, n):
    assert len(start_sol) == 2, "too many cities with just one link"
    end = False
    n1, n2 = start_sol
    from_city = n2
    sol_list = [n1, n2]
    iteration = 0
    while not end:
        for node_connected in sol[str(from_city)]:
            iteration += 1
            if node_connected not in sol_list:
                from_city = node_connected
                sol_list.append(node_connected)
                # print(f"prossimo {node_connected}",
                #       f"possibili {sol[str(from_city)]}",
                #       f"ultim tour {sol_list[-5:]}")
            if iteration > 300:
                if len(sol_list) == n:
                    end = True
    # sol_list.append(n1)
    return sol_list


def multi_fragment_mf(instance):
    mat = np.copy(instance.dist_matrix)
    mat = np.triu(mat)
    mat[mat == 0] = 100000
    solution = {str(i): [] for i in range(instance.nPoints)}
    start_list = [i for i in range(instance.nPoints)]
    inside = 0
    for el in np.argsort(mat.flatten()):
        node1, node2 = el // instance.nPoints, el % instance.nPoints
        possible_edge = [node1, node2]
        if multi_fragment_check_if_available(node1, node2,
                                             solution):
            if multi_fragment_check_if_not_close(possible_edge, solution):
                # print("entrato", inside)
                solution[str(node1)].append(node2)
                solution[str(node2)].append(node1)
                if len(solution[str(node1)]) == 2:
                    start_list.remove(node1)
                if len(solution[str(node2)]) == 2:
                    start_list.remove(node2)
                inside += 1
                # print(node1, node2, inside)
                if inside == instance.nPoints - 1:
                    # print(f"ricostruire la solutione da {start_list}",
                    #       f"vicini di questi due nodi {[solution[str(i)] for i in start_list]}")
                    solution = multi_fragment_create_solution(start_list, solution, instance.nPoints)
                    return solution