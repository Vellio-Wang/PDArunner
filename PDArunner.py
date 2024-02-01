import time
import traceback
import numpy as np
import random as rd
import copy
import multiprocessing as mp
import os
import datetime
import gc
from concurrent.futures import ProcessPoolExecutor
from typing import Union
from math import ceil, sqrt, exp
from tqdm import tqdm


def PDA_run_imap_unordered(individuals: list or set, save_dir: str, core_num=1, **kwargs):
    """
    This function controls the model to run.
    @param individuals:
        The list or set of instances of individuals.
    @param save_dir:
        The directory to save result file.
    @param core_num:
        It specifies the CPU number to be involved in multiprocessing
    @param kwargs:
        chunksize:
            It controls the number assigned to each CPU for each time in multiprocessing. Same as chunksize in multiprocessing.pool.imap
    Note:
        The output file will be always in .txt format. The information for an individual will be saved in one line. Different information
        is separated by ';'. The information include:
        ID;t;birth_hab;birth_loc;step_num;Success;Reason;Tar_log;Path_log
    """
    if 'chunksize' in kwargs and kwargs['chunksize'] is not None:
        chunksize = kwargs['chunksize']
    else:
        chunksize = ceil(len(individuals) / core_num)
    t1 = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
    t0 = datetime.datetime.now()
    file_ = open(r'{}'.format(save_dir), 'w')
    file_.close()
    if core_num > mp.cpu_count():
        print('-' * 10, 'Please assign core_num < {}'.format(mp.cpu_count()), '-' * 10)
        core_num = eval(input('Core_num = '))
        print(f'Core_num is changed to {core_num}')
    print('-' * 30)
    print('-'*10, 'Preparing', '-'*10)
    print(f'Number of individuals : {len(individuals)}')
    print(f'CPU number in use: {core_num}')
    print(f'Results will be saved to {save_dir}')
    print(' ' * 5, "-----Let's go!-----")
    pool = mp.Pool(core_num)
    print('-'*10, 'Pooling', '-'*10)
    with mp.Manager() as manager:
        lock = manager.Lock()
        print('-'*10, 'Creating Lock', '-'*10)
        pool.imap_unordered(Ind_run, [(ind, save_dir, lock) for ind in individuals], chunksize=chunksize)     #
        pool.close()
        pool.join()
    print('-'*30)
    print('Please waite ......')
    time.sleep(5)
    t2 = datetime.datetime.now()
    print(f'Case {save_dir} started at {t1}')
    print('ends at ', t2.strftime('%Y-%m-%d %H:%M:%S'))
    print('Total time consumption ', t2 - t0)
    print('-'*30)


def PDA_run_concurrent(individuals: list or set, save_dir: str, core_num=1, **kwargs):
    """
    This function controls the model to run.
    @param individuals:
        The list or set of instances of individuals.
    @param save_dir:
        The directory to save result file.
    @param core_num:
        It specifies the CPU number to be involved in multiprocessing
    @param kwargs:
        chunksize:
            It controls the number assigned to each CPU for each time in multiprocessing. Same as chunksize in multiprocessing.pool.imap
    Note:
        The output file will be always in .txt format. The information for an individual will be saved in one line. Different information
        is separated by ';'. The information include:
        ID;t;birth_hab;birth_loc;step_num;Success;Reason;Tar_log;Path_log
    """
    if 'chunksize' in kwargs and kwargs['chunksize'] is not None:
        chunksize = kwargs['chunksize']
    else:
        chunksize = ceil(len(individuals) / core_num)
    t1 = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
    t0 = datetime.datetime.now()
    file_ = open(r'{}'.format(save_dir), 'w')
    file_.close()
    if core_num > mp.cpu_count():
        print('-' * 10, 'Please assign core_num < {}'.format(mp.cpu_count()), '-' * 10)
        core_num = eval(input('Core_num = '))
        print(f'Core_num is changed to {core_num}')
    print('-' * 30)
    print('-'*10, 'Preparing', '-'*10)
    print(f'Number of individuals : {len(individuals)}')
    print(f'CPU number in use: {core_num}')
    print(f'Results will be saved to {save_dir}')
    print(' ' * 5, "-----Let's go!-----")
    with ProcessPoolExecutor(core_num) as Executor:
        manager = mp.Manager()
        lock = manager.Lock()
        Executor.map(Ind_run, [(ind, save_dir, lock) for ind in individuals], chunksize=chunksize)
    print('-'*30)
    print('Please waite ......')
    time.sleep(5)
    t2 = datetime.datetime.now()
    print(f'Case {save_dir} started at {t1}')
    print('ends at ', t2.strftime('%Y-%m-%d %H:%M:%S'))
    print('Total time consumption ', t2 - t0)
    print('-'*30)


def PDA_run_single(individuals: list or set, save_dir: str, core_num=1):
    """
        This function controls the model to run using a single CPU core. Typically used for testing or running a small number of instances.
        @param individuals:
            The list or set consisting of instances of individuals.
        @param save_dir:
            The directory to save result file.
        @param core_num:
            It specifies the CPU number to be involved in multiprocessing
        Note:
            The output file will be always in .txt format. The information for an individual will be saved in one line. Different information
            is separated by ';'. The information include:
            ID;t;birth_hab;birth_loc;step_num;Success;Reason;Tar_log;Path_log

        """
    t1 = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
    t0 = datetime.datetime.now()
    file_ = open(r'{}'.format(save_dir), 'w')
    file_.close()
    if core_num > mp.cpu_count():
        print('-' * 10, 'Please assign core_num < {}'.format(mp.cpu_count()), '-' * 10)
        core_num = eval(input('Core_num = '))
        print(f'Core_num is changed to {core_num}')
    print('-' * 30)
    print('-' * 10, 'Preparing', '-' * 10)
    print(f'Number of individuals : {len(individuals)}')
    print(f'CPU number in use: {core_num}')
    print(f'Results will be saved to {save_dir}')
    print(' ' * 5, "-----Let's go!-----")
    for ind in individuals:
        Ind_run_single((ind, save_dir))
    print('-' * 30)
    print('Please waite ......')
    time.sleep(5)
    t2 = datetime.datetime.now()
    print(f'Case {save_dir} started at {t1}')
    print('ends at ', t2.strftime('%Y-%m-%d %H:%M:%S'))
    print('Total time consumption ', t2 - t0)
    print('-' * 30)


def Ind_run(args):
    """
    This is the function that activates an individual, which corresponds to multiprocessing.
    @param args:
        A tuple like (var1, var2, var3), where var1 is the instance of an individual, var2 is file directory to save results,
        var3 is the process lock that keeps the output information in correct order.
    Note:
        In this version, there is no return because all information is archived in the output file. The instance of an individual will be
        deleted after its information is output.
    """
    ind = args[0]
    res_file = args[1]
    lock = args[2]
    for t in range(ind.iter_T):
        if ind.Success is None:
            ind.action()
        else:
            ind.I_time = t + 1
            break
    lock.acquire()
    with open(r'{}'.format(res_file), 'a') as res_f:
        print(ind.ID, ind.I_time, ind.birth_hab, ind.birth_loc, ind.step_n, ind.Success, ind.Reason, ind.Tar_log, ind.Path_log, sep=';', file=res_f)
    lock.release()
    print(f"Job's done for ID-{ind.ID} in process {os.getpid()}, at iter_T {t + 1}!")
    del ind
    gc.collect()


def Ind_run_single(args):
    """
    This is the function that activates an individual, which corresponds to single core mode.
    @param args:
        A tuple like (var1, var2, var3), where var1 is the instance of an individual, var2 is file directory to save results,
        var3 is the process lock that keeps the output information in correct order.
    Note:
        In this version, there is no return because all information is archived in the output file. The instance of an individual will be
        deleted after its information is output.
    """
    ind = args[0]
    res_file = args[1]
    for t in range(ind.iter_T):
        try:
            if ind.Success is None:
                ind.action()
            else:
                ind.I_time = t + 1
                break
        except Exception as e:
            print('=================')
            print('in running', e)
            print(e.args)
            print(traceback.format_exc())
    with open(r'{}'.format(res_file), 'a') as res_f:
        print(ind.ID, ind.I_time, ind.birth_hab, ind.birth_loc, ind.step_n, ind.Success, ind.Reason, ind.Tar_log, ind.Path_log, sep=';', file=res_f)
    print(f"Job's done for ID-{ind.ID} in process {os.getpid()}, at iter_T {t}!")


def distance_point_to_line(point: tuple, line_point1: tuple, line_point2: tuple):
    """
    Calculate the distance from point to the line connecting line_point1 and line_point2
    @param point:
        The point to calculate distance.
    @param line_point1:
        One point in the line.
    @param line_point2:
        Another point in the line.
    @return:
        The distance from parameter point to the line connecting line_point1 and line_point2
    """
    point, line_point1, line_point2 = np.array(point), np.array(line_point1), np.array(line_point2)
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance


def distance(line_point1: tuple, line_point2: tuple):
    """
    The distance between two point.
    @param line_point1:
        One point.
    @param line_point2:
        The other point.
    @return:
        Distance.
    """
    return np.linalg.norm(np.array(line_point2) - np.array(line_point1))


def NeighborLocs(**kwargs) -> set:
    """
    This function can identify neighbor locations for the input grid cell with respect to the specified neighbor principle.
    @param kwargs:
        position: position=(x1, y1)
            The center location to find neighbor locations for.
        principle: principle='4' or '8'
            The neighbor principle to find neighbor locations.
    @return:
        A set containing the neighbor locations for the input position, NOT including the input position.
    Note:
        This function cannot detect whether one grid cell was a background one.
    """
    if 'position' in kwargs:
        position = kwargs['position']
    if 'principle' in kwargs:
        principle = kwargs['principle']
    x, y = position
    if principle == '4':
        neighbors = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}
    if principle == '8':
        neighbors = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y + 1), (x - 1, y - 1), (x + 1, y + 1),
                     (x + 1, y - 1)}
    return neighbors


def GatherHabs(cells_to_conjunct, habitat_seed=None) -> set:
    """
    This function can find adjacent grid cells according to the input variable.
    @param cells_to_conjunct:
        The set/list of grid cells to find adjacent grid cells.
    @param habitat_seed:
        By default, it is None, which implies a random search. If specified, it will find habitat containing the habitat_seed=(x, y).
    @return:
        The set consisting of adjacent grid cells.
    Note:
        There may be a plenty of grid cells in the input variable. These grid cells may be merged into many habitats.
        This function will only RANDOMLY output ONE habitat.
        If the user wants to find the grid cells belonging to the same habitat as specific grid cell, the user should specify the param: habitat_seed.
    """
    if type(cells_to_conjunct) is list:
        cells_to_conjunct = set(cells_to_conjunct)
    if habitat_seed is not None:
        position = habitat_seed
    else:
        position = list(cells_to_conjunct)[0]
    candidate_cells = {position}
    Cellsinonehab = {position}
    for _ in range(len(cells_to_conjunct) + 1):
        if len(candidate_cells) == 0:
            return Cellsinonehab
        else:
            if position in candidate_cells:
                neighbors = NeighborLocs(position=position, principle='4').intersection(cells_to_conjunct)
                if len(neighbors) == 0:
                    Cellsinonehab.update([position])
                    return Cellsinonehab
                else:
                    candidate_cells.remove(position)
                    candidate_cells.update(neighbors)
                    Cellsinonehab.update([position])
            else:
                center = candidate_cells.pop()
                neighbors = NeighborLocs(position=center, principle='4').intersection(cells_to_conjunct)
                neighbors.difference_update(Cellsinonehab)
                candidate_cells.update(neighbors)
                candidate_cells.intersection_update(cells_to_conjunct)
                Cellsinonehab.update([center])
    return Cellsinonehab


def dot_product_angle(v1: tuple, v2: tuple):
    """
    Calculate the dot product angle for the two input vectors.
    @param v1:
        Vector 1: (x1, y1).
    @param v2:
        Vector 2: (x2, y2).
    @return:
        The angle between the two input vectors in degree.
    Note:
        Due to the calculation of the float number in a computer, there are some cases where the cosine value exceeds [-1, 1]. This can raise
        error in calculating the angle for the cosine value. Therefore, we regulate the value as -0.99999999 and 0.99999999 if they are out of range.
    """
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        pass
    else:
        vector_dot_product = np.dot(v1, v2)
        cos_value = vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cos_value > 1:
            cos_value = 0.99999999
        elif cos_value < -1:
            cos_value = -0.99999999
        arccos = np.arccos(cos_value)
        return arccos


def cosine_bias_rate(v1: tuple, v2: tuple):
    """
    It calculates the cosine bias rate for the two input vectors.
    @param v1:
        Vector 1: (x1, y1).
    @param v2:
        Vector 2: (x2, y2).
    @return:
        The corresponding cosine bias rate for the input vectors.
    Note:
        This function is used to calculate the cosine bias rate. In this version, the cosine bias rate (CosBR) is set the form
        CosBR = (cosine<v1, v2> + 1)/2, where cosine<v1, v2> is the cosine value between vectors v1 and v2. The form of directional bias
        can be further modified according to particular species, ecological process or search goals.
    """
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        pass
    else:
        vector_dot_product = np.dot(v1, v2)
        cos_value = vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cos_value >= 1:
            cos_value = 0.99999999
        elif cos_value <= -1:
            cos_value = -0.99999999
        cosine_bias = (cos_value + 1) / 2
        return cosine_bias


def plaint_cost_to_prob(cells_cost: dict) -> dict:
    """
    It assigns each grid cell a probability according to the input dict.
    @param cells_cost:
        The dict in form cells_cost={(x1, y1): 20, (x2, y2): 100}
    @return:
        A dict that pairs the grid cell and its probability to be selected in terms of its cost.
    Note:
        In this version, we assume the probability is inversely proportional to the cost of a grid cell, which is regulated with respect
        to the sum of cost of all grid cells. It can be modified according to particular species, ecological process or search goals.
    """
    cells_prob = {}
    if len(cells_cost) == 1:
        cells_prob = {k: 1 for k in cells_cost.keys()}
    else:
        cost_sum = sum(cells_cost.values())
        for cell in cells_cost:
            cells_prob[cell] = 1 - cells_cost[cell] / cost_sum
    return cells_prob


def clockwise_angle(target_vector: tuple, base_vector: tuple):
    """
    It calculates the clockwise angle for the target_vector based on the base_vector.
    @param target_vector:
        The vector that is to be evaluated.
    @param base_vector:
        The vector serves as the benchmark to calculate the clockwise angle for the angle_vector.
    @return:
        The clockwise angle in degree.
    """
    x1, y1 = target_vector
    x2, y2 = base_vector
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    return theta * 180 / np.pi


def coefficient_of_variation(data: list or np.array):
    """
    It calculates the coefficient of variation for the input data.
    @param data:
        The input data should be a list or a 1-D np.array.
    @return:
        The coefficient of variation.
    """
    mean = np.mean(data)
    std = np.std(data, ddof=0)
    cv = std / mean
    return cv


def turn_loc_detect(path: list, **kwargs) -> list:
    """
    It is used to find the grid cell that leads a vector to turn larger than a specified minimum turn angle.
    @param path:
        A list of grid cells indicating a consecutive movement path from the start to the end of the list.
    @param kwargs:
        min_angle: min_angle=10
            This argument specifies the minimum turn angle. Only grid cells that lead to turn angles larger than min_angle will be regarded
            as a turn loc.
    @return:
        A list of turn locs. The first is always the start of the path.
    """
    turn_loc = []
    if 'min_angle' in kwargs and kwargs['min_angle'] is not None:
        min_angle = kwargs['min_angle']
        path_ = copy.deepcopy(path)
        k = 2
        pointer = 0
        while k < len(path_):
            x0, y0 = path_[0]
            x1, y1 = path_[1]
            x2, y2 = path_[k]
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x0, y2 - y0)
            if pointer == 'mid_stage':
                theta = clockwise_angle(v2, v_base)
            else:
                theta = clockwise_angle(v2, v1)
            if theta > 180:
                theta = 360 - theta
            if theta < min_angle:
                k += 1
            else:
                turn_loc.append(path_[k])
                v_base = (x2 - x0, y2 - y0)
                path_ = path_[k:]
                pointer = 'mid_stage'
                k = 1
    else:
        for k, loc in enumerate(path[2:]):
            x0, y0 = path[k]
            x1, y1 = path[k + 1]
            x2, y2 = loc
            v0 = (x2 - x0, y2 - y0)
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x1, y2 - y1)
            theta0 = dot_product_angle(v1, v0)
            theta1 = dot_product_angle(v2, v0)
            theta2 = dot_product_angle(v2, v1)
            if theta0 == theta1 == theta2:
                pass
            else:
                turn_loc.append(path[k + 1])
    return turn_loc


def Sinuosity_Index(path: list, min_angle: float or int, turn_loc_list=None):
    """
    This function is used to calculate the sinuosity index for the input movement path under specified minimum turn angle.
    @param path:
        A list of consecutive locations for a path, like [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3)].
    @param min_angle:
        The specified minimum turn angle. It will be transferred to function turn_loc_detect(path, min_angle=min_angle)
    @param turn_loc_list:
        If turn_loc_list is specified, the sinuosity index will be calculated without calling function turn_loc_detect(path, min_angle=min_angle).
    @return:
        The sinuosity index for the input path.
    """
    if turn_loc_list is None:
        min_angle_ = min_angle
        turn_locs = turn_loc_detect(path, min_angle=min_angle_)
    else:
        turn_locs = turn_loc_list
    if len(turn_locs) == 0:
        SI = 0
    else:
        base_vector = (path[-1][0] - path[0][0], path[-1][1] - path[0][1])
        vectors_ = []
        vector_length = []
        append_vectors_ = vectors_.append
        append_vector_length = vector_length.append
        for loc_index, turn_loc in enumerate(turn_locs[1:]):
            _x, _y = turn_locs[loc_index][0] - turn_loc[0], turn_locs[loc_index][1] - turn_loc[1]
            append_vectors_((_x, _y))
            append_vector_length(sqrt(_x ** 2 + _y ** 2))
        append_vectors_((turn_locs[0][0] - path[0][0], turn_locs[0][1] - path[0][1]))
        append_vector_length(sqrt(vectors_[-1][0] ** 2 + vectors_[-1][1] ** 2))
        p = np.mean(vector_length)
        turn_angles = []
        append_turn_angles = turn_angles.append
        for tar_vector in vectors_:
            append_turn_angles(round(float(clockwise_angle(tar_vector, base_vector)), 3))
        c, s = np.mean(np.cos(turn_angles)), np.mean(np.sin(turn_angles))
        b = coefficient_of_variation(vector_length)
        base_ = p * ((1 - c ** 2 - s ** 2) / ((1 - c) ** 2 + s ** 2) + b ** 2)
        if base_ < 10e-10:
            SI = 0
        else:
            SI = 2 / sqrt(base_)
    return SI


def add_graph_from_edge(edge: set) -> dict:
    """
    This function generates a formatted graph for Dijkstra algorithm.
    @param edge:
        The edge list/set, like [(node1, node2, weight), ...]/{(node1, node2, weight), ...}.
    @return:
        A formatted graph is a graph described by a dictionary, like {node1: {node2: weight}, ...}. It points out the target nodes directly connected to
        a source node with their weights.
    """
    formatted_graph = {}
    for pairs in edge:
        formatted_graph.setdefault(pairs[0], {})
        formatted_graph[pairs[0]][pairs[1]] = pairs[2]
    return formatted_graph


def last_hop_for_path(last_hop_dict: dict, node_num: int, source, target) -> list:
    """
    It will randomly generate one least cost path in all least cost paths according to the rewrite_Dijkstra_path().
    @param last_hop_dict:
        The dict stores the last node(s) that is in the least cost path, like {node1: [node2, node3], ...}.
        Note that in {node1: [node2, node3], ...}, nodes in the value list for key node1 are the last nodes in the least cost paths.
    @param node_num:
        The maximum number of nodes in the graph, or it can be an arbitrary number larger than the largest node number in all least cost paths.
    @param source:
        The start of path, i.e., the location where the individual is.
    @param target:
        The target of path, i.e., the destination for the individual to move.
    @return:
        A randomly selected least cost path in all least cost paths from source to target.
    """
    path = [target]
    for i in range(node_num):
        if source not in path:
            step = rd.choice(last_hop_dict[target])
            path.append(step)
            target = step
        else:
            break
    path.reverse()
    return path


def find_start(next_start: dict) -> list:
    """
    It generates a list consisting of all possible nodes to be selected for the next start node in Dijkstra algorithm.
    @param next_start:
        All possible nodes that have node been selected as a start in Dijkstra algorithm.
    @return:
        A list/generator that contains nodes: 1. have not been selected as a start and 2. shortest from the source to current node.
    """
    min_distance = min(next_start.values())
    return [vertex for vertex, distance in next_start.items() if distance == min_distance]


def rewrite_Dijkstra_path(formatted_graph: dict, source, target) -> list:
    """
    A modified Dijkstra algorithm to randomly generate one path out of all possible shortest cost paths.
    @param formatted_graph:
        The graph that is converted into suitable format to find the least cost path.
    @param source:
        The source of a shortest path.
    @param target:
        The target of a shortest path.
    @return:
        A randomly selected shortest path out of all possible shortest paths.
    """
    d_dict = {s: float('inf') for s in formatted_graph}
    book = list(formatted_graph.keys())
    start = source
    d_dict[start] = 0
    last_hop = {s: [] for s in formatted_graph}
    for _ in range(len(d_dict)):
        book.remove(start)
        for step_end, step_weight in formatted_graph[start].items():
            if step_weight + d_dict[start] < d_dict[step_end]:
                d_dict[step_end] = step_weight + d_dict[start]
                last_hop[step_end] = [start]
            elif step_weight + d_dict[start] == d_dict[step_end]:
                d_dict[step_end] = step_weight + d_dict[start]
                last_hop[step_end].append(start)
        next_start = {vertex: d_dict[vertex] for vertex in book}
        if len(next_start) != 0:
            _start = find_start(next_start)
            if len(_start) == 1:
                start = _start[0]
            elif len(_start) > 1:
                start = rd.choice(_start)
    path = last_hop_for_path(last_hop, len(d_dict), source, target)
    return path


def save_mat_files(input_mat: np.array, save_dir, file_name):
    """
    An auxiliary function to save np.array. The input data will be saved in .txt format.
    @param input_mat:
        The input data in np.array format.
    @param save_dir:
        The path to save the input data.
    @param file_name:
        The file name for the input data.
    """
    file_ = open(r'{}/{}'.format(save_dir, file_name), 'w')
    for i in input_mat:
        i.tolist()
        print(" ".join(repr(e) for e in i), file=file_)
    file_.close()


class BackgroundSetting:
    """
    This is the background setting class. It is used to do some preliminary treatment on the spatial-temporal variables.
    In this version, it is used to deal with cost surface and habitat surface.
    """
    def __init__(self, cost_surface, habitat_surface, **kwargs):
        """
        Here sets the attribute for a background instance. A background instance is the spatial-temporal entities for individuals to
        interact with. It can be shown in various forms, including a single cost surface or a series of temporally changing spatial entities.
        It can be modified from case to case. Specifically in this version, the attributes are built depending on the cost surface and habitat surface.
        @param cost_surface:
            A cost surface can be a file directory pointing out the cost surface file, or, a 2-D np.array.
        @param habitat_surface:
            A habitat surface indicates habitats in the input cost surface. Grid cells belonging to one same habitat should share one same value and they
            are required to be adjacent from one to another.
        @param kwargs:
            background_value: background_value=1234
                It specifies the value to be treated as the background. By default, the background value is -9999.
                A background will NOT be included anywhere.
            detect_habitat: detect_habitat=True/False
                It tells the program whether to treat adjacent grid cells with respect to their spatial arrangement instead of their value.
            recoding: recoding=True/False
                It controls whether the identified habitats to be recoded. Please refer to method self.HabIndex_Locs() for details.
        Note:
            There are also other attributes:
                self.raw_habid stores the raw value for each habitat.
                self.HabIndex_Locs is a dict shows the habitat ID and the belonging grid cells, like {1:[(x1, y1), (x2, y2)], 2:[(x3, y3)]}.
                self.HabLocs is a set consisting of all habitat grid cells.
                self.AllCellsLocs is a set consisting of all grid cells whose value is NOT background value.
                self.HabIndex_BoarderLocs is a dict shows the habitat ID and the boarder grid cells of the habitat, like {1:[(x1, y1), (x2, y2)], 2:[(x3, y3)]}.
                self.cost_surface is a dict with keys of each location in self.AllCellsLocs and value of the corresponding cost value in self.cost_surface_
            Please mind that after calling self.HabIndex_Locs, the archived self.habitat_surface will be changed
            according to self.detect_habitat and self.recoding.
            Note that, self.cost_surface_ is the cost surface loaded in np.array format; while self.cost_value is the cost surface converted to a dict format.
        """
        print('-'*10, 'Setting Background', '-'*10)
        if type(cost_surface) == str:
            self.cost_surface_ = np.loadtxt(cost_surface, skiprows=6, dtype=float)  # for real use
        elif type(cost_surface) == np.ndarray:
            self.cost_surface_ = cost_surface
        if type(habitat_surface) == str:
            self.habitat_surface = np.loadtxt(habitat_surface, skiprows=6)  # for real use
        elif type(habitat_surface) == np.ndarray:
            self.habitat_surface = habitat_surface
        if 'background_value' in kwargs and kwargs['background_value'] is not None:
            self.background_value = kwargs['background_value']
        else:
            self.background_value = -9999
        if 'detect_habitat' in kwargs:
            self.detect_habitat = kwargs['detect_habitat']
        else:
            self.detect_habitat = False
        if 'recoding' in kwargs:
            self.recoding = kwargs['recoding']
        else:
            self.recoding = False
        self.cost_surface = {}
        legal_locs = np.where(self.cost_surface_ != -9999)
        legal_locs = set(list(zip(legal_locs[0], legal_locs[1])))
        for locs in legal_locs:
            self.cost_surface[locs] = self.cost_surface_[locs]
        self.AllCellsLocs = self.AllCellsLocs_()
        self.HabLocs = self.HabLocs_()
        self.raw_habid = np.unique(self.habitat_surface)[1:].tolist()
        self.HabIndex_Locs = self.HabIndex_Locs_()     # self.habitat_surface has been changed after self.Habitat_identify() functioning
        self.HabIndex_BoarderLocs = self.BoarderDetect()

    def get_HabIndex_BoarderLocs(self):
        return self.HabIndex_BoarderLocs

    def get_HabIndex_Locs(self, return_index=False):
        if return_index:
            return self.raw_habid, self.HabIndex_Locs
        else:
            return self.HabIndex_Locs

    def get_HabLocs(self):
        return self.HabLocs

    def HabIndex_Locs_(self):
        """
        It is used to match the habitat id with corresponding grid cells.
        @return:
            A dict where the key is the id of habitat and the values is a list consisting of grid cells in that habitat, like {1:[(x1, y1), (x2, y2)]}.
        Note:
            The attribute in the BackgroundSetting class will influence the functioning of this method.
            If self.detect_habitat is True, it will automatically treat adjacent grid cells as one habitat, REGARDLESS of their value.
            If self.recoding is False in this condition, each habitat will be given a random ID starting from 0; if self.recoding is True,
            the identified will be sorted by their area (number of grid cells) and a habitat ID will be given sequentially from 0 for the
            largest habitat.
            If self.detect_habitat is False, it will treat grid cells with same value as one habitat. If self.recoding is False at the same time,
            the habitat ID will be value of the grid cells; if self.recoding is True, the habitat ID will given from 0 in the sequence given by the
            value of grid cells.
            Anyway, self.habitat_surface will be updated according to the new code.
            ONLY when self.detect_habitat is False AND self.recoding is False can result in an unchanged habitat surface in terms of the raw input.
        """

        HabLocs = self.HabLocs
        HabIndex_Locs = {}
        if self.detect_habitat is True:
            print('-' * 10, 'Running #-Habitat_detect-#', '-' * 10)
            HabIndex_Locs_ = {}
            hab_id = 0
            for _ in range(len(HabLocs) + 1):
                if len(HabLocs) > 0:
                    hablocs = GatherHabs(HabLocs)
                    HabIndex_Locs_[hab_id] = hablocs
                    hab_id += 1
                    HabLocs = set(HabLocs) - hablocs
                else:
                    break
            if self.recoding is False:
                print('-' * 10, '        #-No recoding-#', '-' * 10)
                return HabIndex_Locs_
            else:
                print('-' * 10, 'Running #-Recoding Habitats-#', '-' * 10)
                re_indexing = {}
                for hab_id, hab_locs in HabIndex_Locs_.items():
                    hab_area_gridcells = len(hab_locs)
                    if hab_area_gridcells in re_indexing.values():
                        hab_area_gridcells += 1
                    re_indexing[hab_id] = hab_area_gridcells
                re_indexing = dict(sorted(re_indexing.items(), key=lambda x: x[0]))
                re_indexing = dict(sorted(re_indexing.items(), reverse=True, key=lambda x: x[1]))
                for sorted_id in range(len(re_indexing.keys())):
                    HabIndex_Locs[sorted_id] = HabIndex_Locs_[list(re_indexing.keys())[sorted_id]]
        else:
            print('-' * 10, '        #-No Habitat_detect is specified-#', '-' * 10)
            raw_hab_id = np.unique(self.habitat_surface)[1:].tolist()
            for hab_id in raw_hab_id:
                hab_x, hab_y = np.where(self.habitat_surface == hab_id)
                if self.recoding is True:
                    hab_id = raw_hab_id.index(hab_id)
                HabIndex_Locs[hab_id] = set(list(zip(hab_x, hab_y)))
            if self.recoding is False:
                print('-' * 10, '        #-No recoding is specified-#', '-' * 10)
                return HabIndex_Locs
        for hab_id, hab_locs in HabIndex_Locs.items():
            for locs in hab_locs:
                self.habitat_surface[locs] = hab_id
        return HabIndex_Locs

    def HabLocs_(self):
        """
        It is used to find all habitat grid cells whose value is NOT the background value.
        @return:
            A set that consist of locations of grid cells, like {(x1, y1), (x2, y2)}
        """
        print('-'*10, 'Running #-HabLocs-#', '-'*10)
        hab_x, hab_y = np.where(self.habitat_surface != self.background_value)
        return set(list(zip(hab_x, hab_y))).intersection(self.AllCellsLocs)

    def AllCellsLocs_(self):
        """
        It is used to find all available grid cells whose value is NOT the background value.
        @return:
            A list that consist of locations of grid cells, like [(x1, y1), (x2, y2)]
        """
        print('-'*10, 'Running #-AllCellsLocs-#', '-'*10)
        x, y = np.where(self.cost_surface_ != self.background_value)
        return set(list(zip(x, y)))

    def BoarderDetect(self):
        """
        Based on 4-neighbor principle, it identifies the border grid cells of each habitat patch.
        @return:
            The dict like {1:{(x1, y1), (x2, y2)}} where the key of the dict, 1, is the code of habitat,
            and the values of the key is a set consisting of grid cells locating at the boarder of the habitat.
        """
        print('-'*10, 'Running #-BoarderDetect-#', '-'*10)
        AllCellsLocs = set(self.AllCellsLocs)
        HabIndex_Locs = self.HabIndex_Locs
        HabIndex_BoarderLocs = {}
        for HabitatIndex, Locs in tqdm(HabIndex_Locs.items(), desc='Boarder detect'):
            HabIndex_BoarderLocs.setdefault(HabitatIndex, set())
            Locs = set(Locs)
            for one_loc in Locs:
                neighbors = NeighborLocs(position=one_loc, principle='4').intersection(AllCellsLocs)
                neighborsInHabitat = neighbors.intersection(Locs)
                if len(neighbors) == 4 and len(neighborsInHabitat) < 4:
                    HabIndex_BoarderLocs[HabitatIndex].update([one_loc])
                elif len(neighbors) == 3 and len(neighborsInHabitat) < 3:
                    HabIndex_BoarderLocs[HabitatIndex].update([one_loc])
                elif len(neighborsInHabitat) < 2:
                    HabIndex_BoarderLocs[HabitatIndex].update([one_loc])
        return HabIndex_BoarderLocs

    def setting(self):
        """
        It matches each required background variable to its keyword.
        @return:
            A dict that contains the required background variables for individuals to request upon keywords.
        Note:
            The keyword must be in accordance to those keywords used by an individual. The keywords serves as a tag for individuals to find
            corresponding correct information.

            If there is a need to create dynamic background variables, the user only needs to correspond the time stamp with the series of
            background variables.
            For example, the below ways can be a reference:

                background_vars_t0 = BackgroundSetting(cost_surface_t0, habitat_surface_t0, **kwargs_t0).setting_static()
                background_vars_t20 = BackgroundSetting(cost_surface_t20, habitat_surface_t20, **kwargs_t1).setting_static()
                t0 = 0
                t20 = 20
                time_series = [0, 20]
                background_vars_series = [background_vars_t0, background_vars_t20]
                ^^^Way1^^^:
                dynamic_background_vars = {t0: background_vars_t0, t20: background_vars_t20}
                ^^^Way2^^^:
                dynamic_background_vars = {}
                for time_stamp_index, time_stamp in enumerate(time_series):
                    dynamic_background_vars[time_stamp] = background_vars_series[time_stamp_index]
                ^^^Way3^^^:
                dynamic_background_vars = {time_stamp: background_vars_series[time_stamp_index] for time_stamp_index, time_stamp in enumerate(time_series)]}
            Way2 and Way3 are useful when there is a need to create many time series background variables.
            Way1 and Way2 will directly generate a dict, whereas Way3 will generate a Python Iterator that may reduce RAM requirement.
            A function Dynamic_background_setting(time_stamp_list, background_vars_list) can also be used to create such dynamic background.
        """
        return {'cost_surface': self.cost_surface, 'AllCellsLocs': self.AllCellsLocs,
                'HabLocs': self.HabLocs, 'HabIndex_Locs': self.HabIndex_Locs, 'HabIndex_BoarderLocs': self.HabIndex_BoarderLocs}


class Individual:
    """
        This is used to create the instance for individuals. The PDA processes are involved in this class, namely, an individual will contain
        PDA processes. There can be one universal PDA process or different PDA processes in accordance to the state of an individual.
    """
    def __init__(self, ID, MDT, Sig, PR, iter_T, birth_hab, birth_loc, resolution, **kwargs):
        """
        Here sets the attributes for an individual. Attributes are inherent for individuals.
        An individual will have below attributes that may be involved in its PDA processes during specific ecological processes.
        By modifying attributes and processes an individual that may have distinct behaviors.
        @param ID:
            The ID to characterise an individual.
        @param MDT:
            The Maximum Distance to Traverse, here refers to the cost-weighted distance.
        @param Sig:
            The capability to get across impediments.
        @param PR:
            The Perception Range delineates a range within which grid cells can be perceived by an individual.
        @param iter_T:
            The time for an individual to run PDA processes.
        @param birth_hab:
            The habitat where an individual is born. Specifically, if a random_birth mode is applied,
            the value will be assigned -1.
        @param birth_loc:
            The location of a grid cell where an individual is born, stored in a list.
        @param resolution:
            The resolution used in geographical calculation for an individual.
        @param kwargs:
            Other possible variables that will be transferred.
            These variables or attributes should be specified as neighbor_principle='4'.
            In this version, they include:
                cost_surface:
                    The cost surface converted to a dict.
                AllCellsLocs:
                    A set of locations of all pixels in the study area in a set.
                HabLocs:
                    A set of locations of all pixels of habitat.
                HabIndex_Locs:
                    The dict consisting of habitat index as keys and the set of locations of pixels of the habitat as the value.
                hab_boarder:
                    The locations of boarder pixels of a habitat. It is used to relocate individuals if an individual was trapped in pixels.
                    surrounded by its natal habitat.
                cost_surface_shape:     (4, 4)
                    The shape of the cost surface matrix.
                neighbor_principle:     neighbor_principle='4' or '8'
                    Specifies the neighbor principle for an individual.
                attenuate_coefficient:      attenuate_coefficient=0.05 or 'default'
                    Assigns the attenuate coefficient for the attenuation of perception.
                    By 'default', it is 0.01.
                perception_mode:    perception_mode='attenuate'
                    It specifies the perception of an individual to attenuate with respect to the attenuate coefficient.
                direction_bias_mode:    direction_bias_mode='cosine' or None
                    It specifies the form of direction bias for an individual.
        Note:
            There are also other attributes for an individual like:
                self.Success that indicates whether an individual has succeeded in dispersal,
                self.Reason documents the reason of the results for particular state or process where in this exemplar case it shows the reasons for
                    the termination of movement,
                self.I_time documents the time for an individual to conduct PDA processes,
                self.Puzzle indicates the state of an individual for different perception mode,
            The user can also develop other attribute like self.Stage for an individual to indicate the ecological process during dispersal,
            i.e., emigration, transfer and immigration, or an attribute like self.Decision_chain for an individual's decision process in a Markov chain.
            What attributes will an individual capture should depend on traits of individuals, particular search goals, as well as the associated
            Perception-Decision-Action processes.
        """
        self.Success = None
        self.Reason = None
        self.Puzzle = False
        self.I_time = None
        self.ID = ID
        self.MDT = MDT
        self.Sig = Sig
        self.PR = ceil(PR / resolution)
        self.iter_T = iter_T
        self.birth_hab = birth_hab
        self.birth_loc = [birth_loc]
        self.position = birth_loc
        self.resolution = resolution
        self.cost_surface = kwargs['cost_surface']
        self.AllCellsLocs = kwargs['AllCellsLocs']
        self.HabLocs = kwargs['HabLocs']
        self.HabIndex_Locs = kwargs['HabIndex_Locs']
        self.boarder = kwargs['hab_boarder']
        self.cost_surface_shape = kwargs['cost_surface_shape']
        self.Path_log = [self.position]    # add self.position to the first location in the individual's Path_log and Tar_log
        self.Tar_log = [self.position]
        self.step_n = None
        if 'neighbor_principle' in kwargs:
            self.principle = kwargs['neighbor_principle']
        else:
            self.principle = '4'
        if 'attenuate_coefficient' in kwargs and type(kwargs['attenuate_coefficient']) is float:
            self.attenuate_coefficient = kwargs['attenuate_coefficient']
        elif 'attenuate_coefficient' in kwargs and kwargs['attenuate_coefficient'] == 'default':
            self.attenuate_coefficient = 0.01
        else:
            self.attenuate_coefficient = None
        if 'perception_mode' in kwargs:
            self.perception_mode = kwargs['perception_mode']
        else:
            self.perception_mode = None
        if 'direction_bias_mode' in kwargs:
            self.direction_bias = kwargs['direction_bias_mode']
        else:
            self.direction_bias = None

    def get_Success(self):
        return self.Success

    def get_Reason(self):
        return self.Reason

    def get_ID(self):
        return self.ID

    def get_I_time(self):
        return self.I_time

    def get_MDT(self):
        return self.MDT

    def get_Sig(self):
        return self.Sig

    def get_PR(self):
        return self.PR

    def get_iter_T(self):
        return self.iter_T

    def get_Attenuate_coefficient(self):
        return self.attenuate_coefficient

    def get_perception_mode(self):
        return self.perception_mode

    def get_direction_bia(self):
        return self.direction_bias

    def get_birth_hab(self):
        return self.birth_hab

    def get_birth_loc(self):
        return self.birth_loc

    def get_neighbor_principle(self):
        return self.principle

    def get_Position(self):
        return self.position

    def get_Pathlog(self):
        return self.Path_log

    def get_Tarlog(self):
        return self.Tar_log

    def perception(self) -> Union[dict, int]:  # returns a dict={loc: prob}, where excluding self.position
        """
        The perception process for an individual. It provides a redistribution kernel as the raw material for its decision process.
        @return:
            The redistribution kernel like cells_prob={(x1, y1): 0.1, (x2, y2): 0.7}.
        """
        CTL = self.CellsInPR(puzzle=self.Puzzle)  # by puzzle=self.Puzzle, CTLs will be identified regardless of sigma
        if len(CTL) == 0:     # if no available non-natal pixels in PR
            return 0            # it returns a null RK
        else:                                               # if there is non-natal pixel in PR
            conjunct_cells_in_pr = self.IsConjunct(CTL)   # check if CTLs are not reachable due to blocked by natal pixels
            conjunct_cells_in_pr.discard(self.position)     # remove self.position from the reachable pixels
            if len(conjunct_cells_in_pr) == 0:      # if no non-natal pixel is reachable from the individual,
                return 0                            # it returns a null RK
            else:                                                  # if there is reachable non-natal pixel,
                RK = self.format_probs(conjunct_cells_in_pr)    # calculate the redistribution kernel
                return RK                                       # return the dict containing pixel locations and probability, RK

    def decision(self) -> Union[list, int]:
        """
        The decision process for an individual.
        In this exemplar case, an individual will randomly select a TL by preferential selection. If all CTLs have been selected as TLs,
        the individual will fall into a puzzle state.
        @return:
            A path for an individual to move, which will be traversed in action process. The start of the path
            is where the individual is and the end of the path is TL.
        Note:
            In this version, an individual is designed to select target location following preferential selection and identify a movement path
            using a modified Dijkstra algorithm. The modified Dijkstra algorithm will randomly generate a path with only the least cost, whereas a classical
            Dijkstra algorithm can only generate a path with the least cost and the shortest path length (minimum number of grid cells).
            It can also be modified according to particular species, ecological processes or search goals.
        """
        re_dk = self.perception()   # the redistribution kernel passed from perception
        if re_dk == 0:    # if there is no available pixel in perception
            if self.Puzzle is False:
                self.Puzzle = True
                return 1
            else:
                return 0      # it returns an indicator
        else:   # if there is at least one reachable pixel in perception
            Tar_log = self.Tar_log
            re_dk_ = {cells: p for cells, p in re_dk.items() if cells not in Tar_log}   # filter CTLs by TL
            if len(re_dk_) == 0:     # if all CTLs have been selected as TL
                return 0             # it returns null path
            else:                                                   # if there is reachable pixel
                if not set(re_dk_).isdisjoint(self.HabLocs):        # if there is at least one pixel of other habitat,
                    tars_ = set(re_dk_).intersection(self.HabLocs)  # find them
                    rd.seed()
                    tar = rd.choice(list(tars_))                    # randomly choose a TL from them
                else:                                                                         # if there is no pixel of other habitat
                    np.random.seed()
                    base_ = sum(re_dk_.values())
                    re_dk_ = {cells: p / base_ for cells, p in re_dk_.items()}
                    tar_index_ = np.random.choice(len(re_dk_), 1, p=list(re_dk_.values()))    # randomly choose one CTL as TL
                    tar = list(re_dk_)[tar_index_[0]]
                Tar_log.append(tar)     # add the TL to self.Tar_log
                re_dk = set(re_dk)
                re_dk.update([self.position])
                G = self.CreateGraph(re_dk)   # create network based on the pixels
                path = rewrite_Dijkstra_path(G, source=self.position, target=self.Tar_log[-1])  # randomly choose one of the LCPs
                return path     # return the movement path to self.action()

    def action(self):
        """
        The action process for an individual.
        When traversing pixels in a cost surface, MDT will be consumed. Death may be incurred when traversing hazardous locations.
        The movement will be archived into self.Path_log and TL will be archived into self.Tar_log. The position of an individual will be updated to
        self.position. The living state of an individual and the corresponding reason of death will also be updated.
        If path in self.action() is 0, the individual will be replaced to another pixel at the boarder of natal habitat.
        Note:
            This is only an exemplar case. It can also be modified according to particular species, ecological processes or search goals.
        """
        path = self.decision()      # the movement path transferred from self.decision
        if path == 1:
            path = self.decision()
        if path == 0:
            self.Replace()
            self.Puzzle = False
        else:       # if the movement path is not empty
            self.Puzzle = False
            HabLoc_check = self.HabLocs.difference(self.HabIndex_Locs[self.birth_hab])
            path_to_go = path[1:]       # the path to move
            append_Path_log = self.Path_log.append
            for loc_ in path_to_go:     # traverse each pixel in the movement path
                append_Path_log(loc_)   # the traversed pixel will be updated in self.Path_log
                self.MDT -= self.cost_surface[loc_] * self.resolution * distance(loc_, self.position)   # traversing a pixel will consume MDT
                self.position = loc_    # update the location of individual
                if self.cost_surface[loc_] > self.Sig:                                 # if hazardous movement
                    rd.seed(time.time())
                    if rd.random() > (self.cost_surface[loc_] - self.Sig) / self.Sig:  # check if alive. if a random number > mortality rate, alive
                        if self.MDT <= 0:                                              # if MDT low, stop moving
                            self.Success = 0                                           # change the success state
                            self.Reason = 'Die for MDT low'                            # describe death reason
                            break                                                      # stop moving
                        if loc_ in HabLoc_check:  # if arrive other habitat, stop moving
                            self.Success = 1
                            self.Reason = 'Successful migration'
                            break
                    else:               # if death occur, stop moving
                        self.Success = 0
                        self.Reason = 'Die for crossing impediment'
                        break
                else:                   # if normal movement
                    if self.MDT <= 0:   # if MDT low, stop moving
                        self.Success = 0
                        self.Reason = 'Die for MDT low'
                        break
                    if loc_ in HabLoc_check:    # if arrive other habitat, stop moving
                        self.Success = 1
                        self.Reason = 'Successful migration'
                        break

    def Replace(self):
        """
        This function replace the individual.
        """
        try:
            self.boarder.remove(self.birth_loc[-1])  # remove the original birth location, i.e., self.birth_loc
            self.position = rd.choice(self.boarder)  # replace the individual
        except IndexError:
            self.Success = -1
            self.Reason = 'Trapped'
        else:
            self.Tar_log.append(self.position)  # add the location after replacement into self.Tar_log
            self.Path_log.append(self.position)  # add the location after replacement into self.Path_log
            self.birth_loc.append(self.position)  # update self.birth_locs to self.position

    def format_probs(self, conjunct_cells_in_pr) -> dict:
        """
        This method generates a redistribution kernel for an individual in perception process.
        @param conjunct_cells_in_pr:
            A set of grid cell locations, like conjunct_cells_in_pr={(x1, y1), (x2, y2)}.
        @return:
            The redistribution kernel for an individual in a dict showing the probability for a grid cell to be selected as a target location,
            like cells_prob={(x1, y1): 0.1, (x2, y2): 0.7}.
        Note:
            This method integrates the results from basic perception about cost values, perception attenuation and directional bias. In other possible
            cases, the memory, the state, the sex, age, etc. of an individual may also have an influence on the consequent redistribution kernel. It can
            be modified according to particular species, ecological process or search goals.
        """
        cost_kernel = self.cost_kernel(conjunct_cells_in_pr)
        RK_base = plaint_cost_to_prob(cost_kernel)
        if self.perception_mode == 'attenuate':
            PA_kernel = self.PA_kernel(conjunct_cells_in_pr)
        else:
            PA_kernel = {}
            for locs in conjunct_cells_in_pr:
                PA_kernel[locs] = 1
        if self.direction_bias == 'cosine':
            DB_kernel = self.DB_kernel(conjunct_cells_in_pr)
        else:
            DB_kernel = {}
            for locs in conjunct_cells_in_pr:
                DB_kernel[locs] = 1
        for cell in RK_base:  # calculate the redistribution kernel
            RK_base[cell] = RK_base[cell] * PA_kernel[cell] * DB_kernel[cell]
        base_ = sum(RK_base.values())
        for cell in RK_base:
            RK_base[cell] = RK_base[cell] / base_
        return RK_base

    def cost_kernel(self, conjunct_cells_in_pr) -> dict:
        """
        This method pairs the grid cells in the input variable with corresponding cost in cost surface.
        @param conjunct_cells_in_pr:
            A set of grid cell locations, like conjunct_cells_in_pr={(x1, y1), (x2, y2)}.
        @return:
            A dict that shows the cost for each grid cell in the input variable, like conjunct_cells_cost={(x1, y1): 20, (x2, y2): 100}
        Note:
            This method only pair the grid cells with their cost.
        """
        conjunct_cells_cost = {}
        cost_surface = self.cost_surface
        for loc in conjunct_cells_in_pr:
            conjunct_cells_cost[loc] = cost_surface[loc]
        return conjunct_cells_cost

    def PA_kernel(self, conjunct_cells_in_pr) -> dict:
        """
        This method calculates the perception attenuation rate according to the input variable.
        @param conjunct_cells_in_pr:
            A set of grid cell locations, like conjunct_cells_in_pr={(x1, y1), (x2, y2)}.
        @return:
            A dict that shows the perception attenuation rate for each grid cell in the input variable, like bias_rate={(x1, y1): 0.1, (x2, y2): 0.7}
        Note:
            In particular, this perception attenuation rate (PAR) is calculated in the form PAR=exp(-a_c * dis<s, t>), where a_c is the attenuation coefficient
            that is an attribute archived in self.attenuate_coefficient; dis<s, t> is the Euclidean distance between a source node (s) and a target node (t).
            The form of perception attenuation can be modified by the user according to particular species, ecological process or search goals.
        """
        attenuate_rate = {}
        x0, y0 = self.position
        a_c = self.attenuate_coefficient
        for loc in conjunct_cells_in_pr:
            dis = sqrt((x0 - loc[0]) ** 2 + (y0 - loc[1]) ** 2)
            attenuate_rate[loc] = exp(-a_c * dis * self.resolution)
        return attenuate_rate  # All conjunct cells in PR, including focal position

    def DB_kernel(self, conjunct_cells_in_pr) -> dict:
        """
        This method calculates the directional bias according to the input variable.
        @param conjunct_cells_in_pr:
            A set of grid cell locations, like conjunct_cells_in_pr={(x1, y1), (x2, y2)}.
        @return:
            A dict that shows the directional bias for each grid cell in the input variable, like bias_rate={(x1, y1): 0.1, (x2, y2): 0.7}
        Note:
            In particular, this directional bias is in cosine form. There are also other forms of directional bias that can be involved, which can
            be further modified according to particular species, ecological process or search goals.
        """
        bias_rate = {}
        if self.direction_bias == 'cosine' and len(self.Path_log) > 1:
            x0, y0 = self.Path_log[-2]
            x1, y1 = self.Path_log[-1]
            base_vector = (x1 - x0, y1 - y0)
            for loc in conjunct_cells_in_pr:
                xt, yt = loc
                tar_vector = (xt - x1, yt - y1)
                bias_rate[loc] = cosine_bias_rate(base_vector, tar_vector)
        elif self.direction_bias == 'cosine' and len(self.Path_log) <= 1:
            for loc in conjunct_cells_in_pr:
                bias_rate[loc] = 1
        return bias_rate

    def IsConjunct(self, cells_to_conjunct) -> set:
        """
        This function finds whether the input location set is connected to where the individual is.
        It will return a set of locations that are reachable from where the individual is.
        @param cells_to_conjunct:
            The input location set that is used to find reachable locations from where the individual is.
        @return:
            The set/subset of locations that are reachable from where the individual is.
            Note that although the input location set does not contain the location of an individual, the return contains.
        """
        candidate_cells = {self.position}
        conjunct_cells = {self.position}
        for _ in range(len(cells_to_conjunct) + 1):
            if len(candidate_cells) == 0:
                return conjunct_cells
            else:
                if self.position in candidate_cells:
                    neighbors = self.NeighborLocs(position=self.position, principle=self.principle).intersection(cells_to_conjunct)
                    if len(neighbors) == 0:
                        return conjunct_cells
                    else:
                        candidate_cells.remove(self.position)
                        candidate_cells.update(neighbors)
                        conjunct_cells.update([self.position])
                else:
                    center = candidate_cells.pop()
                    neighbors = self.NeighborLocs(position=center, principle=self.principle).intersection(cells_to_conjunct)
                    neighbors.difference_update(conjunct_cells)
                    candidate_cells.update(neighbors)
                    candidate_cells.intersection_update(cells_to_conjunct)
                    conjunct_cells.update([center])
        return conjunct_cells

    def NeighborLocs(self, **kwargs) -> set:
        """
        This function returns a set containing the location of neighbors for the individual or specified location.
        @param: kwargs
            position:   position=(x, y)
                It specifies where the individual is.
            principle:    principle='4' or '8'
                It specifies the neighborhood principle used to identify locations directly connected to 'position'.
        @return:
            A set containing the neighbors of an individual.
            If 'position'=(1, 1) and 'principle'='4', it will return a set {(0, 1), (2, 1), (1, 0), (1, 2)}.
            Note: the returned set does not distinguish if the neighbors are inside the calculation boundary.
        """
        if 'position' in kwargs:
            position = kwargs['position']
        else:
            position = self.position
        if 'principle' in kwargs:
            principle = kwargs['principle']
        else:
            principle = self.principle
        x, y = position
        if principle == '4':
            neighbors = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}
        elif principle == '8':
            neighbors = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y + 1), (x - 1, y - 1), (x + 1, y + 1), (x + 1, y - 1)}
        return neighbors

    def CellsInPR(self, puzzle=False) -> set:  # returns cells in perception range
        """
        This function delineates the locations that are inside PR according to where the individual is.
        This is affected by the keyword variable puzzle.
        @param puzzle:
            It indicates whether the individual is perceiving in a puzzle state.
            puzzle=True: delineate locations inside study area and inside PR and not in natal habitat.
            puzzle=False: delineate locations inside study area and inside PR and not in natal habitat and cost value smaller than σ;.
        @return:
            A set containing the locations inside PR in terms of the state of an individual

        Note:
            This function will also be affected by the mode to initialize individuals, indicated by self.birth_hab.
            The returned set excludes where the individual is.
        """
        x0, y0 = self.position
        Sig = self.Sig
        PR = self.PR
        cost_surface, AllCellsLocs,  NatalHabLocs = self.cost_surface, self.AllCellsLocs, self.HabIndex_Locs[self.birth_hab]
        x_low, x_high = max(x0 - PR, 0), min(x0 + PR + 1, self.cost_surface_shape[0])
        y_low, y_high = max(y0 - PR, 0), min(y0 + PR + 1, self.cost_surface_shape[1])
        x_list = np.arange(x_low, x_high)
        y_list = np.arange(y_low, y_high)
        cells_in_pr = set()
        if puzzle:
            for x in x_list:
                for y in y_list:
                    if (x, y) in AllCellsLocs and distance((x, y), (x0, y0)) <= PR and (x, y) not in NatalHabLocs:
                        cells_in_pr.update([(x, y)])
        else:
            for x in x_list:
                for y in y_list:
                    if (x, y) in AllCellsLocs and distance((x, y), (x0, y0)) <= PR and (x, y) not in NatalHabLocs and cost_surface[(x, y)] < Sig:
                        cells_in_pr.update([(x, y)])
        cells_in_pr.discard(self.position)
        return cells_in_pr

    def CreateGraph(self, nodes: set):
        """
        This function is uesed to create graphs for identifying shortest cost paths by Dijkstra algorithm.
        @param nodes:
            The nodes set to create the graph.
        @return:
            It will generate a list consisting of (node1, node2, weight), where (node1, node2, weight) means a pair of nodes with a directed edge
            from node1 pointing to node2 with weight. This list will be transferred to function add_graph_from_edge() to generate a formated graph
            for the Dijkstra algorithm.
        Note:
            The Dijkstra algorithm is modified to randomly generate a least cost path among all possible paths.
        """
        edge_weight = set()
        cost_surface = self.cost_surface
        for one_node in nodes:
            neighbors_4_8 = self.NeighborLocs(position=one_node, principle=self.principle).intersection(nodes)
            for one_neighbor in neighbors_4_8:
                edge_weight.update([(one_node, one_neighbor, cost_surface[one_neighbor] * self.resolution * distance(one_node, one_neighbor))])
        return add_graph_from_edge(edge_weight)


class Data_analyses:
    def __init__(self, result, **kwargs):
        self.results = result
        if "HabIndex_Locs" in kwargs:
            self.HabIndex_Locs = kwargs['HabIndex_Locs']
        else:
            print('HabIndex_Locs is required')
        if "cost_surface" in kwargs:
            self.cost_surface = kwargs['cost_surface']
        else:
            print('cost_surface is required')
        if 'Hab_number' in kwargs:
            self.Hab_number = kwargs['Hab_number']
        elif 'HabIndex_Locs' in kwargs:
            self.Hab_number = len(kwargs['HabIndex_Locs'])
        if 'file_name' in kwargs:
            self.file_name = kwargs['file_name']
        else:
            self.file_name = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        self.suc_transfer_mat = np.zeros((self.Hab_number, self.Hab_number))
        self.HabIndex_Boarder = kwargs['HabIndex_BoarderLocs']
        self.habpair_corrilocs_ = {}  # it is further used to generate corridor_map if self.Identify_Corridors() is used
        self.rsfi_map = None  # initialize the rsfi_map
        self.corridor_map = None  # initialize a vacant corridor_map, a real corridor_map will be generated by calling self.Identify_Corridors()
        if 'save_dir' in kwargs:
            self.save_dir = kwargs['save_dir']
        else:
            self.save_dir = 'D:/'

    def Map_RSFI(self, success, rsfi_threshold):
        print('\n')
        print('----- Start Map_RSFI -----')
        self.rsfi_map = np.zeros_like(self.cost_surface).astype(float)
        for ind in self.results:
            for locs in ind.Path_log:
                self.rsfi_map[locs] += 1
        if success != -1:
            output_rsfimap = np.zeros_like(self.cost_surface).astype(float)
            for ind in self.results:
                if ind.Success == success:
                    for locs in ind.Path_log:
                        output_rsfimap[locs] += 1
        else:
            output_rsfimap = self.rsfi_map
        if rsfi_threshold is not None and type(rsfi_threshold) == int:
            threshold_ = rsfi_threshold
        elif rsfi_threshold is not None and type(rsfi_threshold) == float:
            threshold_ = ceil(rsfi_threshold * np.max(output_rsfimap))
        elif rsfi_threshold is None:
            print('----- Map_RSFI completes -----')
            return output_rsfimap
        output_rsfimap[output_rsfimap < threshold_] = -9999
        print('----- Map_RSFI completes -----')
        return output_rsfimap

    def detect_Habpairs_Corrilocs(self, reason=1):
        print('\n')
        print('----- Start get_Habpairs_Corrilocs -----')
        for ind in self.results:
            if ind.Success == reason:
                source_hab_index = ind.birth_hab
                target_hab_index = [HabIndex for HabIndex, Locs in self.HabIndex_Locs.items() if ind.Path_log[-1] in Locs][0]
                habpair = (min(source_hab_index, target_hab_index), max(source_hab_index, target_hab_index))
                self.habpair_corrilocs_.setdefault(habpair, set())
                self.habpair_corrilocs_[habpair].update(ind.Path_log)
        with open(r'{}/{} habpair_corridors.txt'.format(self.save_dir, self.file_name), 'w') as save_file:
            for habpair, corrilocs in self.habpair_corrilocs_.items():
                print(habpair, corrilocs, sep=';', file=save_file)
        print('----- get_Habpairs_Corrilocs completes -----')
        return self.habpair_corrilocs_

    def get_RSFI_map(self, success=-1, rsfi_threshold=None):
        print('\n')
        print('----- Start get_RSFI_map -----')
        output_rsfimap = self.Map_RSFI(success, rsfi_threshold)
        save_mat_files(output_rsfimap, self.save_dir, '{} rsfi suc={} threshold={}.txt'.format(self.file_name, success, rsfi_threshold))
        save_mat_files(self.rsfi_map, self.save_dir, '{} rsfi all.txt'.format(self.file_name))
        print('----- get_RSFI_map completes -----')
        return output_rsfimap, self.rsfi_map

    def get_Success(self):
        print('\n')
        print('----- Start get_Success -----')
        success_stats = 0
        for ind in self.results:
            if ind.Success == 1:
                success_stats += 1
        with open(r'{}/{} Success.txt'.format(self.save_dir, self.file_name), 'w') as save_file:
            print(f'Success number = {success_stats}', f'Total number = {len(self.results)}', file=save_file)
        print('Overall success rate:', success_stats/len(self.results))
        print('----- get_Success completes -----')
        return success_stats, len(self.results)

    def get_Reasons(self):
        print('\n')
        print('----- Start get_Reasons -----')
        Success = 0
        MDT_low = 0
        Crossing_barriers = 0
        Trapped = 0
        Moving = 0
        for ind in self.results:
            if ind.Success == 1 and ind.Reason == 'Successful migration':
                Success += 1
            elif ind.Success == 0 and ind.Reason == 'Die for MDT low':
                MDT_low += 1
            elif ind.Success == 0 and ind.Reason == 'Die for crossing impediment':
                Crossing_barriers += 1
            elif ind.Success == 0 and ind.Reason == 'Trapped':
                Trapped += 1
            elif ind.Success is None:
                Moving += 1
        with open(r'{}/{} Reasons.txt'.format(self.save_dir, self.file_name), 'w') as save_file:
            print(f'Successful migration = {Success}', file=save_file)
            print(f'Death for MDT low = {MDT_low}', file=save_file)
            print(f'Dead for crossing impediment = {Crossing_barriers}', file=save_file)
            print(f'Trapped = {Trapped}', file=save_file)
            print(f'Moving = {Moving}', file=save_file)
            print(f'Number of individuals = {len(self.results)}', file=save_file)
        print('----- get_Reasons completes -----')
        return Success, MDT_low, Crossing_barriers, Trapped, Moving

    def get_Sinuosity_Index(self, min_angle):
        print('\n')
        print('----- Start get_Sinuosity_Index -----')
        indID_SI = {}
        for ind in self.results:
            indID_SI[ind.ID] = Sinuosity_Index(ind.Path_log, min_angle)
        with open(r'{}/{} SI.txt'.format(self.save_dir, self.file_name), 'w') as save_file:
            for indID, SI in indID_SI.items():
                print(indID, SI, sep=';', file=save_file)
        print('----- get_Sinuosity_Index completes -----')
        return indID_SI

    def get_Successful_Transfer_Mat(self):
        print('\n')
        print('----- Start get_Successful_Dispersal_Mat -----')
        hab_indnum = {}
        for ind in self.results:
            hab_indnum.setdefault(ind.birth_hab, 0)
            hab_indnum[ind.birth_hab] += 1
            if ind.Success == 1 and ind.Reason == 'Successful migration':
                source_hab = ind.birth_hab
                target_hab = [HabIndex for HabIndex, Locs in self.HabIndex_Locs.items() if ind.Path_log[-1] in Locs][0]
                self.suc_transfer_mat[source_hab, target_hab] += 1
        for row in range(self.suc_transfer_mat.shape[0]):
            self.suc_transfer_mat[row, row] = np.sum(self.suc_transfer_mat[row, :])
            self.suc_transfer_mat[row, :] = self.suc_transfer_mat[row, :] / hab_indnum[row]
        self.suc_transfer_mat = self.suc_transfer_mat
        save_mat_files(self.suc_transfer_mat, self.save_dir, '{} suc_transfer_mat.txt'.format(self.file_name))
        print('----- get_Successful_Transfer_Mat completes -----')
        return self.suc_transfer_mat

    def get_Mortality_Map(self):
        print('\n')
        print('----- Start get_Mortality_Map -----')
        mortality_map = np.zeros_like(self.cost_surface)
        for ind in self.results:
            if ind.Success == 0:
                dead_loc = ind.Path_log[-1]
                mortality_map[dead_loc] += 1
        mortality_map = mortality_map / len(self.results)
        save_mat_files(mortality_map, self.save_dir, '{} mortality map.txt'.format(self.file_name))
        print('----- get_Mortality_Map completes -----')
        return mortality_map

    def get_Corridor_Map(self, **kwargs):
        """
        It will generate corridors between specified habitats with specified rsfi threshold.
        @param kwargs: the kwargs will be transfered to self.Identify_Corridors(**kwargs)
            success:    success=1 or -1
                It tells which rsfi map will be used. For 1, only successful movements are involved; -1, all movements are involved.
            specify_habitat:    specify_habitat=[1, 2, 3]   1, 2, 3 are habitat index.
                It specifies which habitats are involved in a corridor map.
            specify_rsfi:   specify_rsfi=int() or float() or 'pairwise_mean' or 'global_mean' or None
                It specifies the rsfi threshold to be used in a corridor map.
            reserve_rsfi:   reserve_rsfi=True or False
                If True, each matrix item in the output corridor map will hold the rsfi value; if false, only a binary index of 0 and 1 is used to
                indicate whether a matrix item is a corridor.
        @return:
        """
        print('\n')
        print('----- Start get_Corridor_Map -----')
        self.detect_Habpairs_Corrilocs()
        corridor_map = self.Identify_Corridors(**kwargs)
        save_mat_files(corridor_map, self.save_dir, '{} corridor map.txt'.format(self.file_name))
        print('----- get_Corridor_Map completes -----')
        return corridor_map

    def Identify_Corridors(self, **kwargs):
        print('\n')
        print('----- Start Identify_Corridors -----')
        if 'success' in kwargs:
            success = kwargs['success']
        else:
            success = 1
        self.corridor_map = np.zeros_like(self.cost_surface, dtype=float)
        reference_rsfi_map = self.Map_RSFI(success=success, rsfi_threshold=None)
        if 'rsfi_threshold' in kwargs and kwargs['rsfi_threshold'] != 'pairwise_mean':
            if 'rsfi_threshold' in kwargs and type(kwargs['rsfi_threshold']) in [int, float]:
                RSFI_threshold = kwargs['rsfi_threshold']
                if type(RSFI_threshold) is float:
                    RSFI_threshold = RSFI_threshold * np.nanmax(reference_rsfi_map)
            elif 'rsfi_threshold' in kwargs and kwargs['rsfi_threshold'] == 'global_mean':
                RSFI_threshold = np.sum(reference_rsfi_map) / len(np.where(reference_rsfi_map > 0)[0])
            for habpair, raw_corrilocs in self.habpair_corrilocs_.items():
                s_hab, t_hab = habpair
                reserved_corrilocs = set([loc for loc in raw_corrilocs if reference_rsfi_map[loc] > RSFI_threshold])

                clusterindex_clusterlocs = {}
                cluster_index = 0
                for _ in range(len(reserved_corrilocs) + 1):
                    if len(reserved_corrilocs) > 0:
                        clusterlocs = GatherHabs(reserved_corrilocs)
                        clusterindex_clusterlocs[cluster_index] = clusterlocs
                        cluster_index += 1
                        reserved_corrilocs = reserved_corrilocs - clusterlocs
                    else:
                        break

                corridors = [clusterlocs for clusterindex, clusterlocs in clusterindex_clusterlocs.items() if
                             not clusterlocs.isdisjoint(self.HabIndex_Boarder[s_hab]) and not clusterlocs.isdisjoint(self.HabIndex_Boarder[t_hab])]

                if 'reserve_rsfi' in kwargs and kwargs['reserve_rsfi'] is True:
                    for cluster in corridors:
                        for loc in cluster:
                            self.corridor_map[loc] = reference_rsfi_map[loc]
                else:
                    for cluster in corridors:
                        for loc in cluster:
                            self.corridor_map[loc] = 1

            self.corridor_map[self.corridor_map == 0] = -9999
            return self.corridor_map
        elif 'rsfi_threshold' in kwargs and kwargs['rsfi_threshold'] == 'pairwise_mean':
            pairwisehab_suclocs = {}
            for ind in self.results:
                if ind.Success == success:
                    source_hab_index = ind.birth_hab
                    target_hab_index = [HabIndex for HabIndex, Locs in self.HabIndex_Locs.items() if ind.Path_log[-1] in Locs][0]
                    habpair = (min(source_hab_index, target_hab_index), max(source_hab_index, target_hab_index))
                    pairwisehab_suclocs.setdefault(habpair, [])
                    pairwisehab_suclocs[habpair].append(ind.Path_log)
            for pairwisehab, sucpaths in pairwisehab_suclocs.items():
                s_hab, t_hab = pairwisehab
                loc_rsfi = {}
                for path in sucpaths:
                    for loc in path:
                        loc_rsfi.setdefault(loc, 0)
                        loc_rsfi[loc] += 1
                RSFI_threshold = sum(loc_rsfi.values()) / len(loc_rsfi)
                reserved_suclocs = set([loc for loc, rsfi in loc_rsfi.items() if rsfi > RSFI_threshold])

                clusterindex_clusterlocs = {}
                cluster_index = 0
                for _ in range(len(reserved_suclocs) + 1):
                    if len(reserved_suclocs) > 0:
                        clusterlocs = GatherHabs(reserved_suclocs)
                        clusterindex_clusterlocs[cluster_index] = clusterlocs
                        cluster_index += 1
                        reserved_suclocs = reserved_suclocs - clusterlocs
                    else:
                        break

                corridors = [clusterlocs for clusterindex, clusterlocs in clusterindex_clusterlocs.items() if
                             not clusterlocs.isdisjoint(self.HabIndex_Boarder[s_hab]) and not clusterlocs.isdisjoint(self.HabIndex_Boarder[t_hab])]
                print(pairwisehab, RSFI_threshold, len(corridors))
                if 'reserve_rsfi' in kwargs and kwargs['reserve_rsfi'] is True:
                    for cluster in corridors:
                        for loc in cluster:
                            self.corridor_map[loc] += loc_rsfi[loc]
                else:
                    for cluster in corridors:
                        for loc in cluster:
                            self.corridor_map[loc] = 1
            self.corridor_map[self.corridor_map == 0] = -9999
            return self.corridor_map
        elif 'rsfi_threshold' in kwargs and kwargs['rsfi_threshold'].split('_')[0] == 'pairwise' and type(eval(kwargs['rsfi_threshold'].split('_')[1])) in [int, float]:
            RSFI_threshold_raw = eval(kwargs['rsfi_threshold'].split('_')[1])
            pairwisehab_suclocs = {}
            for ind in self.results:
                if ind.Success == success:
                    source_hab_index = ind.birth_hab
                    target_hab_index = [HabIndex for HabIndex, Locs in self.HabIndex_Locs.items() if ind.Path_log[-1] in Locs][0]
                    habpair = (min(source_hab_index, target_hab_index), max(source_hab_index, target_hab_index))
                    pairwisehab_suclocs.setdefault(habpair, [])
                    pairwisehab_suclocs[habpair].append(ind.Path_log)

            for pairwisehab, sucpaths in pairwisehab_suclocs.items():
                s_hab, t_hab = pairwisehab
                loc_rsfi = {}
                for path in sucpaths:
                    for loc in path:
                        loc_rsfi.setdefault(loc, 0)
                        loc_rsfi[loc] += 1
                if type(RSFI_threshold_raw) is float:
                    RSFI_threshold = np.max(list(loc_rsfi.values())) * RSFI_threshold_raw
                elif type(RSFI_threshold_raw) is int:
                    RSFI_threshold = RSFI_threshold_raw
                reserved_suclocs = set([loc for loc, rsfi in loc_rsfi.items() if rsfi > RSFI_threshold])

                clusterindex_clusterlocs = {}
                cluster_index = 0
                for _ in range(len(reserved_suclocs) + 1):
                    if len(reserved_suclocs) > 0:
                        clusterlocs = GatherHabs(reserved_suclocs)
                        clusterindex_clusterlocs[cluster_index] = clusterlocs
                        cluster_index += 1
                        reserved_suclocs = reserved_suclocs - clusterlocs
                    else:
                        break

                corridors = [clusterlocs for clusterindex, clusterlocs in clusterindex_clusterlocs.items() if
                             not clusterlocs.isdisjoint(self.HabIndex_Boarder[s_hab]) and not clusterlocs.isdisjoint(self.HabIndex_Boarder[t_hab])]
                print(pairwisehab, RSFI_threshold, len(corridors))
                if 'reserve_rsfi' in kwargs and kwargs['reserve_rsfi'] is True:
                    for cluster in corridors:
                        for loc in cluster:
                            self.corridor_map[loc] += loc_rsfi[loc]
                else:
                    for cluster in corridors:
                        for loc in cluster:
                            self.corridor_map[loc] = 1
            self.corridor_map[self.corridor_map == 0] = -9999
            return self.corridor_map


class reIndividual:
    def __init__(self, **kwargs):
        self.ID = kwargs['ID']
        self.I_time = kwargs['t']
        self.birth_hab = kwargs['birthhab']
        self.birth_loc = kwargs['birthloc']
        self.step_num = kwargs['step_num']
        self.Success = kwargs['Success']
        self.Reason = kwargs['Reason']
        self.Tar_log = kwargs['Tar_log']
        self.Path_log = kwargs['Path_log']


def RecallIndividual(file_dir, file_name):
    file_data = open(r'{}/{}'.format(file_dir, file_name), 'r')
    datalines = file_data.readlines()
    headline = 'ID;t;birthhab;birthloc;step_num;Success;Reason;Tar_log;Path_log'.split(';')
    file_data.close()
    res = []
    for lines in tqdm(datalines):
        data = lines.split(';')
        ind_info = {}
        ind_info[headline[0]] = data[0]         # ID
        ind_info[headline[1]] = eval(data[1])   # t
        ind_info[headline[2]] = eval(data[2])   # birthhab
        ind_info[headline[3]] = eval(data[3])   # birthloc
        if data[4] == 'None':
            ind_info[headline[4]] = data[4]   # step_num
        else:
            ind_info[headline[4]] = eval(data[4])
        ind_info[headline[5]] = eval(data[5])   # Success
        ind_info[headline[6]] = data[6]         # Reason
        ind_info[headline[7]] = eval(data[7])   # Tar_log
        ind_info[headline[8]] = eval(data[8])   # Path_log
        res.append(reIndividual(**ind_info))
    print(f'From {file_dir}/{file_name} reloads \n {len(res)}/{len(datalines)} individuals')
    return res
