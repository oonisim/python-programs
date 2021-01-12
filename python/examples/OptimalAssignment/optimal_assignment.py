"""
https://coderbyte.com/information/Optimal%20Assignments
"""

from itertools import *
import logging
import sys

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
#logger.setLevel(logging.DEBUG)

MACHINE_TASK_COMBINATIONS = []
SIZE = 0

def get_task_estimates_for_a_machine(row):
    """
    Convert task estimation string for a machine "(1,2,1)" into int list [1, 2, 1]
    """
    estimates = []
    for str in row[1:-1].split(','):
        estimates.append(int(str))

    return estimates


def get_machine_task_to_cost_matrix(estimates):
    """
    From the input string, create 2D array representing the matrix of the task
    where (row, col) is the task cost, row is machine id, col is task id
    :param estimates:
        Estimate of the cost for a machine to do a task in the string representation
        of a matrix where row is a machine and column is a task.
        [ "(1,2,1)",      <--- Costs of machine 0 to do task 0,1,2 are 1,2,1
          "(4,1,5)",
          "(5,2,1)"]
    :return: Task matrix
    """
    num_machines = len(estimates)
    num_tasks = len(estimates)
    matrix = [[0 for x in range(num_tasks)] for y in range(num_machines)]

    for row in range(0, num_machines):
        cost_estimates_for_machine = get_task_estimates_for_a_machine(estimates[row])
        for col in range(0, num_tasks):
            matrix[row][col] = cost_estimates_for_machine[col]

    return matrix


def tabs(size):
    tabs = ""
    for i in range(0, SIZE - size):
        tabs += "\t"
    return tabs


def find_nqueens_combinations(accumulator, available_machine_ids, available_task_ids, size):
    """
    Get the N Queen positions of (machine, task) for size x size matrix

    TODO:
        No need to have machine in (mamchine, task). Only available_task_ids suffices.
        [1, 0, 2]  can represent NQ combination [(0,1)(1,0)(2,2)] because its index is machine id.
        If machine ID is not sequential, make it sequential and have a map table back to original ID.

        Machine  Task1 Task2 Task3
        0                X
        1          X
        2                      X

    :param size
    :return: List of the nqueen positions. A position for 3 x 3 can be [(0, 0), (1, 1), (2, 2)]
    """
    global MACHINE_TASK_COMBINATIONS
    if size <= 0:
        accumulator.sort()
        if accumulator not in MACHINE_TASK_COMBINATIONS:
            logger.debug("{}Adding a queen {}".format(tabs(size), accumulator))
            MACHINE_TASK_COMBINATIONS.append(accumulator)
        return

    # --------------------------------------------------------------------------------
    # Available (machine, task) combinations in size x size matrix
    # Same with (x, y) coordinate in N Queens.
    # --------------------------------------------------------------------------------
    #XYs = set([(x,y) for x in listx for y in listy])
    machine_task_combinations = list(product(available_machine_ids, available_task_ids))

    logger.debug("{}Available (machine, task) combinations at size {} is {}".format(
        tabs(size),
        size,
        machine_task_combinations
    ))

    for machine_task in machine_task_combinations:
        machine = machine_task[0]
        task = machine_task[1]

        # --------------------------------------------------------------------------------
        # Pick (machine, task) and remove the machine and task from the availability list.
        # --------------------------------------------------------------------------------
        next_accumulator = accumulator.copy()
        next_accumulator.append((machine, task))

        next_available_machine_ids = available_machine_ids.copy()
        next_available_machine_ids.remove(machine)

        next_available_task_ids = available_task_ids.copy()
        next_available_task_ids.remove(task)

        logger.debug("{}Assignment{}: assigned task {} to machine {} ".format(
            tabs(size),
            (machine, task),
            machine,
            task
        ))
        find_nqueens_combinations(
            next_accumulator,
            next_available_machine_ids,
            next_available_task_ids,
            (size - 1)
        )


def get_cost_for_machine_task(machine_task_to_cost_matrix, machine, task):
    """
    Get the cost at (machine, task)
    :param machine_task_to_cost_matrix: 
    :param machine: 
    :param task: 
    :return: cost of (machine, task)
    """
    cost = machine_task_to_cost_matrix[machine][task]
    """
    logger.debug("matrix is {} cost at ({} {}) is {}".format(
        machine_task_to_cost_matrix, 
        machine, 
        task, 
        cost)
    )
    """
    return cost


def get_machine_task_combinations_at_minimum_cost(matrix):
    """
    Find (machine, task) combinations that has the minimal cost
    """
    costs = []
    for nq in MACHINE_TASK_COMBINATIONS:
        cost = 0
        for coordinate in nq:
            #logger.debug("coodinate is {}".format(coordinate))
            cost += get_cost_for_machine_task(matrix, coordinate[0], coordinate[1])
        costs.append(cost)

    index = costs.index(min(costs))
    logger.debug("min cost in {} is {}".format(costs, costs[index]))
    logger.debug("combination is {}".format(MACHINE_TASK_COMBINATIONS[index]))
    logger.debug("Result is {}".format(MACHINE_TASK_COMBINATIONS[index]))

    return MACHINE_TASK_COMBINATIONS[index]


def OptimalAssignments(estimates):
    matrix = get_machine_task_to_cost_matrix(estimates)
    logger.debug("Machine Task to Cost matrix is [".format(matrix))
    for index in range(0, len(matrix)):
        logger.debug("\t{}".format(matrix[index]))
    logging.debug("]")

    # ----------------------------------------------------------------------
    # Get permissible (machine, task) combinations as N queens
    # ----------------------------------------------------------------------
    size = len(estimates)
    available_machine_ids = list(range(0, size))
    available_task_ids = list(range(0, size))
    find_nqueens_combinations(
        [],
        available_machine_ids,
        available_task_ids,
        size
    )
    logger.debug("MACHINE_TASK_COMBINATIONS is {}".format(MACHINE_TASK_COMBINATIONS))

    answer = ""
    minimum_cost_machine_task_combinations = get_machine_task_combinations_at_minimum_cost(matrix)
    for machine_task in minimum_cost_machine_task_combinations:
        machine = str(machine_task[0] + 1)  # Adjust to start from 1
        task = str(machine_task[1] + 1)     # Adjust to start from 1
        position = "".join(['(', str(machine),  '-', str(task), ')'])
        answer += position

    return answer


if __name__ == "__main__":
    tests = []
    tests.append(["(1,2,1)","(4,1,5)","(5,2,1)"])
    tests.append(["(5,4,2)","(12,4,3)","(3,4,13)"])
    tests.append(["(13,4,7,6)","(1,11,5,4)","(6,7,2,8)", "(1,3,5,9)"])

    MACHINE_TASK_COMBINATIONS = []
    test = tests[2]
    SIZE = len(test)
    print(OptimalAssignments(tests[2]))

