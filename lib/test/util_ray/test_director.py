"""Module to test util_ray director class
"""
import logging
from typing import (
    List,
    Any,
)
import multiprocessing

from util_ray.director import (     # pylint: disable=import-error
    Director
)
import numpy as np
import ray


class CalcDirector(Director):
    """Director example class for test"""
    @ray.remote
    def worker(self, task: List[int]) -> int:
        """Worker implementation to run the task"""
        return int(np.sum(np.power(task, 2, dtype=np.int32)))

    def aggregate(self, results: List[Any]) -> int:
        """Aggregate the results of worker tasks"""
        summed = np.sum(results)
        return int(summed)


def test_no_task():
    """Test Director with no task
    Test Condition:
        1. Director raises RuntimeError for no task to do.
    """
    try:
        calculator: Director = CalcDirector()
        calculator.run([])
        calculator.close()

        assert False, "expected RuntimeError, but succeeded."
    except RuntimeError:
        pass

    finally:
        ray.shutdown()


def test_single_task():
    """Test Director with a single task
    Test Condition:
        1. Director can handle single task
    """
    try:
        expected: int = int(np.sum(np.power([3], 2)))

        calculator: Director = CalcDirector()
        result: int = calculator.run([3])
        assert result == np.sum(np.power([3], 2)), \
            f"expected {expected} but {result}."

        calculator.close()

    finally:
        ray.shutdown()


def test_multiple_tasks_less_than_workers():
    """Test Director with multiple tasks whose size is less than num_workers.
    Test Conditions:
        1. Director can handle tasks whose size is less than its workers.
    """
    try:
        tasks: List[int] = [1, 2, 3]
        num_workers: int = len(tasks) + 1
        assert len(tasks) < num_workers

        expected: int = int(np.sum(np.power(tasks, 2)))
        calculator: Director = CalcDirector(num_workers=num_workers)
        result: int = calculator.run(tasks)
        assert result == expected, f"expected {expected} but {result}"

        calculator.close()

    finally:
        ray.shutdown()


def test_multiple_tasks_more_than_workers():
    """Test Director with multiple tasks whose size is less than num_workers.
    Test Conditions:
        1. Director can handle tasks whose size is less than its workers.
    """
    try:
        num_workers: int = multiprocessing.cpu_count()
        tasks: List[int] = list(range(num_workers * 2 + 1))
        assert 0 < num_workers < len(tasks)

        expected: int = int(np.sum(np.power(tasks, 2)))
        calculator: Director = CalcDirector(num_workers=num_workers, log_level=logging.INFO)
        result: int = calculator.run(tasks)
        assert result == expected, f"expected {expected} but {result}"

        calculator.close()

    finally:
        ray.shutdown()
