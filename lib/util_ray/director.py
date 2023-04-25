"""
Director pattern of assigning tasks to workers
"""
import time
import logging
import multiprocessing
from typing import (
    List,
    Sequence,
    Any,
)

from util_logging import (              # pylint disable=wrong-import-order
    get_logger,
    DEFAULT_LOG_LEVEL
)
from util_python.generator import (     # pylint disable=wrong-import-order
    split
)
import ray


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
NUM_CPUS: int = multiprocessing.cpu_count()
MAX_WORKERS: int = NUM_CPUS


# --------------------------------------------------------------------------------
# Class
# --------------------------------------------------------------------------------
class Director:
    """Director pattern implementation class to assign tasks to workers"""
    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------
    @property
    def logger(self) -> logging.Logger:
        """Provides the logger instance"""
        return self._logger

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            num_workers: int = NUM_CPUS,
            log_level: int = DEFAULT_LOG_LEVEL,
            initialize_ray: bool = True
    ):
        """
        Args:
            num_workers: number of workers to use
            log_level: logging level according to the python logging module
            initialize_ray: flag if initialize ray
        """
        assert 0 < num_workers <= MAX_WORKERS, \
            f"expected 0 < num_workers < {MAX_WORKERS+1}, got {num_workers}"

        # --------------------------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------------------------
        self._logger: logging.Logger = get_logger(__name__)
        self._logger.setLevel(log_level)

        # --------------------------------------------------------------------------------
        # Ray initialization (https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html)
        # --------------------------------------------------------------------------------
        self._num_workers: int = num_workers
        self._ray_initialized: bool = False
        if initialize_ray:
            self._logger.info("main(): initializing Ray using %s workers...", self._num_workers)
            try:
                ray.init(num_cpus=self._num_workers, num_gpus=0, logging_level=log_level)
                self._ray_initialized = True

            except Exception as error:   # Ray init() returns Exception as in the document
                self.logger.error("shutting down ray as ray.init() failed due to %s", error)
                time.sleep(3)
                ray.shutdown()
                raise RuntimeError("ray.init() failed.") from error

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    @ray.remote(num_returns=1)
    def worker(self, task: Any) -> Any:
        """Worker task to execute based on the instruction message
        To be implemented in the child class.

        NOTE:
            Like the best practice in Spark, make sure to reduce the shuffle or
            network traffic. If the end goal in aggregate is to sum all up, then
            do the subtotal in worker.

            For instance, if the task is [1,2,3] and calculate their power,
            instead of returning [1,4,9], return the subtotal 14 if the end
            result at aggregate is total sum.

        Args:
            task: task
        """
        raise NotImplementedError("TBD")

    def assign(self, tasks: Sequence) -> List[Any]:
        """assign tasks to the worker instances.
        The child class implements the concrete worker to execute a specific job.

        Args:
            tasks: a slice-able collection of records
        Returns: List of task results executed by the workers.
        Raises: RuntimeError if there is no task
        """
        if not tasks or len(tasks) == 0:
            raise RuntimeError(f"invalid tasks:{tasks}")

        num_workers: int = min(len(tasks), self._num_workers)
        assert num_workers > 0

        # --------------------------------------------------------------------------------
        # Asynchronously invoke child workers with tasks.
        # See https://docs.ray.io/en/latest/ray-overview/getting-started.html#ray-core-quick-start
        #
        # NOTE: Need to pass "self" as worker.remote(self, task) not worker.remote(task).
        # Python runtime automatically insert self if it is an instance method, but
        # Ray "remote" proxy is a function, not a class instance method.
        # Alternatively make the remote method as static, however you cannot access
        # instance/class members.
        # --------------------------------------------------------------------------------
        futures = [
            self.worker.remote(self, task)  # pylint: disable=no-member
            for task in split(sliceable=tasks, num=num_workers)
        ]
        assert len(futures) == num_workers, \
            f"Expected {num_workers} tasks but got {len(futures)}."

        # --------------------------------------------------------------------------------
        # Wait for the completion of all tasks
        # --------------------------------------------------------------------------------
        waits: List[Any] = []
        while futures:
            completed, futures = ray.wait(futures)
            waits.extend(completed)

        # --------------------------------------------------------------------------------
        # Collect the results from workers.
        # See https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html#ray.get
        # --------------------------------------------------------------------------------
        assert len(waits) == num_workers, f"Expected {num_workers} tasks but got {len(waits)}"
        results: List[Any] = ray.get(waits)

        self.logger.debug("results: %s", results)
        return results

    def aggregate(self, results: List[Any]) -> Any:
        """Aggregate the result results from the workers.
        NOTE:
            Beware the size of the result that each worker returns can be different.
            If the size of tasks is 7 e.g. [1,2,3,4,5,6,7] assigned to 3 workers,
            the sizes are not the same as split to [1,2,3], [4,5], [6,7].

            Then np.sum([[1,2,3], [4,5], [6,7]]) will result in unexpected
            [1,2,3,4,5,6,7] as numpy cannot handle non-equal length causing
            unexpected result.
        """
        raise NotImplementedError("TBD")

    def run(self, tasks: Sequence):
        """Run the process of the director
        """
        return self.aggregate(self.assign(tasks=tasks))

    def close(self):
        """Complete the director"""
        if self._ray_initialized:
            ray.shutdown()
            time.sleep(3)
