import logging
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT
)
import function.common.base as base
import function.nn.weight as weight

Logger = logging.getLogger(__name__)


class Function(base.Function):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def weights(M: TYPE_INT, D: TYPE_INT, scheme: str = "uniform"):
        return weight.Weights(M=M, D=D, scheme=scheme)

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: ID name
        """
        super().__init__(name=name, log_level=log_level)
