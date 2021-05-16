import logging
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR
)
import function.common.base as base
import function.nn.weight as weight

Logger = logging.getLogger(__name__)


class Function(base.Function):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def build_weights(M: TYPE_INT, D: TYPE_INT, scheme: str = "normal", **parameters) -> TYPE_TENSOR:
        return weight.Weights(M=M, D=D, initialization_scheme=scheme, **parameters).weights

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
