from typing import (
    List
)
import numpy as np
import logging
from common.test_config import (
    GRADIENT_DIFF_CHECK_TRIGGER,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)


def check_with_numerical_gradient(dS: List[np.ndarray], gn: List[np.ndarray], logger: logging.Logger):
    # ********************************************************************************
    #  Constraint. Numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
    # ********************************************************************************
    if not (
            np.all(gn[0] <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.allclose(
                a=dS[0][gn[0] != 0],  # dL/dX: (N,M1)
                b=gn[0][gn[0] != 0],  # dL/dX: (N,M1)
                atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
                rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
            )
    ):
        logger.error(
            "dL/dX analytical gradient \n%s \nneed to close to numerical gradient \n%s\ndifference=\n%s\n",
            dS[0], gn[0], (dS[0] - gn[0])
        )

    if not (
            np.all(gn[1] <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.allclose(
                a=dS[1][gn[1] != 0],
                b=gn[1][gn[1] != 0],
                atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
                rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
            )
    ):
        logger.error(
            "Need similar analytical and numerical dL/dW. \n"
            "Analytical=\n%s\nNumerical\n%s\ndifference=\n%s\n",
            dS[1], gn[1], (dS[1] - gn[1])
        )
