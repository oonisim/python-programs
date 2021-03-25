from typing import (
    List
)
import numpy as np
import logging
from test import (
    GRADIENT_DIFF_CHECK_TRIGGER,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)
from common.constants import (
    ENFORCE_STRICT_ASSERT
)


def check_with_numerical_gradient(dS: List[np.ndarray], gn: List[np.ndarray], logger: logging.Logger):
    for ds, gn in zip(dS, gn):
        if not (
                np.all(gn <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.allclose(
                    a=ds[gn != 0],  # dL/dX: (N,M1)
                    b=gn[gn != 0],  # dL/dX: (N,M1)
                    atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
                    rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
                )
        ):
            logger.error(
                "Need similar analytical and numerical ds. \n"
                "Analytical=\n%s\nNumerical\n%s\ndifference=\n%s\n",
                ds, gn, (ds - gn)
            )
            assert ENFORCE_STRICT_ASSERT
