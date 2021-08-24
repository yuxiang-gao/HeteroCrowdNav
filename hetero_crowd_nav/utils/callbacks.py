import numpy as np
import time
from pandas import DataFrame

from stable_baselines3.common.callbacks import BaseCallback, EvalCallBack


class LogCallBack(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param log_path: (string) where to save the training log
    :param eval_interval: (int) frequency fo evaluation
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(
        self,
        log_path: str = None,
        eval_interval: int = 10000,
        verbose: int = 1,
    ) -> None:
        super(LogCallBack, self).__init__(verbose)
        self.log_path = log_path
        self.eval_interval = eval_interval

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
