"""Logic for executing task on franka panda"""
from typing import List, Any
from dataclasses import dataclass

import numpy as np


@dataclass
class TaskInfo:
    img_rgb: np.ndarray
    img_depth: np.ndarray
    bbox: List[Any]
    rotation: int


class TaskExecutor:

    def pick(self, data: TaskInfo):
        """Execute pick task"""
    
    def place(self, data: TaskInfo):
        """Execute place task"""

    def home(self,):
        """Goto home position"""