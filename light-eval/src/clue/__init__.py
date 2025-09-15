"""
CLUE Benchmark评估包
"""

from .clue_tasks import (
    CLUETask,
    AFQMCTask,
    TNEWSTask,
    CMNLITask,
    IFLYTEKTask,
    CSLTask,
    WSCTask,
    TASK_REGISTRY,
    get_task
)

from .download_clue import (
    CLUE_TASKS,
    download_clue_task,
    download_all_clue_tasks
)

__all__ = [
    'CLUETask',
    'AFQMCTask',
    'TNEWSTask',
    'CMNLITask',
    'IFLYTEKTask',
    'CSLTask',
    'WSCTask',
    'TASK_REGISTRY',
    'get_task',
    'CLUE_TASKS',
    'download_clue_task',
    'download_all_clue_tasks'
]

__version__ = '1.0.0'