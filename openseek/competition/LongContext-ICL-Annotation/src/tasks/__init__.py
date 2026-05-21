"""Task registry: maps task_id to task class."""
from tasks.task_1 import Task1
from tasks.task_2 import Task2
from tasks.task_3 import Task3
from tasks.task_4 import Task4
from tasks.task_5 import Task5
from tasks.task_6 import Task6
from tasks.task_7 import Task7
from tasks.task_8 import Task8

TASK_REGISTRY = {
    1: Task1,
    2: Task2,
    3: Task3,
    4: Task4,
    5: Task5,
    6: Task6,
    7: Task7,
    8: Task8,
}
