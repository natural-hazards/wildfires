
from enum import Enum


# define Enum class for workload managers
class WorkloadManagerType(Enum):

    SLURM = 'slurm'


class BaseWorkloadManager(object):

    def __init__(self) -> None:

        pass
