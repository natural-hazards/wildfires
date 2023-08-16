"""

"""

from mlfire.utils.functool import lazy_import

# lazy imports
_datetime = lazy_import('datetime')

_workload_managers_base = lazy_import('scripts.workload_managers.base')
_BaseWorkloadManager = _workload_managers_base.BaseWorkloadManager


class SlurmWorkloadManager(_BaseWorkloadManager):

    def __init__(self,
                 project_id: str = None,
                 job_name: str = None,
                 partition: str = '',
                 ntasks: int = 1,
                 nnodes: int = 1,
                 executable_path: str = '',
                 wall_time: _datetime.time = _datetime.time(hour=0, minute=10)
                 ):

        self._project_id = None
        self.project_id = project_id

        self._job_name = None
        self.job_name = job_name

        self._partition = None
        self.partition = partition

        self._ntasks = None
        self.ntasks = ntasks

        self._nnodes = None
        self.nnodes = nnodes

        self._wall_time = None
        self.wall_time = wall_time

        self._executable_path = None
        self.executable_path = executable_path

    # properties

    @property
    def project_id(self) -> str:

        return self._project_id

    @project_id.setter
    def project_id(self, project_id) -> None:

        self._project_id = project_id

    @property
    def job_name(self) -> str:

        return self._job_name

    @job_name.setter
    def job_name(self, job_name) -> None:

        self._job_name = job_name

    @property
    def ntasks(self) -> int:

        return self._ntasks

    @ntasks.setter
    def ntasks(self, ntasks: int):

        self._ntasks = ntasks

    @property
    def nnodes(self) -> int:

        return self._nnodes

    @nnodes.setter
    def nnodes(self, nnodes: int) -> None:

        self._nnodes = nnodes

    @property
    def wall_time(self) -> _datetime.time:

        return self._wall_time

    @wall_time.setter
    def wall_time(self, wall_time: _datetime.time) -> None:

        self._wall_time = wall_time

    @property
    def executable_path(self) -> str:

        return self._executable_path

    @executable_path.setter
    def executable_path(self, bin_path):

        self._executable_path = bin_path

    def getHeaderDirectives(self) -> list:

        str_wtime_hours = '{:02d}'.format(self.wall_time.hour)
        str_wtime_minutes = '{:02d}'.format(self.wall_time.minute)

        directives = [
            f'#SBATCH --account {self.project_id}\n',
            f'#SBATCH --job-name {self.job_name}\n',
            f'#SBATCH --partition {self.partition}\n',
            f'#SBATCH --nodes {self.nnodes}\n',
            f'#SBATCH --ntasks {self.ntasks}\n',
            f'#SBATCH --time {str_wtime_hours}:{str_wtime_minutes}:00\n'
            f'#SBATCH --output {self.job_name.lower()}.%j.out\n',
            f'#SBATCH --error {self.job_name.lower()}.%j.err\n',
        ]

        return directives

    def getRunOptions(self) -> list:

        srun_options = [
            'srun ',
            '\\\n',
            f'{self.executable_path} \\\n',
        ]

        return srun_options


if __name__ == '__name__':

    pass
