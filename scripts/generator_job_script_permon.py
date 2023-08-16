
from typing import Union

from mlfire.utils.functool import lazy_import

# lazy imports
_datetime = lazy_import('datetime')
_typing = lazy_import('typing')

_device = lazy_import('scripts.models.device')
_workload_managers = lazy_import('scripts.workload_managers')

_DeviceType = _device.DeviceTypes

_ModelPERMON = lazy_import('scripts.models.permon')
_ScoreTypes = lazy_import('scripts.models.scores').ScoreTypes
_WorkloadManagerType = lazy_import('scripts.workload_managers.base').WorkloadManagerType


class JobScriptBuilder(object):

    def __init__(self,
                 script_name: str = None,
                 project_id: str = None,
                 workload_manager_type: _WorkloadManagerType = _WorkloadManagerType.SLURM,
                 # directives values
                 nnodes: int = 1,
                 ntasks: int = 1,
                 partition: str = '',
                 wall_time: _datetime.time = _datetime.time(hour=0, minute=10),
                 # command before run
                 commands_before_run: Union[list[str], tuple[str]] = None,
                 # locations
                 executable_path: str = None,
                 remote_data_dir: str = None,
                 remote_output_dir: str = None,
                 # prefix
                 ds_prefix: str = None,
                 ds_calib_set: bool = False,
                 # training model settings
                 probability_model: bool = False,
                 loss_type: _ModelPERMON.LossType = _ModelPERMON.LossType.L1,
                 svm_c_value: float = .01,
                 desicion_threshold: float = .5,
                 # hyperparameters optimization
                 run_hyperopt: bool = False,
                 hyperopt_warm_start: bool = False,
                 hyperopt_scores: _typing.Union[list[_ScoreTypes], tuple[_ScoreTypes]] = (_ScoreTypes.JACCARD_INDEX,),
                 hyperopt_cv_nfolds: int = 3,
                 hyperopt_gs_logcbase: float = 10.,
                 hyperopt_gs_grid_range: _typing.Union[list[float], tuple[float]] = None,
                 # solver
                 solver_type: _ModelPERMON.SolverType = _ModelPERMON.SolverType.MPGP,
                 mpgp_gamma: float = 10.,
                 # device
                 device: _DeviceType = _DeviceType.CUDA):

        self._workload_manager = None

        self._workload_manager_type = None
        self.workload_manager_type = workload_manager_type

        self._script_name = None
        self.script_name = script_name

        self._project_id = None
        self.project_id = project_id

        self._nnodes = None
        self.nnodes = nnodes

        self._ntasks = None
        self.ntasks = ntasks

        self._partition = None
        self.partition = partition

        self._wall_time = None
        self.wall_time = wall_time

        # command before run

        self._commands_before_run = None
        self.commands_before_run = commands_before_run

        # locations

        self._remote_data_dir = None
        self.remote_data_dir = remote_data_dir

        self._remote_output_dir = None
        self.remote_output_dir = remote_output_dir

        self._executable_path = None
        self.executable_path = executable_path

        # prefixes

        self._ds_prefix = None
        self.ds_prefix = ds_prefix

        # training model options

        self._probability_model = None
        self.probability_model = probability_model

        self._loss_type = None
        self.loss_type = loss_type

        self._svm_c_value = None
        self.svm_c_value = svm_c_value

        # hyperparameters optimization

        self._run_hyperopt = None
        self.run_hyperopt = run_hyperopt

        self._hyperopt_warm_start = None
        self.hyperopt_warm_start = hyperopt_warm_start

        self._hyperopt_scores = None
        self.hyperopt_scores = hyperopt_scores

        self._hyperopt_cv_nfolds = None
        self.hyperopt_cv_nfolds = hyperopt_cv_nfolds

        self._hyperopt_gs_logcbase = None
        self.hyperopt_gs_logcbase = hyperopt_gs_logcbase

        self._hyperopt_gs_grid_range = None
        self.hyperopt_gs_grid_range = hyperopt_gs_grid_range

        self._desicion_threshold = None
        self.desicion_threshold = desicion_threshold

        # solver settings

        self._solver_type = None
        self.solver_type = solver_type

        self._mpgp_gamma = None
        self.mpgp_gamma = mpgp_gamma

        # device

        self._device = None
        self.device = device

    @property
    def workload_manager_type(self) -> _WorkloadManagerType:

        return self._workload_manager_type

    @workload_manager_type.setter
    def workload_manager_type(self, manager_type: _WorkloadManagerType):

        if self._workload_manager is None:

            if manager_type == _WorkloadManagerType.SLURM:

                SlurmWorkloadManager = lazy_import('scripts.workload_managers.slurm').SlurmWorkloadManager
                self._workload_manager = SlurmWorkloadManager()

            else:

                msg = 'Workload manager of type {} is not implemented!'.format(str(manager_type))
                raise NotImplementedError(msg)

        self._workload_manager_type = manager_type

    @property
    def script_name(self) -> str:

        return self._script_name

    @script_name.setter
    def script_name(self, n: str) -> None:

        self._script_name = n

    @property
    def project_id(self) -> str:

        return self._project_id

    @project_id.setter
    def project_id(self, pid: str) -> None:

        self._project_id = pid

    @property
    def partition(self) -> str:

        return self._partition

    @partition.setter
    def partition(self, p: str) -> None:

        self._partition = p

    @property
    def nnodes(self) -> int:

        return self._nnodes

    @nnodes.setter
    def nnodes(self, nnodes) -> None:

        self._nnodes = nnodes

    @property
    def ntasks(self) -> int:

        return self._ntasks

    @ntasks.setter
    def ntasks(self, n: int) -> None:

        self._ntasks = n

    @property
    def wall_time(self) -> _datetime.time:

        return self._wall_time

    @wall_time.setter
    def wall_time(self, wtime: _datetime.time) -> None:

        self._wall_time = wtime

    @property
    def commands_before_run(self) -> Union[list[str], tuple[str]]:

        return self._commands_before_run

    @commands_before_run.setter
    def commands_before_run(self, cmds: Union[list[str], tuple[str]]) -> None:

        self._commands_before_run = cmds

    """
    Locations
    """

    @property
    def executable_path(self) -> str:

        return self._executable_path

    @executable_path.setter
    def executable_path(self, bin_path: str) -> None:

        self._executable_path = bin_path

    @property
    def remote_data_dir(self) -> str:

        return self._remote_data_dir

    @remote_data_dir.setter
    def remote_data_dir(self, p: str) -> None:

        self._remote_data_dir = p

    @property
    def remote_output_dir(self) -> str:

        return self._remote_output_dir

    @remote_output_dir.setter
    def remote_output_dir(self, p: str) -> None:

        self._remote_output_dir = p

    """
    Prefixes and suffixes
    """

    @property
    def ds_prefix(self) -> str:

        return self._ds_prefix

    @ds_prefix.setter
    def ds_prefix(self, prefix: str) -> None:

        self._ds_prefix = prefix

    """
    Training model options
    """

    @property
    def probability_model(self) -> bool:

        return self._probability_model

    @probability_model.setter
    def probability_model(self, flg: bool) -> None:

        self._probability_model = flg

    @property
    def loss_type(self) -> _ModelPERMON.LossType:

        return self._loss_type

    @loss_type.setter
    def loss_type(self, loss: _ModelPERMON.LossType) -> None:

        self._loss_type = loss

    @property
    def svm_c_value(self) -> float:

        return self._svm_c_value

    @svm_c_value.setter
    def svm_c_value(self, v: float) -> None:

        self._svm_c_value = v

    @property
    def desicion_threshold(self) -> float:

        return self._desicion_threshold

    @desicion_threshold.setter
    def desicion_threshold(self, v: float) -> None:

        self._desicion_threshold = v

    """
    Hyperparameter optimization
    """

    @property
    def run_hyperopt(self) -> bool:

        return self._run_hyperopt

    @run_hyperopt.setter
    def run_hyperopt(self, flg: bool) -> None:

        self._run_hyperopt = flg

    @property
    def hyperopt_warm_start(self) -> bool:

        return self._hyperopt_warm_start

    @hyperopt_warm_start.setter
    def hyperopt_warm_start(self, flg: bool) -> None:

        self._hyperopt_warm_start = flg

    @property
    def hyperopt_scores(self) -> _typing.Union[list[_ScoreTypes], tuple[_ScoreTypes]]:

        return self._hyperopt_scores

    @hyperopt_scores.setter
    def hyperopt_scores(self, lst) -> None:

        self._hyperopt_scores = lst

    @property
    def hyperopt_cv_nfolds(self) -> int:

        return self._hyperopt_cv_nfolds

    @hyperopt_cv_nfolds.setter
    def hyperopt_cv_nfolds(self, nfolds: int) -> None:

        self._hyperopt_cv_nfolds = nfolds

    @property
    def hyperopt_gs_logcbase(self) -> float:

        return self._hyperopt_gs_logcbase

    @hyperopt_gs_logcbase.setter
    def hyperopt_gs_logcbase(self, val: float) -> None:

        self._hyperopt_gs_logcbase = val

    @property
    def hyperopt_gs_grid_range(self) -> _typing.Union[list[float], tuple[float]]:

        return self._hyperopt_gs_grid_range

    @hyperopt_gs_grid_range.setter
    def hyperopt_gs_grid_range(self, grid: _typing.Union[list[float], tuple[float]]) -> None:

        self._hyperopt_gs_grid_range = grid

    """
    Solver settings
    """

    @property
    def solver_type(self) -> _ModelPERMON.SolverType:

        return self._solver_type

    @solver_type.setter
    def solver_type(self, solver_type: _ModelPERMON.SolverType) -> None:

        self._solver_type = solver_type

    """
    Device
    """

    @property
    def device(self) -> _DeviceType:

        return self._device

    @device.setter
    def device(self, device: _DeviceType) -> None:

        self._device = device

    @staticmethod
    def _writeShebang(f) -> None:

        shebang = [
            '#!/bin/bash --login\n'
        ]

        f.writelines(shebang)

    def _writeBatchFileDirectives(self, f) -> None:

        job_name = self.ds_prefix.upper()
        job_name = '{}_{}'.format(
            job_name,
            'PROBABILITY' if self.probability_model else 'LABEL'
        )

        if self.workload_manager_type == _WorkloadManagerType.SLURM:
            mpiprocs = self.ntasks
        else:
            raise NotImplementedError

        job_name = f'{job_name}_{self.solver_type.value.upper()}'

        if self.run_hyperopt:

            job_name = f'{job_name}_GRID_SEARCH'
            if self.hyperopt_warm_start: job_name = f'{job_name}_WITH_WARM_START'
            job_name += '\n'

        job_name = f'{job_name}_'
        if self.device == _DeviceType.CPU:
            if mpiprocs > 1: job_name = f'{job_name}_{mpiprocs}'
            job_name = f'{job_name}_CPU'
            if mpiprocs > 1: job_name = f'{job_name}S'
        else:
            job_name = '{}{}'.format(job_name, 'SINGLE_GPU' if mpiprocs == 1 else f'{mpiprocs}_GPUS')

        self._workload_manager.project_id = self.project_id
        self._workload_manager.job_name = job_name

        self._workload_manager.partition = self.partition
        self._workload_manager.ntasks = self.ntasks
        self._workload_manager.nnodes = self.nnodes

        options = self._workload_manager.getHeaderDirectives()
        f.writelines(options)

    def _writeDirectoryLocations(self, f) -> None:

        if self.remote_data_dir is None:
            raise RuntimeError('Data directory is not set!')

        if self.remote_output_dir is None:
            raise RuntimeError('Output directory is not set!')

        locations = [
            '# Directories\n\n',
            'DATA_DIR={}\n'.format(self.remote_data_dir),
            'OUTPUT_DIR={}\n'.format(self.remote_output_dir),
        ]

        f.writelines(locations)

    def _writePrefixesAndSuffixes(self, f) -> None:

        if self.ds_prefix is None:

            raise RuntimeError('Data set prefix is not set!')

        # data set prefix

        DS_PREFIX = self.ds_prefix
        if DS_PREFIX != '': DS_PREFIX = f'{DS_PREFIX}_'

        # output suffix

        OUTPUT_SUFFIX = self.solver_type.value

        if self.run_hyperopt:
            STR_GRID_SEARCH = _ModelPERMON.HyperParameterOptimization.GRID_SEARCH.value
            OUTPUT_SUFFIX = '{}_{}'.format(OUTPUT_SUFFIX, STR_GRID_SEARCH)

        if self.hyperopt_warm_start:
            STR_WARM_START = _ModelPERMON.HyperParameterOptimization.GRID_SEARCH.value
            OUTPUT_SUFFIX = '{}_{}'.format(OUTPUT_SUFFIX, STR_WARM_START)

        prefixes_suffixes = [
            '# Prefixes and suffixes\n\n',
            f'DS_PREFIX={DS_PREFIX}\n',
            f'OUTPUT_SUFFIX={OUTPUT_SUFFIX}\n'
        ]

        f.writelines(prefixes_suffixes)

    def _writeFileLocations(self, f) -> None:

        files = [
            '# Sources and outputs files\n\n',
            'FN_TRAIN=${DATA_DIR}/${DS_PREFIX}_training.h5\n',
        ]

        if self.probability_model:
            files.append('FN_CALIB=${DATA_DIR}/${DS_PREFIX}_calib.h5\n')

        files.extend([
            'FN_TEST=${DATA_DIR}/${DS_PREFIX}_test.h5\n',
            '\n',
            'FN_TRAIN_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_training_predictions_${OUTPUT_SUFFIX}.h5\n'
            'FN_TEST_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_test_predictions_${OUTPUT_SUFFIX}.h5'
        ])

        f.writelines(files)

    def _writeHyperParameterOptimizationSettings(self, f) -> None:

        if self.hyperopt_scores is None:

            raise RuntimeError('Scores for hyperparameter optimization are not set!')

        HYPEROPT_SCORES = ''

        nscores = len(self.hyperopt_scores)
        for i, s in enumerate(self.hyperopt_scores):

            score_name = '{}'.format(s.value)
            if i < nscores - 1: score_name += ','

            HYPEROPT_SCORES = f'{HYPEROPT_SCORES}{score_name}'

        hyperopt_settings = [
            '# hyperparameter optimization settings\n\n',
            'HYPEROPT_SCORE_TYPES={}'.format(HYPEROPT_SCORES),
            '\n\n',
            'CROSS_VALIDATION_TYPE=stratified_kfold\n',
            'CROSS_VALIDATION_NFOLDS={}'.format(self.hyperopt_cv_nfolds),
            '\n\n',
            'GRID_SEARCH_LOGC_BASE={}\n'.format(self.hyperopt_gs_logcbase)
        ]

        if self.hyperopt_gs_grid_range is not None:

            start = self.hyperopt_gs_grid_range[0]
            stop = self.hyperopt_gs_grid_range[1]
            step = self.hyperopt_gs_grid_range[2]

            hyperopt_settings.append(
                'GRID_SEARCH_RANGE={},{},{}\n'.format(start, stop, step)
            )

        f.writelines(hyperopt_settings)

    def _writeSolverSettings_BLMVM(self, f) -> None:

        solver_settings = [
            f'SOLVER_QPS_TYPE=tao\n',
            f'SOLVER_QPS_TAO_TYPE={self.solver_type.value}\n',
        ]

        f.writelines(solver_settings)

    def _writeSolverSettings_MPGP(self, f) -> None:

        solver_settings = [
            f'SOLVER_QPS_TYPE={self.solver_type.value}\n',
            f'MPGP_GAMMA={self.mpgp_gamma}\n'
        ]

        f.writelines(solver_settings)

    def _writeSolverSettings(self, f) -> None:

        f.write('# solver settings (csvm)\n\n')

        if self.solver_type == _ModelPERMON.SolverType.MPGP:

            self._writeSolverSettings_MPGP(f)

        elif self.solver_type == _ModelPERMON.SolverType.BLMVM:

            self._writeSolverSettings_BLMVM(f)

    def _writeTrainingSettings(self, f) -> None:

        training_settings = [
            '# settings for training model\n\n',
            f'LOSS_TYPE={self.loss_type.value}\n',
        ]

        if not self.run_hyperopt:

            training_settings.append(
                f'SVM_C={self.svm_c_value}\n'
            )

        if self.probability_model:

            training_settings.append(
                f'\nDECISION_THRESHOLD={self._desicion_threshold}\n'
            )

        f.writelines(training_settings)

    def _writeObjectSettings(self, f) -> None:

        MAT_TYPE = 'dense'
        if self.device != _DeviceType.CPU:
            MAT_TYPE = f'{MAT_TYPE}{self.device.value}'

        object_settings = [
            '# object settings\n\n'
            f'MAT_TYPE={MAT_TYPE}\n'
        ]

        if self.device != _DeviceType.CPU:
            object_settings.append(
                f'VEC_TYPE={self.device.value}\n'
            )

        f.writelines(object_settings)

    def _writeRunOptions(self, f) -> None:

        run_options = []

        if self.commands_before_run is not None:

            run_options.append('# Commands before training models\n\n')
            run_options.extend(self.commands_before_run)
            run_options.append('\n\n')

        run_options.append('# Model training\n\n')

        self._workload_manager.executable_path = self.executable_path
        run_options.extend(self._workload_manager.getRunOptions())

        if self.device == _DeviceType.CUDA:

            run_options.extend([
                '-device_enable eager -use_gpu_aware_mpi 0 \\\n',
                '-vec_type ${VEC_TYPE} \\\n'
                ]
            )

        run_options.append(
            '-f_training ${FN_TRAIN} -Xt_training_mat_type ${MAT_TYPE}'
        )

        if self.probability_model:

            run_options.append(
                ' -f_calib ${FN_CALIB} -Xt_calib_mat_type ${MAT_TYPE} '
            )

        run_options.append(
            ' \\\n'
        )

        run_options.append(
            '-f_test ${FN_TEST} -Xt_test_mat_type ${MAT_TYPE} \\\n'
        )

        run_options.append(
            '-svm_view_io \\\n'
        )

        """
        Solver settings
        """

        QPS_PREFIX = 'uncalibrated_' if self.probability_model else ''

        if self.solver_type == _ModelPERMON.SolverType.MPGP:

            run_options.extend([
                f'-{QPS_PREFIX}qps_type', '${SOLVER_QPS_TYPE} \\n',
                f'-{QPS_PREFIX}qps_mpgp_expansion_type projcg ',
                f'-{QPS_PREFIX}mpgp_gamma ', '${MPGP_GAMMA}  \\\n'
            ])

        elif self.solver_type.BLMVM:

            run_options.extend([
                f'-{QPS_PREFIX}qps_type ', '${SOLVER_QPS_TYPE} ',
                f'-{QPS_PREFIX}qps_tao_type ', '${SOLVER_QPS_TAO_TYPE} ',
            ])

        run_options.append(
            f'-{QPS_PREFIX}qps_view_convergence \\\n'
        )

        run_options.extend([
            f'-{QPS_PREFIX}svm_loss_type ', '${LOSS_TYPE} '
        ])

        if not self.run_hyperopt:

            run_options.extend([
                f'-{QPS_PREFIX}svm_C ', '${SVM_C} '
            ])

        run_options.append('\\\n')

        if self.run_hyperopt:

            run_options.extend([
                f'-{QPS_PREFIX}svm_hyperopt 1 ',
                f'-{QPS_PREFIX}svm_hyperopt_score_types ', '${HYPEROPT_SCORE_TYPES} \\\n',
                f'-{QPS_PREFIX}svm_cv_type ', '${CROSS_VALIDATION_TYPE} ',
                f'-{QPS_PREFIX}svm_nfolds ', '${CROSS_VALIDATION_NFOLDS} \\\n',
                f'-{QPS_PREFIX}svm_gs_logC_base ', '${GRID_SEARCH_LOGC_BASE} ',
                f'-{QPS_PREFIX}svm_gs_logC_stride ', '${GRID_SEARCH_RANGE} \\\n',
            ])

            if self.solver_type == _ModelPERMON.SolverType.MPGP:

                run_options.extend([
                    '-cross_qps_mpgp_expansion_type projcg ',
                    '-cross_qps_mpgp_gamma ${MPGP_GAMMA} ',
                    '-cross_qps_view_convergence \\\n',
                ])

            elif self.solver_type == _ModelPERMON.SolverType.BLMVM:

                run_options.extend([
                    '-cross_qps_type ${SOLVER_QPS_TYPE} ',
                    '-cross_qps_tao_type ${SOLVER_QPS_TAO_TYPE} ',
                    '-cross_qps_view_convergence \\\n'
                ])

            run_options.extend([
                f'-cross_svm_warm_start {int(self.hyperopt_warm_start)} \\\n',
                '-cross_svm_info -cross_svm_view_report \\\n'
            ])

        if self.probability_model:

            run_options.extend([
                '-svm_threshold ${DECISION_THRESHOLD} -tao_view -H_bce_mat_type ${MAT_TYPE} \\\n'
                '-svm_view_report \\\n'
            ])

        run_options.extend([
            '-f_training_predictions ${FN_TRAIN_PREDICTS} ',
            '-f_test_predictions ${FN_TEST_PREDICTS} \\\n'
        ])

        if self.workload_manager_type == _WorkloadManagerType.SLURM:
            nprocs = self.ntasks
        else:
            raise NotImplementedError

        STR_DEVICE = '{}_{}'.format(nprocs, 'cpu' if self.device == _DeviceType.CPU else 'gpu')
        if nprocs > 1: STR_DEVICE += 's'

        run_options.extend([
            '2>&1 | tee ${OUTPUT_DIR}/${DS_PREFIX}_',
            STR_DEVICE,
            '_${OUTPUT_SUFFIX}.log\n'
        ])

        f.writelines(run_options)

    def createScript(self) -> None:

        if self.script_name is None:

            raise RuntimeError('Name for generated script is not set!')

        with open(self.script_name, 'w') as f:

            self._writeShebang(f)
            self._writeBatchFileDirectives(f)
            f.write('\n')
            self._writeDirectoryLocations(f)
            f.write('\n')
            self._writePrefixesAndSuffixes(f)
            f.write('\n')
            self._writeFileLocations(f)
            f.write('\n')
            if self.run_hyperopt:
                self._writeHyperParameterOptimizationSettings(f)
            f.write('\n')
            self._writeSolverSettings(f)
            f.write('\n')
            self._writeTrainingSettings(f)
            f.write('\n')
            self._writeObjectSettings(f)
            f.write('\n')
            self._writeRunOptions(f)


if __name__ == '__main__':

    _os = lazy_import('os')

    VAR_PROJECT_ID = 'OPEN-27-10'
    VAR_USER = 'mari'

    VAR_DEVICE = _DeviceType.CUDA
    VAR_PARTITION = 'qdgx'

    VAR_NTASKS = 16
    VAR_NNODES = 1
    VAR_WALL_TIME = _datetime.time(hour=0, minute=10)

    VAR_SCRIPT_NAME = 'test_{}.sh'.format(VAR_DEVICE.value)

    # locations

    VAR_REMOTE_APPS_DIR = _os.path.join('/scratch/user', VAR_USER, 'apps')
    VAR_REMOTE_PROJECT_DIR = '/mnt/proj2/open-27-10/wildfires'

    VAR_DATA_DIR = _os.path.join(VAR_REMOTE_PROJECT_DIR, 'data')
    VAR_OUTPUT_DIR = _os.path.join(VAR_REMOTE_PROJECT_DIR, 'outputs')

    VAR_PERMON_SVM_DIR = _os.path.join(VAR_REMOTE_APPS_DIR, 'permonsvm')

    # prefix

    VAR_DS_PREFIX = 'ak_modis_2004_850_horizontal_pca_80'

    # training model settings

    VAR_PROBABILITY_OUTPUT = True
    VAR_LOSS_TYPE = _ModelPERMON.LossType.L1

    # hyperparameters optimization

    VAR_RUN_HYPEROPT = False
    VAR_HYPEROPT_WARM_START = True

    VAR_HYPEROPT_SCORES = (_ScoreTypes.RECALL, _ScoreTypes.JACCARD_INDEX)
    VAR_HYPEROPT_CV_NFOLDS = 3
    VAR_HYPEROPT_GRID_RANGE = (1, -2, -0.1)  # start,stop,step

    # solver settings

    VAR_SOLVER_TYPE = _ModelPERMON.SolverType.MPGP
    VAR_MPGP_GAMMA = 10

    # commands before training models

    VAR_ENV_FILE = _os.path.join(VAR_REMOTE_APPS_DIR, 'env.env')
    VAR_CMDS_BEFORE_TRAINING = (f'source {VAR_ENV_FILE}',)

    # executable

    # VAR_PRECISION = 'single'
    VAR_PRECISION = 'double'

    if VAR_PROBABILITY_OUTPUT:
        VAR_EXECUTABLE_PATH = _os.path.join(VAR_PERMON_SVM_DIR, f'src/tutorials/ex5_{VAR_PRECISION}')
    else:
        PETSC_ARCH = f'intel-cuda-{VAR_PRECISION}-opt'
        VAR_EXECUTABLE_PATH = _os.path.join(VAR_PERMON_SVM_DIR, PETSC_ARCH, 'bin/permonsvm')

    script_builder = JobScriptBuilder(
        script_name=VAR_SCRIPT_NAME,
        #
        project_id=VAR_PROJECT_ID,
        partition=VAR_PARTITION,
        nnodes=VAR_NNODES,
        ntasks=VAR_NTASKS,
        wall_time=VAR_WALL_TIME,
        # commands before training models
        commands_before_run=VAR_CMDS_BEFORE_TRAINING,
        # locations
        executable_path=VAR_EXECUTABLE_PATH,
        remote_data_dir=VAR_DATA_DIR,
        remote_output_dir=VAR_OUTPUT_DIR,
        # prefix
        ds_prefix=VAR_DS_PREFIX,
        # training model
        probability_model=VAR_PROBABILITY_OUTPUT,
        loss_type=VAR_LOSS_TYPE,
        # hyperparameters optimization
        run_hyperopt=VAR_RUN_HYPEROPT,
        hyperopt_warm_start=VAR_HYPEROPT_WARM_START,
        hyperopt_scores=VAR_HYPEROPT_SCORES,
        hyperopt_cv_nfolds=VAR_HYPEROPT_CV_NFOLDS,
        hyperopt_gs_grid_range=VAR_HYPEROPT_GRID_RANGE,
        # solver settings
        solver_type=VAR_SOLVER_TYPE,
        mpgp_gamma=VAR_MPGP_GAMMA,
        # device
        device=VAR_DEVICE
    )

    script_builder.createScript()
