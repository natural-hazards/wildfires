from mlfire.utils.functool import lazy_import

# lazy imports
_datetime = lazy_import('datetime')
_enum = lazy_import('enum')
_os = lazy_import('os')
_typing = lazy_import('typing')


class SolverType(_enum.Enum):

    MPGP = 'mpgp'
    BLMVM = 'blmvm'


class ModelOutput(_enum.Enum):

    LABEL = 'label'
    PROBABILITY = 'probability'


class HyperParameterOptimization(_enum.Enum):

    GRID_SEARCH = 'grid_search'
    WARM_START = 'with_warm_start'


class LossType(_enum.Enum):

    L1 = 'L1'
    L2 = 'L2'


class ScoreType(_enum.Enum):

    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'F1'
    JACCARD_INDEX = 'jaccard'
    AUC_ROC = 'auc_roc'


class Device(_enum.Enum):

    CPU = 'cpu'
    CUDA = 'cuda'
    HIP = 'hip'


class BSubScriptBuilder(object):

    def __init__(self,
                 script_name: str = None,
                 project_id: str = None,
                 wall_time: _datetime.time = _datetime.time(hour=0, minute=10),
                 alloc_flags: str = 'smt4 gpumps',
                 # resources
                 nnodes: int = 1,
                 resources: int = 1,
                 ranks_rs: int = 4,
                 ncores: int = 7,
                 # locations
                 remote_data_dir: str = None,
                 remote_home_dir: str = None,
                 petsc_arch: str = None,
                 permonsvm_dir: str = None,
                 remote_output_dir: str = None,
                 # prefix
                 ds_prefix: str = None,
                 # training model settings
                 probability_model: bool = False,
                 loss_type: LossType = LossType.L1,
                 svm_c_value: float = .01,
                 desicion_threshold: float = .5,
                 # hyperparameters optimization
                 run_hyperopt: bool = False,
                 hyperopt_warm_start: bool = False,
                 hyperopt_scores: _typing.Union[list[ScoreType], tuple[ScoreType]] = (ScoreType.JACCARD_INDEX,),
                 hyperopt_cv_nfolds: int = 3,
                 hyperopt_gs_logcbase: float = 10.,
                 hyperopt_gs_grid_range: _typing.Union[list[float], tuple[float]] = None,
                 # solver
                 solver_type: SolverType = SolverType.MPGP,
                 mpgp_gamma: float = 10.,
                 # device
                 device: Device = Device.CUDA):

        self._script_name = None
        self.script_name = script_name

        #

        self._project_id = None
        self.project_id = project_id

        self._wall_time = None
        self.wall_time = wall_time

        self._alloc_flags = None
        self.alloc_flags = alloc_flags

        self._nnodes = None
        self.nnodes = nnodes

        self._resources = None
        self.resources = resources

        self._rank_rs = None
        self.rank_rs = ranks_rs

        self._ncores = None
        self.ncores = ncores

        # locations

        self._remote_home_dir = None
        self.remote_home_dir = remote_home_dir

        self._permonsvm_dir = None
        self.permonsvm_dir = permonsvm_dir

        self._petsc_arch = None
        self.petsc_arch = petsc_arch

        self._remote_data_dir = None
        self.remote_data_dir = remote_data_dir

        self._remote_output_dir = None
        self.remote_output_dir = remote_output_dir

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
    def script_name(self) -> str:

        return self._script_name

    @script_name.setter
    def script_name(self, n: str) -> None:

        self._script_name = n

    @property
    def project_id(self) -> str:

        return self._project_id

    @project_id.setter
    def project_id(self, pid) -> None:

        self._project_id = pid

    @property
    def wall_time(self) -> _datetime.time:

        return self._wall_time

    @wall_time.setter
    def wall_time(self, wtime: _datetime.time) -> None:

        self._wall_time = wtime

    @property
    def nnodes(self) -> int:

        return self._nnodes

    @nnodes.setter
    def nnodes(self, nnodes) -> None:

        self._nnodes = nnodes

    @property
    def resources(self) -> int:

        return self._resources

    @resources.setter
    def resources(self, rs: int) -> None:

        self._resources = rs

    @property
    def rank_rs(self) -> int:

        return self._rank_rs

    @rank_rs.setter
    def rank_rs(self, v: int) -> None:

        self._rank_rs = v

    @property
    def ncores(self) -> int:

        return self._ncores

    @ncores.setter
    def ncores(self, c: int) -> None:

        self._ncores = c

    @property
    def alloc_flags(self) -> str:

        return self._alloc_flags

    @alloc_flags.setter
    def alloc_flags(self, flags) -> None:

        self._alloc_flags = flags

    """
    Locations
    """

    @property
    def remote_data_dir(self) -> str:

        return self._remote_data_dir

    @remote_data_dir.setter
    def remote_data_dir(self, p: str) -> None:

        self._remote_data_dir = p

    @property
    def remote_home_dir(self) -> str:

        return self._remote_home_dir

    @remote_home_dir.setter
    def remote_home_dir(self, p: str) -> None:

        self._remote_home_dir = p

    @property
    def petsc_arch(self) -> str:

        return self._petsc_arch

    @petsc_arch.setter
    def petsc_arch(self, arch: str) -> None:

        self._petsc_arch = arch

    @property
    def remote_output_dir(self) -> str:

        return self._remote_output_dir

    @remote_output_dir.setter
    def remote_output_dir(self, p: str) -> None:

        self._remote_output_dir = p

    @property
    def permonsvm_dir(self) -> str:

        return self._permonsvm_dir

    @permonsvm_dir.setter
    def permonsvm_dir(self, p: str) -> None:

        self._permonsvm_dir = p

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
    def loss_type(self) -> LossType:

        return self._loss_type

    @loss_type.setter
    def loss_type(self, lt: LossType) -> None:

        self._loss_type = lt

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
    def hyperopt_scores(self) -> _typing.Union[list[ScoreType], tuple[ScoreType]]:

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
    def solver_type(self) -> SolverType:

        return self._solver_type

    @solver_type.setter
    def solver_type(self, t: SolverType) -> None:

        self._solver_type = t

    """
    Device
    """

    @property
    def device(self) -> Device:

        return self._device

    @device.setter
    def device(self, dev: Device) -> None:

        self._device = dev

    @staticmethod
    def _writeShebang(f) -> None:

        shebang = [
            '#!/bin/bash\n'
        ]

        f.writelines(shebang)

    def _writeBsubOptions(self, f) -> None:

        str_wtime_hours = '{:02d}'.format(self.wall_time.hour)
        str_wtime_minutes = '{:02d}'.format(self.wall_time.minute)

        job_name = self.ds_prefix.upper()
        job_name = '{}_{}'.format(
            job_name,
            'PROBABILITY' if self.probability_model else 'LABEL'
        )

        if self.device == Device.CPU:
            job_name = f'{job_name}_CPUS'
        else:
            if self.resources == 1:
                job_name = f'{job_name}_SINGLE_GPU'
            else:
                job_name = f'{job_name}_MULTIPLE_GPU'

        job_name = f'{job_name}_{self.solver_type.value.upper()}'

        if self.run_hyperopt:

            job_name = f'{job_name}_GRID_SEARCH'
            if self.hyperopt_warm_start: job_name = f'{job_name}_WITH_WARM_START'
            job_name += '\n'

        options = [
            '### Begin BSUB Options\n',
            f'# BSUB -P {self.project_id}\n',
            f'# BSUB -J {job_name}',
            '# BSUB -W {}:{}\n'.format(str_wtime_hours, str_wtime_minutes),
            f'# BSUB -nnodes {self.nnodes}\n',
            '# BSUB -alloc_flags "{}"\n'.format(self.alloc_flags),
            '### End BSUB Options and begin shell commands\n\n',
            f'RS={self.nnodes * self.resources}\n',
            f'RANKS_PER_RS={self.rank_rs}\n',
            f'CORES={self.ncores}\n'
        ]

        f.writelines(options)

    def _writeDirectoryLocations(self, f) -> None:

        if self.remote_data_dir is None:
            raise RuntimeError('Data directory is not set!')

        if self.remote_home_dir is None:
            raise RuntimeError('Home directory is not set!')

        if self.remote_output_dir is None:
            raise RuntimeError('Output directory is not set!')

        if self.permonsvm_dir is None:
            raise RuntimeError('Location of PermonSVM is not set!')

        locations = [
            '# Directories\n\n',
            'HOME_DIR={}\n\n'.format(self.remote_home_dir),
            'DATA_DIR={}\n'.format(self.remote_data_dir),
            'OUTPUT_DIR={}\n\n'.format(self.remote_output_dir),
            'PERMON_SVM_DIR={}\n'.format(self.permonsvm_dir),
        ]

        if not self.probability_model:

            locations.append(
                'PERMON_ARCH={}\n'.format(self.petsc_arch)
            )

        f.writelines(locations)

    def _writePrefixesAndSuffixes(self, f) -> None:

        if self.ds_prefix is None:

            raise RuntimeError('Data set prefix is not set!')

        # data set prefix

        DS_PREFIX = self.ds_prefix
        if DS_PREFIX != '': DS_PREFIX = f'{DS_PREFIX}_'

        if self.probability_model:
            DS_PREFIX = '{}{}'.format(DS_PREFIX, ModelOutput.PROBABILITY.value)
        else:
            DS_PREFIX = '{}{}'.format(DS_PREFIX, ModelOutput.LABEL.value)

        # output suffix

        OUTPUT_SUFFIX = self.solver_type.value

        if self.run_hyperopt:
            OUTPUT_SUFFIX = '{}_{}'.format(OUTPUT_SUFFIX, HyperParameterOptimization.GRID_SEARCH.value)

        if self.hyperopt_warm_start:
            OUTPUT_SUFFIX = '{}_{}'.format(OUTPUT_SUFFIX, HyperParameterOptimization.WARM_START.value)

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
            'FN_TEST_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_test_predictions_${OUTPUT_SUFFIX}.h5\n'
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

        if self.solver_type == SolverType.MPGP:

            self._writeSolverSettings_MPGP(f)

        elif self.solver_type == SolverType.BLMVM:

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
        if self.device != Device.CPU:
            MAT_TYPE = f'{MAT_TYPE}{self.device.value}'

        object_settings = [
            '# object settings\n\n'
            f'MAT_TYPE={MAT_TYPE}\n'
        ]

        if self.device != Device.CPU:
            object_settings.append(
                f'VEC_TYPE={self.device.value}\n'
            )

        f.writelines(object_settings)

    def _writeJSRunOptions(self, f) -> None:

        if self.probability_model:
            BIN_FILE = '${PERMON_SVM_DIR}/src/tutorials/ex5'
        else:
            BIN_FILE = '${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/permonsvmfile'

        jsrun_options = [
            'jsrun ',
            '--smpiargs="-gpu"' if self.device == Device.CUDA else '',
            '-n ${RS} -a ${RANK_RS} -c ${CORES} ',
            '-g 1' if self.device != Device.CPU else '',
            '\\\n',
            f'{BIN_FILE} \\\n',
        ]

        if self.device == Device.CUDA:

            jsrun_options.extend([
                '-device_enable eager -use_gpu_aware_mpi 0 \\\n',
                '-vec_type ${VEC_TYPE} \\\n'
                ]
            )

        jsrun_options.append(
            '-f_training ${FN_TRAIN} -Xt_training_mat_type ${MAT_TYPE}'
        )

        if self.probability_model:

            jsrun_options.append(
                '-f_calib ${FN_CALIB} - Xt_calib_mat_type ${MAT_TYPE} '
            )

        jsrun_options.append(
            ' \\\n'
        )

        jsrun_options.append(
            '-f_test ${FN_TEST} - Xt_test_mat_type ${MAT_TYPE} \\\n'
        )

        jsrun_options.append(
            '-svm_view_io \\\n'
        )

        """
        Solver settings
        """

        QPS_PREFIX = 'uncalibrated_' if self.probability_model else ''

        if self.solver_type == SolverType.MPGP:

            jsrun_options.extend([
                f'-{QPS_PREFIX}qps_mpgp_expansion_type projcg ',
                f'-{QPS_PREFIX}mpgp_gamma ', '${MPGP_GAMMA}  \\\n'
            ])

        elif self.solver_type.BLMVM:

            jsrun_options.extend([
                f'-{QPS_PREFIX}qps_type ', '${SOLVER_QPS_TYPE} ',
                f'-{QPS_PREFIX}qps_tao_type ', '${SOLVER_QPS_TAO_TYPE} ',
            ])

        jsrun_options.append(
            f'-{QPS_PREFIX}qps_view_convergence \\\n'
        )

        jsrun_options.extend([
            f'-{QPS_PREFIX}svm_loss_type ', '${LOSS_TYPE} '
        ])

        if not self.run_hyperopt:

            jsrun_options.extend([
                f'-{QPS_PREFIX}svm_C ', '${SVM_C} '
            ])

        jsrun_options.append('\\\n')

        if self.run_hyperopt:

            jsrun_options.extend([
                f'-{QPS_PREFIX}svm_hyperopt 1 ',
                f'-{QPS_PREFIX}svm_hyperopt_score_types ', '${HYPEROPT_SCORE_TYPES} \\\n',
                f'-{QPS_PREFIX}svm_cv_type ', '${CROSS_VALIDATION_TYPE} ',
                f'-{QPS_PREFIX}svm_nfolds ', '${CROSS_VALIDATION_NFOLDS} \\\n',
                f'-{QPS_PREFIX}svm_gs_logC_base ', '${GRID_SEARCH_LOGC_BASE} ',
                f'-{QPS_PREFIX}svm_gs_logC_stride ', '${GRID_SEARCH_RANGE} \\\n',
            ])

            if self.solver_type == SolverType.MPGP:

                jsrun_options.extend([
                    '-cross_qps_mpgp_expansion_type projcg ',
                    '-cross_qps_mpgp_gamma ${MPGP_GAMMA} ',
                    '-cross_qps_view_convergence \\\n',
                ])

            elif self.solver_type == SolverType.BLMVM:

                jsrun_options.extend([
                    '-cross_qps_type ${SOLVER_QPS_TYPE} ',
                    '-cross_qps_tao_type ${SOLVER_QPS_TAO_TYPE} ',
                    '-cross_qps_view_convergence \\\n'
                ])

            jsrun_options.extend([
                f'-cross_svm_warm_start {int(self.hyperopt_warm_start)} \\\n',
                '-cross_svm_info -cross_svm_view_report \\\n'
            ])

        if self.probability_model:

            jsrun_options.extend([
                '-svm_threshold ${DECISION_THRESHOLD} -tao_view -H_bce_mat_type ${MAT_TYPE} \\\n'
                '-svm_view_report \\\n'
            ])

        jsrun_options.extend([
            '-f_training_predictions ${FN_TRAIN_PREDICTS} ',
            '-f_test_predictions ${FN_TEST_PREDICTS} \\\n'
        ])

        if self.device == Device.CPU:
            nprocs = self.nnodes * self.resources * self.rank_rs
            STR_DEVICE = '{}_cpu'.format(nprocs)
            if nprocs > 1: STR_DEVICE += 's'
        else:
            if self.resources == 1:
                STR_DEVICE = 'single_gpu'
            else:
                STR_DEVICE = 'multiple_gpu'

        jsrun_options.extend([
            '2>&1 | tee ${OUTPUT_DIR}/${DS_PREFIX}_',
            STR_DEVICE,
            '_${OUTPUT_SUFFIX}.log\n'
        ])

        f.writelines(jsrun_options)

    def createScript(self) -> None:

        if self.script_name is None:
            raise RuntimeError('Name for generated script is not set!')

        with open(self.script_name, 'w') as f:

            self._writeShebang(f)
            self._writeBsubOptions(f)
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
            self._writeJSRunOptions(f)


if __name__ == '__main__':

    VAR_PROJECT_ID = 'CSC314'
    VAR_USER = 'pecham'

    VAR_WALL_TIME = _datetime.time(hour=0, minute=10)
    VAR_RESOURCES = 1
    VAR_RANKS_PER_RESOURCE = 4
    VAR_NCORES = 7

    VAR_DEVICE = Device.CPU

    VAR_SCRIPT_NAME = 'test_{}.sh'.format(VAR_DEVICE.value)

    # locations

    VAR_REMOTE_HOME_DIR = '/gpfs/alpine/proj-shared/{}/{}'.format(VAR_PROJECT_ID.lower(), VAR_USER)

    VAR_PERMON_SVM_DIR = '${VAR_REMOTE_HOME_DIR}/permon/permonsvm'
    VAR_PETSC_ARCH = 'arch-olcf-summit-double-precision-gcc-O3'

    VAR_DATA_DIR = '${VAR_REMOTE_HOME_DIR}/data'
    VAR_OUTPUT_DIR = '${VAR_REMOTE_HOME_DIR}/outputs'

    # prefix

    VAR_DS_PREFIX = 'ak_modis_2004_2005_100km'

    # training model settings

    VAR_PROBABILITY_OUTPUT = True
    VAR_LOSS_TYPE = LossType.L1

    # hyperparameters optimization

    VAR_RUN_HYPEROPT = True
    VAR_HYPEROPT_WARM_START = True

    VAR_HYPEROPT_SCORES = (ScoreType.RECALL, ScoreType.JACCARD_INDEX)
    VAR_HYPEROPT_CV_NFOLDS = 3
    VAR_HYPEROPT_GRID_RANGE = (1, -2, -0.1)  # start,stop,step

    # solver settings

    VAR_SOLVER_TYPE = SolverType.MPGP
    VAR_MPGP_GAMMA = 10

    script_builder = BSubScriptBuilder(
        script_name=VAR_SCRIPT_NAME,
        project_id=VAR_PROJECT_ID,
        wall_time=VAR_WALL_TIME,
        resources=VAR_RESOURCES,
        ranks_rs=VAR_RANKS_PER_RESOURCE,
        # locations
        remote_home_dir=VAR_REMOTE_HOME_DIR,
        permonsvm_dir=VAR_PERMON_SVM_DIR,
        petsc_arch=VAR_PETSC_ARCH,
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
