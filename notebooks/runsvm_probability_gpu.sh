#!/bin/bash --login

PERMONSVM_BIN=$1

FN_TRAIN=$2
FN_CALIB=$3
FN_TEST=$4

FN_TRAIN_PREDICTS=$5
FN_TEST_PREDICTS=$6

LOSS_TYPE=L1

MAT_TYPE=densecuda
VEC_TYPE=cuda

mpirun -np 2 \
${PERMONSVM_BIN} \
-fllop_trace \
-device_enable eager -use_gpu_aware_mpi 0 \
-vec_type ${VEC_TYPE} \
-f_training ${FN_TRAIN} -Xt_training_mat_type ${MAT_TYPE} -f_calib ${FN_CALIB} -Xt_calib_mat_type ${MAT_TYPE} \
-f_test ${FN_TEST} -Xt_test_mat_type ${MAT_TYPE} \
-svm_view_io \
-uncalibrated_qps_mpgp_expansion_type projcg -uncalibrated_qps_mpgp_gamma 10 -uncalibrated_qps_view_convergence \
-uncalibrated_svm_loss_type ${LOSS_TYPE} -uncalibrated_svm_binary_mod 2 -uncalibrated_svm_C 0.01 \
-svm_threshold 0.50 -tao_view \
-svm_view_report \
-f_training_predictions ${FN_TRAIN_PREDICTS} -f_test_predictions ${FN_TEST_PREDICTS}
