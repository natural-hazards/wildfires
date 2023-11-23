#!/bin/bash --login

PERMONSVM_BIN=$1

FN_TRAIN=$2
FN_TEST=$3

FN_TRAIN_PREDICTS=$4
FN_TEST_PREDICTS=$5

LOSS_TYPE=L1
MAT_TYPE=dense

${PERMONSVM_BIN} \
-fllop_trace \
-f_training ${FN_TRAIN} -Xt_training_mat_type ${MAT_TYPE} \
-f_test ${FN_TEST} -Xt_test_mat_type ${MAT_TYPE} \
-svm_view_io \
-qps_mpgp_expansion_type projcg -qps_mpgp_gamma 10 -qps_view_convergence \
-svm_loss_type ${LOSS_TYPE} -svm_binary_mod 2 -svm_C 0.01 \
-svm_view_report \
-f_training_predictions ${FN_TRAIN_PREDICTS} -f_test_predictions ${FN_TEST_PREDICTS}
