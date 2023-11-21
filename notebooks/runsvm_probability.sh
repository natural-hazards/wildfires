PERMON_SVM_DIR=$1
TOOLS_DIR=$2

DATA_DIR=${TOOLS_DIR}/data/h5/mtbs
OUTPUT_DIR=${TOOLS_DIR}/output/permonsvm

DS_PREFIX=ak_modis_2004_2005_13k

FN_TRAIN=${DATA_DIR}/${DS_PREFIX}_probability_training.h5
FN_CALIB=${DATA_DIR}/${DS_PREFIX}_probability_calib.h5
FN_TEST=${DATA_DIR}/${DS_PREFIX}_probability_test.h5

FN_TRAIN_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_training_probability_predictions.h5
FN_TEST_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_test_probability_predictions.h5

LOSS_TYPE=L1
MAT_TYPE=dense

${PERMON_SVM_DIR}/src/tutorials/ex5 \
-fllop_trace \
-f_training ${FN_TRAIN} -Xt_training_mat_type ${MAT_TYPE} -f_calib ${FN_CALIB} -Xt_calib_mat_type ${MAT_TYPE} \
-f_test ${FN_TEST} -Xt_test_mat_type ${MAT_TYPE} \
-svm_view_io \
-uncalibrated_qps_mpgp_expansion_type projcg -uncalibrated_qps_mpgp_gamma 10 -uncalibrated_qps_view_convergence \
-uncalibrated_svm_loss_type ${LOSS_TYPE} -uncalibrated_svm_binary_mod 2 -uncalibrated_svm_C 0.01 \
-svm_threshold 0.50 -tao_view \
-svm_view_report \
-f_training_predictions ${FN_TRAIN_PREDICTS} -f_test_predictions ${FN_TEST_PREDICTS}
