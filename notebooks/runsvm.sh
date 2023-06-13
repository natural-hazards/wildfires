PERMON_SVM_DIR=/Users/marek/Playground/permon/permonsvm
TOOL_DIR=/Users/marek/Playground/wildfires

DATA_DIR=${TOOL_DIR}/data/h5/mtbs
OUTPUT_DIR=${TOOL_DIR}/output/permonsvm

DS_PREFIX=ak_modis_2004_2005_100km

FN_TRAIN=${DATA_DIR}/${DS_PREFIX}_training.h5
FN_CALIB=${DATA_DIR}/${DS_PREFIX}_calib.h5
FN_TEST=${DATA_DIR}/${DS_PREFIX}_test.h5

FN_TRAIN_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_training_predictions.h5
FN_TEST_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_test_predictions.h5

${PERMON_SVM_DIR}/src/tutorials/ex5 -f_training ${FN_TRAIN} -Xt_training_mat_type dense -f_calib ${FN_CALIB} -Xt_calib_mat_type dense -svm_view_io \
-fllop_trace -uncalibrated_qps_mpgp_expansion_type projcg -uncalibrated_qps_mpgp_gamma 10 -uncalibrated_svm_C 0.01 \
-uncalibrated_svm_loss_type L2 -f_test ${FN_TEST} -Xt_test_mat_type dense -svm_view_report -svm_threshold 0.50 -tao_view \
-f_training_predictions ${FN_TRAIN_PREDICTS} -f_test_predictions ${FN_TEST_PREDICTS} -uncalibrated_qps_view_convergence \
-uncalibrated_svm_binary_mod 2
