PERMON_SVM_DIR=$1
PETSC_ARCH=$2
TOOLS_DIR=$3

echo $PERMON_SVM_DIR

DATA_DIR=${TOOLS_DIR}/data/h5/mtbs
OUTPUT_DIR=${TOOLS_DIR}/output/permonsvm

DS_PREFIX=ak_modis_2004_2005_13k

FN_TRAIN=${DATA_DIR}/${DS_PREFIX}_labels_training.h5
FN_TEST=${DATA_DIR}/${DS_PREFIX}_labels_test.h5

FN_TRAIN_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_training_labels_predictions.h5
FN_TEST_PREDICTS=${OUTPUT_DIR}/${DS_PREFIX}_test_labels_predictions.h5

LOSS_TYPE=L1
MAT_TYPE=dense

${PERMON_SVM_DIR}/${PETSC_ARCH}/bin/permonsvmfile \
-fllop_trace \
-f_training ${FN_TRAIN} -Xt_training_mat_type ${MAT_TYPE} \
-f_test ${FN_TEST} -Xt_test_mat_type ${MAT_TYPE} \
-svm_view_io \
-qps_mpgp_expansion_type projcg -qps_mpgp_gamma 10 -qps_view_convergence \
-svm_loss_type ${LOSS_TYPE} -svm_binary_mod 2 -svm_C 0.01 \
-svm_view_report \
-f_training_predictions ${FN_TRAIN_PREDICTS} -f_test_predictions ${FN_TEST_PREDICTS}
