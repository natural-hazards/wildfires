#!/bin/bash --login

# get temperature (2004)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C5Hgjw1PCK4MetttkfMOPr9AsmdqDeK1' \
-O ak_lst_january_december_2004_13k_epsg3338_area_0.tif

# get temperature (2005)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Od_mSpXTe2_3QDRb4CaRYh5ip7XqZhzx' \
-O ak_lst_january_december_2005_13k_epsg3338_area_0.tif

# get reflectance (2004)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sC8wB6dCO2cc7fZVWKtEKE278GLzsALJ' \
-O ak_reflec_january_december_2004_13k_epsg3338_area_0.tif

# get reflectance (2005)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1siTPYZvcDbGYuQL_1OORMd6mEWpwiDjN' \
-O ak_reflec_january_december_2005_13k_epsg3338_area_0.tif

# get MTBS labels (2004)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-v3fF9m-ryhTP42zh8vXguIsgH5KMOwl' \
-O ak_january_december_2004_13k_epsg3338_area_0_mtbs_labels.tif

# get MTBS labels (2005)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CBLAGOHyK_MALOGGOdR2ZMgE5Yqz5JK4' \
-O ak_january_december_2005_13k_epsg3338_area_0_mtbs_labels.tif