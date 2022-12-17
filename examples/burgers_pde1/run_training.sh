#!/bin/bash
# Note, first need to run gen_parameter_files.sh

BASE_DIR=./script_data/study_0001
RUN_I=00000

CASE='VAE__Analytic_Projection'
#CASE=VAE__Point_Cloud_Projection_00000
#CASE='VAE__No_Projection'
#CASE='AE__Analytic_Projection'
#CASE='AE__No_Projection'
#CASE='VAE__2d'
#CASE='VAE__10d'

python ./train_model_periodic1.py -p ${BASE_DIR}/${CASE}_${RUN_I}/params.pickle

