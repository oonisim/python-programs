#!/usr/bin/env bash
ln -s ../model/model.npy .
ln -s ../training/simple_linear_regr.py .
ln -s ../training/simple_linear_regr_utils.py .

python main.py

rm model.npy
rm simple_linear_regr.py
rm simple_linear_regr_utils.py
