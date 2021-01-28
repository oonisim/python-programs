#!/bin/bash
#--------------------------------------------------------------------------------
# Anacondai installation
# https://docs.anaconda.com/anaconda/install/silent-mode/#
#--------------------------------------------------------------------------------
CONDA_INSTALLER='Anaconda3-2019.10-Linux-x86_64.sh'
(cd /tmp && curl -O https://repo.anaconda.com/archive/${CONDA_INSTALLER})
/bin/bash /tmp/${CONDA_INSTALLER} -b -f -p $HOME/conda

echo -e '\nexport PATH=$HOME/conda/bin:$PATH' >> $HOME/.bashrc

source $HOME/.bashrc
conda config --set auto_activate_base true
conda init

#--------------------------------------------------------------------------------
# Environment
#--------------------------------------------------------------------------------
export ENV=jupyter_notbook

conda create -n ${ENV} python=3.6
conda activate ${ENV}
conda  install -y jupyter notebook scipy numpy scikit-learn pandas seaborn matplotlib jupyter notebook scikit-learn