#!/bin/bash
#set -x
 
# https://code.ornl.gov/jqyin/mldl-hpc/tree/master/utils
# https://developer.ibm.com/tutorials/install-pytorch-on-power/
 
source /etc/profile.d/z00_lmod.sh
# module load python/3.7.2
module load cuda/11.1
module load gcc/8
 
# point to the OpenBLAS and Magma installs
export LD_LIBRARY_PATH=`pwd`/magma-2.5.4/install/lib:`pwd`/OpenBLAS-0.3.13/install/lib:$LD_LIBRARY_PATH
export MAGMA_HOME=`pwd`/magma-2.5.4/install
 
# install anaconda to install dependencies needed to build pytorch
# installdir=`pwd`/anaconda
# bash /collab/usr/gapps/python/blueos_3_ppc64le_ib/conda/Anaconda3-2020.02-Linux-ppc64le.sh -b -f -p $installdir
 
# activate conda
# source $installdir/bin/activate
 
# configure conda to use LLNL certificates for web downloads
conda config --set ssl_verify /etc/pki/tls/cert.pem
 
# create a new python-3.7 environment
# conda create -y -n torch181 python=3.7
# conda activate torch181
 
# install dependencies needed to build pytorch
conda install numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
 
# building in /dev/shm makes things much faster
pushd /dev/shm
 
# clone the python repo
rm -rf pytorch
git clone --recursive https://github.com/pytorch/pytorch
pushd pytorch
  # checkout a particular PyTorch version
  git checkout v1.9.1
  
  # update submodules to match version we checked out
  git submodule sync
  git submodule update --init --recursive
 
  # be sure CMake picks up our newer gcc, rather than /usr/tce/bin/cc
  export CC=gcc
  export CXX=g++
 
  # disable ROCM, since pytorch can't be built with ROCM and CUDA
  export USE_ROCM=OFF
  
  # point to system install of CUDNN
  export USE_CUDNN=1
  export CUDNN_HOME=/collab/usr/global/tools/nvidia/cudnn/blueos_3_ppc64le_ib_p9/cudnn-7.5.1.10
  export CUDNN_ROOT=${CUDNN_HOME}
  export CUDNN_INCLUDE_DIR=${CUDNN_HOME}/include
  export CUDNN_LIB_DIR=${CUDNN_HOME}/lib
  export LD_LIBRARY_PATH=${CUDNN_HOME}/lib:$LD_LIBRARY_PATH
  
  # point to system install of NCCL
  export USE_NCCL=1
  export NCCL_ROOT_DIR=/usr/global/tools/nvidia/nccl/blueos_3_ppc64le_ib/nccl_2.4.2-1+cuda10.1_ppc64le
  export LD_LIBRARY_PATH=${NCCL_ROOT_DIR}/lib:$LD_LIBRARY_PATH
  
  # build the pytorch wheel file
  export USE_DISTRIBUTED=1
  python setup.py bdist_wheel
  
  # install the pytorch wheel, use the --no-deps to avoid installing torchvision and others that won't work
  ls dist/torch-*.whl
  pip install --no-dependencies dist/torch-*.whl
 
  #pip install torchvision
popd
 
# pop out of /dev/shm
popd