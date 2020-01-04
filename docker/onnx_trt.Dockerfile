# Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VERSION=11.3.1
ARG OS_VERSION=18.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
#LABEL maintainer="NVIDIA CORPORATION"

ARG TRT=21.08
#FROM nvcr.io/nvidia/tensorrt:${TRT}-py3

ENV TRT_VERSION 8.0.1.6
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    vim

RUN apt-get install -y python3-pip
# Install PyPI packages
RUN pip3 install --upgrade pip

# Install PyPI packages
# RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
RUN pip3 install jupyter jupyterlab
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc && rm ngccli_cat_linux.zip ngc.md5 && echo "no-apikey\nascii\n" | ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

RUN pip3 install tensorflow==2.6.2 onnx==1.10.1 tf2onnx==1.9.1 

RUN git clone https://github.com/protocolbuffers/protobuf.git && \
    cd protobuf && \
    git checkout tags/v3.8.0 && \
    git submodule update --init --recursive && \
    ./autogen.sh && \
    ./configure && \
    make -j8 && \
    make install && \
    ldconfig && \
    make clean && \
    cd .. && \
    rm -r protobuf

RUN pip3 install torch torchvision torchaudio

RUN apt-get update && apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6

RUN pip3 install opencv_python==4.2.0.32 Shapely==1.7.0 xmljson==0.2.0 thop==0.0.31.post2005241907 matplotlib==3.1.3 imgaug==0.4.0 scipy==1.4.1 p_tqdm==1.3.3 lxml==4.5.0 tqdm==4.43.0 ujson==1.35 PyYAML==5.3.1 scikit_learn==0.23.2 tensorboard==2.6.0 gdown==3.12.2

#RUN pip3 install nvidia-tensorrt==7.2.* --index-url https://pypi.ngc.nvidia.com

RUN apt-get install -y graphviz

RUN pip3 install torchsummary torchviz

RUN pip3 install onnx-simplifier==0.3.5 loguru Pillow ninja tabulate onnxruntime==1.8.0 pathspec addict

RUN mkdir -p ~/opencv cd ~/opencv && \
    wget https://github.com/opencv/opencv/archive/4.5.3.tar.gz && \
    tar -zxvf 4.5.3.tar.gz && \
    rm 4.5.3.tar.gz && \
    mv opencv-4.5.3 OpenCV && \
    cd OpenCV && \
    wget https://github.com/opencv/opencv_contrib/archive/4.5.3.tar.gz && \
    echo $PWD && \
    tar -zxvf 4.5.3.tar.gz && \
    rm 4.5.3.tar.gz && \
    mkdir build && \ 
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv \
      -D WITH_CUDA=OFF -D CUDA_ARCH_BIN="7.5" -D CUDA_ARCH_PTX="" \
      -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON \
      -D OPENCV_EXTRA_MODULES_PATH=$PWD/../opencv_contrib-4.5.3/modules \
      -D OPENCV_GENERATE_PKGCONFIG=on -D ENABLE_PRECOMPILED_HEADERS=OFF .. && \
    make -j8 && \
    make install && \ 
    ldconfig

RUN pip3 install pytorch-lightning==1.2.0
RUN pip3 install --upgrade wandb
RUN pip3 install pandas seaborn requests

RUN cd /root && git clone https://github.com/tencent/ncnn
RUN mkdir -p /root/ncnn/build && cd /root/ncnn/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && make -j$(nproc) && make install

RUN apt-get update && apt-get install -y libglfw3 libglfw3-dev

WORKDIR /home/euclid

#USER trtuser
RUN ["/bin/bash"]


