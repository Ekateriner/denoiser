#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

path=egs/full/tr
if test ! -e $path ; then
    mkdir -p $path
fi
python3 -m denoiser.audio noisy_trainset_28spk_wav > $path/noisy.json
python3 -m denoiser.audio clean_trainset_28spk_wav > $path/clean.json

path=egs/full/tt
if test ! -e $path ; then
    mkdir -p $path
fi
python3 -m denoiser.audio noisy_testset_wav > $path/noisy.json
python3 -m denoiser.audio clean_testset_wav > $path/clean.json

path=egs/full/ch
if test ! -e $path ; then
    mkdir -p $path
fi
python3 -m denoiser.audio dataset/debug/noisy > $path/noisy.json
