#!/bin/bash

path=test_dir
if test ! -e $path ; then
    mkdir -p $path
fi
python3 -m denoiser.audio noisy_testset_wav > $path/noisy.json
python3 -m denoiser.audio clean_testset_wav > $path/clean.json
