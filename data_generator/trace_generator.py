# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i) Kaggle Display Advertising Challenge Dataset
#     https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/

from __future__ import absolute_import, division, print_function, unicode_literals

# numpy, pandas
import numpy as np
import pandas as pd
from numpy import random as ra

# others
import bisect
import collections
import random
import time


# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1

def read_dist_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
            print("len(lines) = ", len(lines))
    except Exception:
        print("Wrong file or file path")
    # read cumulative distribution (elements are passed as two separate lists)
    list_sd = [int(el) for el in lines[0].split(", ")]
    cumm_sd = [float(el) for el in lines[1].split(", ")]

    return list_sd, cumm_sd

# stack distance sampling and generation
def generate_stack_distance(cumm_val, cumm_dist, max_i, i, enable_padding=False):
    # cumm_val: stack distances
    # cumm_dist: cumm probability of stack distances
    # max_i: max stack distances
    # i: current counter of new references in the generated trace

    # u samples between [0, 1]
    u = ra.rand(1)
    if i < max_i:
        # current new reference less than the max stack distance (bounded by cache size)
        # only generate stack distances up to the number of new references seen so far
        j = bisect.bisect(cumm_val, i) - 1
        fi = cumm_dist[j]
        u *= fi  # shrink distribution support to exclude last values
    elif enable_padding:
        # WARNING: disable generation of new references (once all have been seen)
        fi = cumm_dist[0]
        u = (1.0 - fi) * u + fi  # remap distribution support to exclude first value

    for (j, f) in enumerate(cumm_dist):
        if u <= f:
            return cumm_val[j]


def trace_generate_lru(table_size, list_sd, cumm_sd, out_trace_len, enable_padding=False):
    line_accesses = random.sample(range(table_size), table_size) # unique indices in the table lookup
    max_sd = list_sd[-1] # max stack distance

    l = len(line_accesses) # num of unique indices
    i = 0 # count new references

    ztrace = [] # new generated syn traces

    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0
        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses.pop(0)
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            line_accesses.pop(l - sd)
            line_accesses.append(line_ref)
        # save generated memory reference
        ztrace.append(mem_ref)

    return ztrace

def write_trace_to_file(file_path, syn_trace):
    try:
        with open(file_path, "w") as f:
            # syn_trace
            s = str(syn_trace)
            f.write(s[1 : len(s) - 1] + "\n")
    except Exception:
        print("Wrong file or file path")


if __name__ == "__main__":
    import sys
    import os
    import operator
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Trace Generator")
    parser.add_argument("--cumm_file", type=str, default="./profile/sd_cumm")
    parser.add_argument("--gen_trace_file", type=str, default="./syn_traces/tbl1")

    parser.add_argument("--numpy_rand_seed", type=int, default=123)
    parser.add_argument("--print_precision", type=int, default=5)

    parser.add_argument("--table_size", type=int, default=1000000)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--pooling_factor", type=int, default=80)

    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    print("cumm filepath = ", args.cumm_file)
    list_sd, cumm_sd = read_dist_from_file(args.cumm_file)
    table_size = args.table_size
    gen_len = args.num_batches * args.mini_batch_size * args.pooling_factor

    ### generate corresponding synthetic ###
    print("Begin gen traces ....")
    start = time.time()
    synthetic_trace = trace_generate_lru(table_size, list_sd, cumm_sd, gen_len, False)
    end = time.time()
    time_cost = (end - start) / 60
    print("syn_trace time cost (min) = ", time_cost)
    print("max of syn_trace = ", max(synthetic_trace))

    write_trace_to_file(args.gen_trace_file, synthetic_trace)
