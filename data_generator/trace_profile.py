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


# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1

# read original traces to profile
def read_trace_from_file(file_path):
    print("begin load trace: ", file_path)
    raw_1D_indices_trace = pd.read_csv(file_path, sep=' ', header=None)
    raw_1D_indices_trace = np.array(raw_1D_indices_trace).flatten()
    return raw_1D_indices_trace

# profile the trace to get the stack distribution
def trace_profile(trace, max_stack_distance):
    trace = np.asarray(trace)

    stack_distances = []  # SDS
    line_accesses = []  # unique indices

    for i in range(len(trace)):
        x = trace[i]

        if i < max_stack_distance:
            trace_tmp = trace[0:i]
        else:
            trace_tmp = trace[i-max_stack_distance:i]

        x_index = np.where(trace_tmp==x)

        if len(x_index[0] > 0):
            last_idx = x_index[0][-1]
            sub_region = trace_tmp[last_idx:i]
            stack_dist_tmp = len(set(sub_region))
            stack_distances.append(stack_dist_tmp)
        else:
            stack_distances.append(0)
            line_accesses.append(x)

    return (stack_distances, line_accesses)


def write_dist_to_file(file_path, list_sd, cumm_sd):
    try:
        with open(file_path, "w") as f:
            # list_sd
            s = str(list_sd)
            f.write(s[1 : len(s) - 1] + "\n")
            # cumm_sd
            s = str(cumm_sd)
            f.write(s[1 : len(s) - 1] + "\n")
    except Exception:
        print("Wrong file or file path")


if __name__ == "__main__":
    import sys
    import os
    import operator
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Profile traces stack distance distribution")
    parser.add_argument("--profile_trace_file", type=str, default="./original_traces/tbl1")
    parser.add_argument("--profile_trace_random", type=int, default=1)
    parser.add_argument("--profile_len", type=int, default=1000000)
    parser.add_argument("--max_stack_distance", type=int, default=10000)

    parser.add_argument("--sd_file_prob", type=str, default="./profile/sd_prob")
    parser.add_argument("--sd_file_cumm", type=str, default="./profile/sd_cumm")

    parser.add_argument("--numpy_rand_seed", type=int, default=123)
    parser.add_argument("--print_precision", type=int, default=5)

    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    print("args.profile_trace_random = ", args.profile_trace_random)

    if args.profile_trace_random == 0:
        print("================ profile dataset trace =========================")
        raw_trace = read_trace_from_file(args.profile_trace_file)
        profile_len = min(args.profile_len, len(raw_data))
        trace = raw_trace[0:profile_len]
        print("len of raw_trace, trace = ", len(raw_trace), len(trace))
    else:
        print("================ profile dataset trace =========================")
        num_indices = args.profile_len
        table_size = 10000000
        raw_trace = np.random.randint(0, table_size, num_indices, np.int32)
        trace = raw_trace

    ### profile trace ###
    print("================== begin profile trace =========================")
    (stack_distances, line_accesses) = trace_profile(trace, args.max_stack_distance)
    print("len of stack_distances = ", len(stack_distances))
    print("len of line accesses = ", len(line_accesses))

    ### compute probability distribution ###
    # count items
    l = len(stack_distances)
    dc = sorted(collections.Counter(stack_distances).items(), key=operator.itemgetter(0))
    print("len of stack_distances and dc = ", l, len(dc))

    # create a distribution
    list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
    dist_sd = list(map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc))  # k = tuple_x_k[1]
    prob_sd = []
    cumm_sd = []  # np.cumsum(dc).tolist() #prefixsum
    for i, (_, k) in enumerate(dc):
        if i == 0:
            prob_sd.append(k / float(l))
            cumm_sd.append(k / float(l))
        else:
            # add the 2nd element of the i-th tuple in the dist_sd list
            prob_sd.append(k / float(l))
            cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

    ### write stack_distance and line_accesses to a file ###
    write_dist_to_file(args.sd_file_prob, list_sd, prob_sd)
    write_dist_to_file(args.sd_file_cumm, list_sd, cumm_sd)
