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

# others
import bisect
import collections

import numpy as np

from data_generator.data_generator import DataGenerator

# numpy
from numpy import random as ra

class DLRMDataGenerator(DataGenerator):
  def __init__(self, args):
    super(DLRMDataGenerator,self).__init__(args)
    return

  def generate_input_data(self, ):
    ln_bot = np.fromstring(self.args.arch_mlp_bot, dtype=int, sep="-")
    if self.args.data_generation == "dataset":
        sys.exit("ERROR: Dataset based DLRM instrumentation is currently not supported")

    elif self.args.data_generation == "random":
      ln_emb = np.fromstring(self.args.arch_embedding_size, dtype=int, sep="-")
      m_den = ln_bot[0]
      (nbatches, lX, lS_l, lS_i) = self.generate_random_input_data(
          self.args.num_batches, self.args.max_mini_batch_size,
          self.args.round_targets, self.args.num_indices_per_lookup,
          self.args.num_indices_per_lookup_fixed, m_den, ln_emb)
      return (nbatches, lX, lS_l, lS_i)

    elif self.args.data_generation == "synthetic":
      ln_emb = np.fromstring(self.args.arch_embedding_size, dtype=int, sep="-")
      m_den = ln_bot[0]
      (nbatches, lX, lS_l, lS_i) = generate_synthetic_input_data(
          self.args.num_batches, self.args.max_mini_batch_size,
          self.args.round_targets, self.args.num_indices_per_lookup,
          self.args.num_indices_per_lookup_fixed, m_den, ln_emb,
          self.args.data_trace_file, self.args.data_trace_enable_padding)
      return (nbatches, lX, lS_l, lS_i)

    else:
        sys.exit("ERROR: --data-generation="
                 + self.args.data_generation + " is not supported")

  def generate_output_data(self, ):
    (nbatches, lT) = self.generate_random_output_data(
        self.args.num_batches, self.args.max_mini_batch_size,
        round_targets=self.args.round_targets)
    return (nbatches, lT)

  # uniform ditribution (input data)
  def generate_random_input_data(self,
      num_batches,
      mini_batch_size,
      round_targets,
      num_indices_per_lookup,
      num_indices_per_lookup_fixed,
      m_den,
      ln_emb,
  ):
      nbatches = num_batches

      # inputs and targets
      lX = []
      lS_lengths = []
      lS_indices = []
      for j in range(0, nbatches):
          # number of data points in a batch
          n = mini_batch_size

          # dense feature
          Xt = ra.rand(n, m_den).astype(np.float32)
          lX.append(Xt)

          # sparse feature (sparse indices)
          lS_emb_lengths = []
          lS_emb_indices = []
          # for each embedding generate a list of n lookups,
          # where each lookup is composed of multiple sparse indices
          for size in ln_emb:
              lS_batch_lengths = []
              lS_batch_indices = []
              for _ in range(n):
                  # num of sparse indices to be used per embedding (between)
                  sparse_group_size = np.int32(num_indices_per_lookup)

                  # sparse indices to be used per embedding
                  r = ra.random(sparse_group_size)
                  sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int32))

                  while sparse_group.size != sparse_group_size:
                    r = ra.random(sparse_group_size)
                    sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int32))

                  # reset sparse_group_size in case some index duplicates were removed
                  sparse_group_size = np.int32(sparse_group.size)
                  # store lengths and indices
                  lS_batch_lengths += [sparse_group_size]
                  lS_batch_indices += sparse_group.tolist()

              lS_emb_lengths.append(lS_batch_lengths)
              lS_emb_indices.append(lS_batch_indices)

          lS_lengths.append(lS_emb_lengths)
          lS_indices.append(lS_emb_indices)

      return (nbatches, lX, lS_lengths, lS_indices)


# uniform distribution (output data)
  def generate_random_output_data(
      self,
      num_batches,
      mini_batch_size,
      num_targets=1,
      round_targets=False):

      nbatches = num_batches

      lT = []
      for j in range(0, nbatches):
          # number of data points in a batch
          n = mini_batch_size
          # target (probability of a click)
          if round_targets:
              P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.int32)
          else:
              P = ra.rand(n, num_targets).astype(np.float32)
          lT.append(P)

      return (nbatches, lT)


# synthetic distribution (input data)
  def generate_synthetic_input_data(
      num_batches,
      mini_batch_size,
      round_targets,
      num_indices_per_lookup,
      num_indices_per_lookup_fixed,
      m_den,
      ln_emb,
      trace_file,
      enable_padding=False,
  ):
      nbatches = num_batches
      # print("Total number of batches %d" % nbatches)

      # inputs and targets
      lX = []
      lS_lengths = []
      lS_indices = []
      for j in range(0, nbatches):
          # number of data points in a batch
          n = mini_batch_size
          # dense feature
          Xt = ra.rand(n, m_den).astype(np.float32)
          lX.append(Xt)
          # sparse feature (sparse indices)
          lS_emb_lengths = []
          lS_emb_indices = []
          # for each embedding generate a list of n lookups,
          # where each lookup is composed of multiple sparse indices
          for i, size in enumerate(ln_emb):
              lS_batch_lengths = []
              lS_batch_indices = []
              for _ in range(n):
                  # num of sparse indices to be used per embedding (between
                  if num_indices_per_lookup_fixed:
                      sparse_group_size = np.int32(num_indices_per_lookup)
                  else:
                      # random between [1,num_indices_per_lookup])
                      r = ra.random(1)
                      sparse_group_size = np.int32(
                          max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                      )
                  # sparse indices to be used per embedding
                  file_path = trace_file
                  line_accesses, list_sd, cumm_sd = read_dist_from_file(
                      file_path.replace("j", str(i))
                  )

                  r = trace_generate_lru(
                      line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
                  )

                  # WARNING: if the distribution in the file is not consistent with
                  # embedding table dimensions, below mod guards against out of
                  # range access
                  sparse_group = np.unique(r).astype(np.int32)
                  minsg = np.min(sparse_group)
                  maxsg = np.max(sparse_group)
                  if (minsg < 0) or (size <= maxsg):
                      print(
                          "WARNING: distribution is inconsistent with embedding "
                          + "table size (using mod to recover and continue)"
                      )
                      sparse_group = np.mod(sparse_group, size).astype(np.int32)
                  # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int32))
                  # reset sparse_group_size in case some index duplicates were removed
                  sparse_group_size = np.int32(sparse_group.size)
                  # store lengths and indices
                  lS_batch_lengths += [sparse_group_size]
                  lS_batch_indices += sparse_group.tolist()
              lS_emb_lengths.append(lS_batch_lengths)
              lS_emb_indices.append(lS_batch_indices)
          lS_lengths.append(lS_emb_lengths)
          lS_indices.append(lS_emb_indices)

      return (nbatches, lX, lS_lengths, lS_indices)


  def generate_stack_distance(cumm_val, cumm_dist, max_i, i, enable_padding=False):
      u = ra.rand(1)
      if i < max_i:
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


# WARNING: global define, must be consistent across all synthetic functions
  cache_line_size = 1


  def trace_generate_lru(
      line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
  ):
      max_sd = list_sd[-1]
      l = len(line_accesses)
      i = 0
      ztrace = []
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



  def trace_profile(trace, enable_padding=False):
      # number of elements in the array (assuming 1D)
      # n = trace.size

      rstack = []  # S
      stack_distances = []  # SDS
      line_accesses = []  # L
      for x in trace:
          r = np.uint64(x / cache_line_size)
          l = len(rstack)
          try:  # found #
              i = rstack.index(r)
              # WARNING: I believe below is the correct depth in terms of meaning of the
              #          algorithm, but that is not what seems to be in the paper alg.
              #          -1 can be subtracted if we defined the distance between
              #          consecutive accesses (e.g. r, r) as 0 rather than 1.
              sd = l - i  # - 1
              # push r to the end of stack_distances
              stack_distances.insert(0, sd)
              # remove r from its position and insert to the top of stack
              rstack.pop(i)  # rstack.remove(r)
              rstack.insert(l - 1, r)
          except ValueError:  # not found #
              sd = 0  # -1
              # push r to the end of stack_distances/line_accesses
              stack_distances.insert(0, sd)
              line_accesses.insert(0, r)
              # push r to the top of stack
              rstack.insert(l, r)

      if enable_padding:
          # WARNING: notice that as the ratio between the number of samples (l)
          # and cardinality (c) of a sample increases the probability of
          # generating a sample gets smaller and smaller because there are
          # few new samples compared to repeated samples. This means that for a
          # long trace with relatively small cardinality it will take longer to
          # generate all new samples and therefore obtain full distribution support
          # and hence it takes longer for distribution to resemble the original.
          # Therefore, we may pad the number of new samples to be on par with
          # average number of samples l/c artificially.
          l = len(stack_distances)
          c = max(stack_distances)
          padding = int(np.ceil(l / c))
          stack_distances = stack_distances + [0] * padding

      return (rstack, stack_distances, line_accesses)


# auxiliary read/write routines
  def read_trace_from_file(file_path):
      try:
          with open(file_path) as f:
              if args.trace_file_binary_type:
                  array = np.fromfile(f, dtype=np.uint64)
                  trace = array.astype(np.uint64).tolist()
              else:
                  line = f.readline()
                  trace = list(map(lambda x: np.uint64(x), line.split(", ")))
              return trace
      except Exception:
          print("ERROR: no input trace file has been provided")


  def write_trace_to_file(file_path, trace):
      try:
          if args.trace_file_binary_type:
              with open(file_path, "wb+") as f:
                  np.array(trace).astype(np.uint64).tofile(f)
          else:
              with open(file_path, "w+") as f:
                  s = str(trace)
                  f.write(s[1 : len(s) - 1])
      except Exception:
          print("ERROR: no output trace file has been provided")


  def read_dist_from_file(file_path):
      try:
          with open(file_path, "r") as f:
              lines = f.read().splitlines()
      except Exception:
          print("Wrong file or file path")
      # read unique accesses
      unique_accesses = [int(el) for el in lines[0].split(", ")]
      # read cumulative distribution (elements are passed as two separate lists)
      list_sd = [int(el) for el in lines[1].split(", ")]
      cumm_sd = [float(el) for el in lines[2].split(", ")]

      return unique_accesses, list_sd, cumm_sd


  def write_dist_to_file(file_path, unique_accesses, list_sd, cumm_sd):
      try:
          with open(file_path, "w") as f:
              # unique_acesses
              s = str(unique_accesses)
              f.write(s[1 : len(s) - 1] + "\n")
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
    parser = argparse.ArgumentParser(description="Generate Synthetic Distributions")
    parser.add_argument("--trace-file", type=str, default="./input/trace.log")
    parser.add_argument("--trace-file-binary-type", type=bool, default=False)
    parser.add_argument("--trace-enable-padding", type=bool, default=False)
    parser.add_argument("--dist-file", type=str, default="./input/dist.log")
    parser.add_argument(
        "--synthetic-file", type=str, default="./input/trace_synthetic.log"
    )
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--print-precision", type=int, default=5)
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    ### read trace ###
    trace = read_trace_from_file(args.trace_file)
    # print(trace)

    ### profile trace ###
    (_, stack_distances, line_accesses) = trace_profile(
        trace, args.trace_enable_padding
    )
    stack_distances.reverse()
    line_accesses.reverse()
    # print(line_accesses)
    # print(stack_distances)

    ### compute probability distribution ###
    # count items
    l = len(stack_distances)
    dc = sorted(
        collections.Counter(stack_distances).items(), key=operator.itemgetter(0)
    )

    # create a distribution
    list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
    dist_sd = list(
        map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc)
    )  # k = tuple_x_k[1]
    cumm_sd = []  # np.cumsum(dc).tolist() #prefixsum
    for i, (_, k) in enumerate(dc):
        if i == 0:
            cumm_sd.append(k / float(l))
        else:
            # add the 2nd element of the i-th tuple in the dist_sd list
            cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

    ### write stack_distance and line_accesses to a file ###
    write_dist_to_file(args.dist_file, line_accesses, list_sd, cumm_sd)

    ### generate correspondinf synthetic ###
    # line_accesses, list_sd, cumm_sd = read_dist_from_file(args.dist_file)
    synthetic_trace = trace_generate_lru(
        line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    )
    write_trace_to_file(args.synthetic_file, synthetic_trace)
