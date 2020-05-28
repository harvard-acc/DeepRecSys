from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import argparse
import numpy as np
import json

def debugPrint(args, system_tag, message):
  if args.debug_mode:
    print("[" + str(system_tag) + "] " + str(message))
    sys.stdout.flush()
  return


def cli():
    ### parse arguments ###
    parser = argparse.ArgumentParser(description="DeepRecBench")

    # =========================================================================================
    # Model related configurations
    # =========================================================================================
    parser.add_argument("--arch_sparse_feature_size", type=int, default=2)
    parser.add_argument("--arch_embedding_size", type=str, default="4-3-2")
    parser.add_argument("--arch_mlp_bot", type=str, default="4-3-2")
    parser.add_argument("--arch_mlp_top", type=str, default="4-2-1")
    parser.add_argument("--arch_mlp_tasks", type=str, default="4-2-1")
    parser.add_argument("--num_multi_tasks", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--arch_interaction_op", type=str, default="dot")
    parser.add_argument("--arch_interaction_itself", action="store_true", default=False)
    parser.add_argument("--inter_op_workers", type=int, default=1)
    parser.add_argument('--sls_workers', type=int, default=1) # maximum intra-operator parallelism for sls
    parser.add_argument('--fc_workers', type=int, default=1) # maximum intra-operator parallelism for fc
    parser.add_argument("--model_type", type=str, default="dlrm", help="Type of model to run (dlrm | wnd | ncf | din | dien)")
    parser.add_argument("--user_behavior_tables", type=int, default=1000)

    # =========================================================================================
    # Inference related configurations
    # =========================================================================================
    parser.add_argument("--inference_only", action="store_true", default=True)
    parser.add_argument("--save_proto_types_shapes", action="store_true", default=False)
    parser.add_argument("--output_log_file", type=str, default=None)

    # =========================================================================================
    # Dataset related configurations
    # =========================================================================================
    parser.add_argument("--num_batches", type=int, default=0)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--max_mini_batch_size", type=int, default=1)
    parser.add_argument("--avg_mini_batch_size", type=float, default=1)
    parser.add_argument("--var_mini_batch_size", type=float, default=1)
    parser.add_argument("--batch_size_distribution", type=str, default="fixed") # synthetic or dataset
    parser.add_argument("--batch_dist_file", type=str, default="config/batch_distribution.txt") # synthetic or dataset
    parser.add_argument("--sub_task_batch_size", type=int, default=16)

    parser.add_argument("--data_generation", type=str, default="random") # synthetic or dataset
    parser.add_argument("--data_trace_file", type=str,default="./input/dist_emb_j.log")
    parser.add_argument("--data_set", type=str, default="kaggle") # or terabyte
    parser.add_argument("--raw_data_file", type=str, default="")
    parser.add_argument("--processed_data_file", type=str, default="")
    parser.add_argument("--data_randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data_trace_enable_padding", type=bool, default=False)
    parser.add_argument("--num_indices_per_lookup", type=int, default=10)
    parser.add_argument("--num_indices_per_lookup_fixed", type=bool, default=False)

    # =========================================================================================
    # DeepRecSys related configurations
    # =========================================================================================
    parser.add_argument("--queue", action="store_true", default=False)
    parser.add_argument("--inference_engines", type=int, default=1)
    parser.add_argument("--avg_arrival_rate", type=float, default=10)
    parser.add_argument("--target_latency", type=float, default=10)
    parser.add_argument("--req_granularity", type=int, default=64)

    parser.add_argument("--batch_configs", type=str, default="32-64-128-256-512-1024")
    parser.add_argument("--tune_batch_qps", action="store_true", default=False)

    parser.add_argument("--tune_accel_qps", action="store_true", default=False)
    parser.add_argument("--accel_configs", type=str, default="128-256-512")

    parser.add_argument("--stable_region", type=float, default=0.10)
    parser.add_argument("--max_arr_range", type=float, default=100)
    parser.add_argument("--min_arr_range", type=float, default=1)
    parser.add_argument("--arr_steps", type=int, default=20)
    parser.add_argument("--sched_timeout", type=int, default=100)

    # =========================================================================================
    # Hardware related configurations
    # =========================================================================================
    parser.add_argument("--use_accel", action="store_true", default=False)
    parser.add_argument("--model_accel", action="store_true", default=False)
    parser.add_argument("--accel_request_size_thres", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--accel_root_dir", type=str, default="accelerator/")

    # =========================================================================================
    # Activations and loss function related configurations
    # =========================================================================================
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--loss_function", type=str, default="mse")  # or bce
    parser.add_argument("--loss_threshold", type=float, default=0.0) # 1.0e-7
    parser.add_argument("--round_targets", type=bool, default=False)

    # =========================================================================================
    # Training related configurations
    # =========================================================================================
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--print_precision", type=int, default=5)
    parser.add_argument("--numpy_rand_seed", type=int, default=123)
    parser.add_argument("--sync_dense_params", type=bool, default=True)
    parser.add_argument("--caffe2_net_type", type=str, default="simple")
    parser.add_argument('--engine', type=str, default='TBB') # can also be async_scheduling

    # =========================================================================================
    # Debugging related configurations
    # =========================================================================================
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--print_time", action="store_true", default=False)
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--enable_profiling", action="store_true", default=False)
    parser.add_argument("--plot_compute_graph", action="store_true", default=False)

    parser.add_argument("--log_file", type=str, default="log/output.log")
    # =========================================================================================
    # Experimental configuration
    # =========================================================================================
    parser.add_argument("--config_file", type=str, default=None)

    args = parser.parse_args()

    if args.model_type == "din":
      table_sizes = list(np.fromstring(args.arch_embedding_size, dtype=int, sep="-"))
      user_profile_table = table_sizes[0]
      user_behavior_table = table_sizes[1]
      table_sizes = table_sizes[1:]
      for i in range(args.user_behavior_tables):
        table_sizes = [user_behavior_table] + table_sizes

      arch_emb_str = ""
      table_sizes = [user_profile_table] + table_sizes

      for i, table in enumerate(table_sizes):
        if i == len(table_sizes)-1:
          arch_emb_str += str(table)
        else:
          arch_emb_str += str(table) + "-"

      args.arch_embedding_size = arch_emb_str

    # Use configuration file as master configuration of experiment. That is,
    # configuration file gets master control over command line arguments in
    # order to set experiment parameters
    if args.config_file:
      with open(args.config_file, "r") as f:
        config = json.load(f)

      for key in dict(config).keys():
        type_of = type(getattr(args, key))
        setattr(args, key, type_of(config[key]))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)

    return args

