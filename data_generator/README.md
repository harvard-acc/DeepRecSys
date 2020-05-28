# Data generation

DeepRecSys can be instrumented with random, synthetic, or training-set based data.
This capability is based on Facebook's open-sourced Deep Learning Recommendation Model (DLRM) data generation.
For details on the data generation and synthetic trace generator, please follow the original DLRM data generation codebase ([link](https://github.com/facebookresearch/dlrm)).

## Synthetic trace generator

step-1: trace profile

step-2: trace generator

### 1. trace profile: trace_profile.py
input:<br/>
--profile_trace_file: set the path of the original path to profile<br/>
--profile_trace_random: 1 to profile random generated trace, 0 to profile traces in profile_trace_file<br/>
--profile_len: length of trace to profile<br/>
--max_stack_distance: maximum stack distance (related with cache size)<br/>

output:<br/>
--sd_file_prob: pdf filepath<br/>
--sd_file_cumm: cdf filepath<br/>



### 2. trace generator: trace_generator.py
input:<br/>
--sd_file_cumm: cdf filepath<br/>
--table_size: set emb table size for the trace<br/>
--num_batches, mini_batch_size, pooling_factor: the length of the generated trace<br/>

output:<br/>
--gen_trace_file: syn_trace filepath<br/>
