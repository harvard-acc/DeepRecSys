#/bin/bash

###########################################################################
# CPU experiments
###########################################################################

# DLRM RMC1 Network
python dlrm_s_caffe2.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dlrm_rm1.json" --enable_profiling --engine "prof_dag" --nepochs 32

# DLRM RMC2 Network
python dlrm_s_caffe2.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dlrm_rm2.json" --enable_profiling --engine "prof_dag" --nepochs 32

# DLRM RMC3 Network
python dlrm_s_caffe2.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dlrm_rm3.json" --enable_profiling --engine "prof_dag" --nepochs 32

# Wide and Deep Network
python wide_and_deep.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/wide_and_deep.json" --enable_profiling --engine "prof_dag" --nepochs 32

# Multi Task Wide and Deep Network
python multi_task_wnd.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/mtwnd.json" --enable_profiling --engine "prof_dag" --nepochs 32

# Neural Collaborative Filtering Network
python wide_and_deep.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/ncf.json" --enable_profiling --engine "prof_dag" --nepochs 32

# Deep Interest Network
python din.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/din.json" --enable_profiling --engine "prof_dag" --nepochs 32

# Deep Interest Evolution Network
python din.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dien.json" --enable_profiling --engine "prof_dag" --nepochs 32

###########################################################################
# GPU experiments
###########################################################################

# DLRM RMC1 Network
python dlrm_s_caffe2.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dlrm_rm1.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

# DLRM RMC2 Network
python dlrm_s_caffe2.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dlrm_rm2.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

# DLRM RMC3 Network
python dlrm_s_caffe2.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dlrm_rm3.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

# Wide and Deep Network
python wide_and_deep.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/wide_and_deep.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

# Multi Task Wide and Deep Network
python multi_task_wnd.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/mtwnd.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

# Neural Collaborative Filtering Network
python wide_and_deep.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/ncf.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

# Deep Interest Network
python din.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/din.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

# Deep Interest Evolution Network
python din.py --inference_only --num_batches 512  --caffe2_net_type async_dag --config_file "configs/dien.json" --enable_profiling --engine "prof_dag" --nepochs 32 --use_accel

