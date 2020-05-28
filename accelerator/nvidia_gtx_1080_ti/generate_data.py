from __future__ import print_function
import os, sys

def sweep_cmd(config_file, model_file, output_file):

	num_epochs  = 100
	num_batches = 4

	rt_config_gpu = "--inference_only --inter_op_workers 1 --caffe2_net_type async_dag --use_accel "
	model_config  = "--config_file ../../models/configs/" + config_file + " "

	with open(output_file, 'w') as outfile:
		sys.stdout = outfile

		# Sweep Batch Size from {2^0 = 1, ... ,2^14 = 16384}
		for x in range(6):
			n = 4**x

			data_config = "--nepochs " + str(num_epochs) + " --num_batches " + str(num_batches) + " --mini_batch_size " + str(n) + " --max_mini_batch_size " + str(n)
			gpu_command = "python ../../models/" + model_file + " " + rt_config_gpu + model_config + data_config

			print("--------------------Running ("+model_file+") GPU Test with Batch Size " + str(n) +"--------------------\n")
			outfile.write(os.popen(gpu_command).read()+"\n")

		sys.stdout = sys.__stdout__



if __name__ == "__main__":
	if not os.path.exists("raw_data"):
		os.mkdir("raw_data")

	sweep_cmd("wide_and_deep.json", "wide_and_deep.py", "raw_data/results_wnd.txt")
	sweep_cmd("dlrm_rm1.json", "dlrm_s_caffe2.py", "raw_data/results_rm1.txt")
	sweep_cmd("dlrm_rm2.json", "dlrm_s_caffe2.py", "raw_data/results_rm2.txt")
	sweep_cmd("dlrm_rm3.json", "dlrm_s_caffe2.py", "raw_data/results_rm3.txt")
	sweep_cmd("ncf.json", "ncf.py", "raw_data/results_ncf.txt")
	sweep_cmd("mtwnd.json", "multi_task_wnd.py", "raw_data/results_mtwnd.txt")
	sweep_cmd("din.json", "din.py", "raw_data/results_din.txt")
	sweep_cmd("dien.json", "dien.py", "raw_data/results_dien.txt")