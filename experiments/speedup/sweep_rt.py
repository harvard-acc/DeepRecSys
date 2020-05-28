from __future__ import print_function
import os, sys
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rgb(r,g,b):
	return (float(r)/256.,float(g)/256.,float(b)/256.)

# Stores all CPU characterization results
class CPU_Data:
	
	def parse(self, filename):
		i = 0
		file           = "raw_data/"+filename # Path to file
		results        = []                   # Raw results
		results_tuples = []                   # Processed results
		if not os.path.isfile(file):
			print("Error: "+file+" does not exist.")
			sys.exit()
		with open(file) as f:
			for line in f:
				if "***" in line:
					result = line[line.rindex('*') + 1 : line.rindex('ms')]
					results.append(float(result)) # Add each raw result
		print("CPU_Data:parse: "+str(len(results))+" data points parsed from "+file) # Print number of results
		while (i < len(results)):
			tup = []
			for x in range (12):
				tup.append(results[i+x])
			i = i + 12
			results_tuples.append(tup)
		results_load = np.asarray(results_tuples)[:,1]
		results_comp = np.asarray(results_tuples)[:,3]
		results_exec = np.asarray(results_tuples)[:,5]
		return results_load, results_comp, results_exec
	
	def gen_out(self, config, profile_type, directory):
		data = eval("self."+config+"_"+profile_type)
		file = directory+"/"+config+"_"+profile_type+".txt"
		with open(file, 'w') as f:
			for n in range(data.size):
				batch_size = 4**n
				output = str(batch_size)+" , "+str(data[n])
				print(output, file=f)
	
	def gen_out_all(self, hw_type, model_type):
		config    = hw_type+"_"+model_type
		directory = "characterization/cpu_data"
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.gen_out(config, "load", directory)
		self.gen_out(config, "comp", directory)
		self.gen_out(config, "exec", directory)
		
	def __init__(self):
		# Skylake Results
		self.sl_wnd_load, self.sl_wnd_comp, self.sl_wnd_exec       = self.parse("results_wnd.txt")
		self.sl_rm1_load, self.sl_rm1_comp, self.sl_rm1_exec       = self.parse("results_rm1.txt")
		self.sl_rm2_load, self.sl_rm2_comp, self.sl_rm2_exec       = self.parse("results_rm2.txt")
		self.sl_rm3_load, self.sl_rm3_comp, self.sl_rm3_exec       = self.parse("results_rm3.txt")
		self.sl_ncf_load, self.sl_ncf_comp, self.sl_ncf_exec       = self.parse("results_ncf.txt")
		self.sl_mtwnd_load, self.sl_mtwnd_comp, self.sl_mtwnd_exec = self.parse("results_mtwnd.txt")
		self.sl_din_load, self.sl_din_comp, self.sl_din_exec       = self.parse("results_din.txt")
		self.sl_dien_load, self.sl_dien_comp, self.sl_dien_exec    = self.parse("results_dien.txt")
		
		self.gen_out_all("sl", "wnd")
		self.gen_out_all("sl", "rm1")
		self.gen_out_all("sl", "rm2")
		self.gen_out_all("sl", "rm3")
		self.gen_out_all("sl", "ncf")
		self.gen_out_all("sl", "mtwnd")
		self.gen_out_all("sl", "din")
		self.gen_out_all("sl", "dien")
		

# Stores all GPU characterization results
class GPU_Data:
	
	def parse(self, filename):
		i = 0
		file           = "raw_data/"+filename # Path to file
		results        = []                   # Raw results
		results_tuples = []                   # Processed results
		if not os.path.isfile(file):
			print("Error: "+file+" does not exist.")
			sys.exit()
		with open(file) as f:
			for line in f:
				if "***" in line:
					result = line[line.rindex('*') + 1 : line.rindex('ms')]
					results.append(float(result)) # Add each raw result
		print("GPU_Data:parse: "+str(len(results))+" data points parsed from "+file) # Print number of results
		while (i < len(results)):
			tup = []
			for x in range (12):
				tup.append(results[i+x])
			i = i + 12
			results_tuples.append(tup)
		results_load = np.asarray(results_tuples)[:,7]
		results_comp = np.asarray(results_tuples)[:,9]
		results_exec = np.asarray(results_tuples)[:,11]
		return results_load, results_comp, results_exec
	
	def gen_out(self, config, profile_type, directory):
		data = eval("self."+config+"_"+profile_type)
		file = directory+"/"+config+"_"+profile_type+".txt"
		with open(file, 'w') as f:
			for n in range(data.size):
				batch_size = 4**n
				output = str(batch_size)+" , "+str(data[n])
				print(output, file=f)
	
	def gen_out_all(self, hw_type, model_type):
		config    = hw_type+"_"+model_type
		directory = "characterization/gpu_data"
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.gen_out(config, "load", directory)
		self.gen_out(config, "comp", directory)
		self.gen_out(config, "exec", directory)
		
	def __init__(self):
		# 1080 Ti Results
		self.nv1080ti_wnd_load, self.nv1080ti_wnd_comp, self.nv1080ti_wnd_exec       = self.parse("results_wnd.txt")
		self.nv1080ti_rm1_load, self.nv1080ti_rm1_comp, self.nv1080ti_rm1_exec       = self.parse("results_rm1.txt")
		self.nv1080ti_rm2_load, self.nv1080ti_rm2_comp, self.nv1080ti_rm2_exec       = self.parse("results_rm2.txt")
		self.nv1080ti_rm3_load, self.nv1080ti_rm3_comp, self.nv1080ti_rm3_exec       = self.parse("results_rm3.txt")
		self.nv1080ti_ncf_load, self.nv1080ti_ncf_comp, self.nv1080ti_ncf_exec       = self.parse("results_ncf.txt")
		self.nv1080ti_mtwnd_load, self.nv1080ti_mtwnd_comp, self.nv1080ti_mtwnd_exec = self.parse("results_mtwnd.txt")
		self.nv1080ti_din_load, self.nv1080ti_din_comp, self.nv1080ti_din_exec       = self.parse("results_din.txt")
		self.nv1080ti_dien_load, self.nv1080ti_dien_comp, self.nv1080ti_dien_exec    = self.parse("results_dien.txt")
		
		self.gen_out_all("nv1080ti", "wnd")
		self.gen_out_all("nv1080ti", "rm1")
		self.gen_out_all("nv1080ti", "rm2")
		self.gen_out_all("nv1080ti", "rm3")
		self.gen_out_all("nv1080ti", "ncf")
		self.gen_out_all("nv1080ti", "mtwnd")
		self.gen_out_all("nv1080ti", "din")
		self.gen_out_all("nv1080ti", "dien")
		

def parse(file):
	results        = []                   # Raw results
	if not os.path.isfile(file):
		print("Error: "+file+" does not exist.")
		sys.exit()
	with open(file) as f:
		for line in f:
			result = line[line.rindex(',') + 1 : ]
			results.append(float(result)) # Add each raw result
	print("parse: "+str(len(results))+" data points parsed from "+file) # Print number of results
	return np.asarray(results)

def sweep_cmd(config_file, model_file, output_file):

	num_epochs  = 100
	num_batches = 4

	rt_config_cpu = "--inference_only --inter_op_workers 1 --caffe2_net_type async_dag "
	rt_config_gpu = "--inference_only --inter_op_workers 1 --caffe2_net_type async_dag --use_accel "
	model_config  = "--config_file ../../models/configs/" + config_file + " "

	with open(output_file, 'w') as outfile:
		sys.stdout = outfile

		# Sweep Batch Size from {4^0 = 1, ... ,4^5 = 1024}
		for x in range(6):
			n = 4**x

			data_config = "--nepochs " + str(num_epochs) + " --num_batches " + str(num_batches) + " --mini_batch_size " + str(n) + " --max_mini_batch_size " + str(n)
			cpu_command = "python ../../models/" + model_file + " " + rt_config_cpu + model_config + data_config
			gpu_command = "python ../../models/" + model_file + " " + rt_config_gpu + model_config + data_config

			print("--------------------Running ("+model_file+") CPU Test with Batch Size " + str(n) +"--------------------\n")
			outfile.write(os.popen(cpu_command).read()+"\n")
			print("--------------------Running ("+model_file+") GPU Test with Batch Size " + str(n) +"--------------------\n")
			outfile.write(os.popen(gpu_command).read()+"\n")

		sys.stdout = sys.__stdout__



if __name__ == "__main__":

	# Creating Directories
	if not os.path.exists("raw_data"):
		os.mkdir("raw_data")
	if not os.path.exists("characterization"):
		os.mkdir("characterization")
	if not os.path.exists("pdf"):
		os.mkdir("pdf")
	if not os.path.exists("png"):
		os.mkdir("png")

	
	sweep_cmd("wide_and_deep.json", "wide_and_deep.py", "raw_data/results_wnd.txt")
	sweep_cmd("dlrm_rm1.json", "dlrm_s_caffe2.py", "raw_data/results_rm1.txt")
	sweep_cmd("dlrm_rm2.json", "dlrm_s_caffe2.py", "raw_data/results_rm2.txt")
	sweep_cmd("dlrm_rm3.json", "dlrm_s_caffe2.py", "raw_data/results_rm3.txt")
	sweep_cmd("ncf.json", "ncf.py", "raw_data/results_ncf.txt")
	sweep_cmd("mtwnd.json", "multi_task_wnd.py", "raw_data/results_mtwnd.txt")
	sweep_cmd("din.json", "din.py", "raw_data/results_din.txt")
	sweep_cmd("dien.json", "dien.py", "raw_data/results_dien.txt")
	
	
	cpu_data = CPU_Data()
	gpu_data = GPU_Data()

	crimson = rgb(172,63,64)
	blue    = rgb(62,145,189)
	teal    = rgb(98,189,153)
	orange  = rgb(250,174,83)
	black   = rgb(0,0,0)
	violet  = rgb(139,5,242)
	brown   = rgb(224,152,115)
	grey    = rgb(128, 128, 128)
	green   = rgb(53,137,88)
	pink    = rgb(255,182,193)
	dark_brown = rgb(101, 67, 33)

	#   luminance channel sweeps from dark to light, (for ordered comparisons)
	clr = [crimson, blue, black, teal, orange, violet, brown, grey, green, pink, dark_brown]

	models       = ["WND", "RM1", "RM2", "RM3", "NCF", "MTWND", "DIN", "DIEN"]
	cpu_data_dir = "characterization/cpu_data/"
	gpu_data_dir = "characterization/gpu_data/"
	batch_size   = 4**np.arange(6)

	# Skylake Results

	# WND
	sl_wnd_load = parse(cpu_data_dir+"sl_wnd_load.txt")
	sl_wnd_comp = parse(cpu_data_dir+"sl_wnd_comp.txt")
	sl_wnd_exec = parse(cpu_data_dir+"sl_wnd_exec.txt")
	# RM1
	sl_rm1_load = parse(cpu_data_dir+"sl_rm1_load.txt")
	sl_rm1_comp = parse(cpu_data_dir+"sl_rm1_comp.txt")
	sl_rm1_exec = parse(cpu_data_dir+"sl_rm1_exec.txt")
	# RM2
	sl_rm2_load = parse(cpu_data_dir+"sl_rm2_load.txt")
	sl_rm2_comp = parse(cpu_data_dir+"sl_rm2_comp.txt")
	sl_rm2_exec = parse(cpu_data_dir+"sl_rm2_exec.txt")
	# RM3
	sl_rm3_load = parse(cpu_data_dir+"sl_rm3_load.txt")
	sl_rm3_comp = parse(cpu_data_dir+"sl_rm3_comp.txt")
	sl_rm3_exec = parse(cpu_data_dir+"sl_rm3_exec.txt")
	# NCF
	sl_ncf_load = parse(cpu_data_dir+"sl_ncf_load.txt")
	sl_ncf_comp = parse(cpu_data_dir+"sl_ncf_comp.txt")
	sl_ncf_exec = parse(cpu_data_dir+"sl_ncf_exec.txt")
	# MTWND
	sl_mtwnd_load = parse(cpu_data_dir+"sl_mtwnd_load.txt")
	sl_mtwnd_comp = parse(cpu_data_dir+"sl_mtwnd_comp.txt")
	sl_mtwnd_exec = parse(cpu_data_dir+"sl_mtwnd_exec.txt")
	# DIN
	sl_din_load = parse(cpu_data_dir+"sl_din_load.txt")
	sl_din_comp = parse(cpu_data_dir+"sl_din_comp.txt")
	sl_din_exec = parse(cpu_data_dir+"sl_din_exec.txt")
	# DIEN
	sl_dien_load = parse(cpu_data_dir+"sl_dien_load.txt")
	sl_dien_comp = parse(cpu_data_dir+"sl_dien_comp.txt")
	sl_dien_exec = parse(cpu_data_dir+"sl_dien_exec.txt")

	# 1080 Ti Results

	# WND
	nv1080ti_wnd_load = parse(gpu_data_dir+"nv1080ti_wnd_load.txt")
	nv1080ti_wnd_comp = parse(gpu_data_dir+"nv1080ti_wnd_comp.txt")
	nv1080ti_wnd_exec = parse(gpu_data_dir+"nv1080ti_wnd_exec.txt")
	# RM1
	nv1080ti_rm1_load = parse(gpu_data_dir+"nv1080ti_rm1_load.txt")
	nv1080ti_rm1_comp = parse(gpu_data_dir+"nv1080ti_rm1_comp.txt")
	nv1080ti_rm1_exec = parse(gpu_data_dir+"nv1080ti_rm1_exec.txt")
	# RM2
	nv1080ti_rm2_load = parse(gpu_data_dir+"nv1080ti_rm2_load.txt")
	nv1080ti_rm2_comp = parse(gpu_data_dir+"nv1080ti_rm2_comp.txt")
	nv1080ti_rm2_exec = parse(gpu_data_dir+"nv1080ti_rm2_exec.txt")
	# RM3
	nv1080ti_rm3_load = parse(gpu_data_dir+"nv1080ti_rm3_load.txt")
	nv1080ti_rm3_comp = parse(gpu_data_dir+"nv1080ti_rm3_comp.txt")
	nv1080ti_rm3_exec = parse(gpu_data_dir+"nv1080ti_rm3_exec.txt")
	# NCF
	nv1080ti_ncf_load = parse(gpu_data_dir+"nv1080ti_ncf_load.txt")
	nv1080ti_ncf_comp = parse(gpu_data_dir+"nv1080ti_ncf_comp.txt")
	nv1080ti_ncf_exec = parse(gpu_data_dir+"nv1080ti_ncf_exec.txt")
	# MTWND
	nv1080ti_mtwnd_load = parse(gpu_data_dir+"nv1080ti_mtwnd_load.txt")
	nv1080ti_mtwnd_comp = parse(gpu_data_dir+"nv1080ti_mtwnd_comp.txt")
	nv1080ti_mtwnd_exec = parse(gpu_data_dir+"nv1080ti_mtwnd_exec.txt")
	# DIN
	nv1080ti_din_load = parse(gpu_data_dir+"nv1080ti_din_load.txt")
	nv1080ti_din_comp = parse(gpu_data_dir+"nv1080ti_din_comp.txt")
	nv1080ti_din_exec = parse(gpu_data_dir+"nv1080ti_din_exec.txt")
	# DIEN
	nv1080ti_dien_load = parse(gpu_data_dir+"nv1080ti_dien_load.txt")
	nv1080ti_dien_comp = parse(gpu_data_dir+"nv1080ti_dien_comp.txt")
	nv1080ti_dien_exec = parse(gpu_data_dir+"nv1080ti_dien_exec.txt")

	plt.figure(figsize = (7.5,5), dpi = 250)

	nv1080ti_wnd_speedup = sl_wnd_exec/nv1080ti_wnd_exec
	nv1080ti_rm1_speedup = sl_rm1_exec/nv1080ti_rm1_exec
	nv1080ti_rm2_speedup = sl_rm2_exec/nv1080ti_rm2_exec
	nv1080ti_rm3_speedup = sl_rm3_exec/nv1080ti_rm3_exec
	nv1080ti_ncf_speedup = sl_ncf_exec/nv1080ti_ncf_exec
	nv1080ti_mtwnd_speedup = sl_mtwnd_exec/nv1080ti_mtwnd_exec
	nv1080ti_din_speedup = sl_din_exec/nv1080ti_din_exec
	nv1080ti_dien_speedup = sl_dien_exec/nv1080ti_dien_exec

	plt.title("Nvidia GTX 1080 Ti Model Execution Time Speedup")
	plt.xlabel("Batch Size")
	plt.ylabel("Speedup (Relative to Skylake)")

	plt.xscale('log')
	plt.xlim([batch_size[0],batch_size[-1]])

	plt.axhline(1, color = 'black', linestyle = '--')
	plt.scatter(batch_size, nv1080ti_wnd_speedup, color = clr[0])
	plt.plot(batch_size, nv1080ti_wnd_speedup, color = clr[0], label = 'WnD')
	plt.scatter(batch_size, nv1080ti_rm1_speedup, color = clr[1])
	plt.plot(batch_size, nv1080ti_rm1_speedup, color = clr[1], label = 'RM1')
	plt.scatter(batch_size, nv1080ti_rm2_speedup, color = clr[2])
	plt.plot(batch_size, nv1080ti_rm2_speedup, color = clr[2], label = 'RM2')
	plt.scatter(batch_size, nv1080ti_rm3_speedup, color = clr[3])
	plt.plot(batch_size, nv1080ti_rm3_speedup, color = clr[3], label = 'RM3')
	plt.scatter(batch_size, nv1080ti_ncf_speedup, color = clr[4])
	plt.plot(batch_size, nv1080ti_ncf_speedup, color = clr[4], label = 'NCF')
	plt.scatter(batch_size, nv1080ti_mtwnd_speedup, color = clr[5])
	plt.plot(batch_size, nv1080ti_mtwnd_speedup, color = clr[5], label = 'MTWND')
	plt.scatter(batch_size, nv1080ti_din_speedup, color = clr[6])
	plt.plot(batch_size, nv1080ti_din_speedup, color = clr[6], label = 'DIN')
	plt.scatter(batch_size, nv1080ti_dien_speedup, color = clr[7])
	plt.plot(batch_size, nv1080ti_dien_speedup, color = clr[7], label = 'DIEN')

	plt.legend(loc = 'upper left', frameon = False)

	plt.savefig('pdf/model_speedup.pdf', bbox_inches='tight')
	plt.savefig('png/model_speedup.png', bbox_inches='tight')