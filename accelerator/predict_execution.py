from __future__ import print_function

import os, sys, math
import numpy as np

# Stores all the relevant GPU Characterization Statistics for Use
class GPU_Data:

    # Parses a GPU-Only results file.
    def parse_gpu(self, filename):
        results        = [] # Raw results
        results_tuples = [] # Organize results into tuples
        i = 0
        if not os.path.isfile(filename):
            print("File " + filename + " does not exist")
            sys.exit()
        with open(filename) as file:
            for line in file:
                if "***" in line:
                    result = line[line.rindex('*') + 1 : line.rindex('ms')]
                    results.append(float(result)) # Add each raw result
        while (i < len(results)):
            tup = []
            for x in range (6):
                tup.append(results[i+x])
            i = i + 6
            results_tuples.append(tup)
            # print(tup) # Print each tuple
        return results_tuples

    def gen_out(self, model_type, gpu_type = "1080_ti"):
        characterization_dir = self.hardware + "/characterization_data"
        if not os.path.exists(characterization_dir):
            os.mkdir(characterization_dir)

        if gpu_type == "1080_ti":
            gpu_data = eval("self." + model_type + "_exec_time")
            filename = "nvidia_gtx_1080_ti_" + model_type + ".txt"
        elif gpu_type == "960":
            gpu_data = eval("self." + model_type + "_exec_time_960")
            filename = "nvidia_gtx_960_" + model_type + ".txt"

        with open(characterization_dir+"/"+filename, 'w') as f:
            for n in range(gpu_data.size):
                batch_size = 4**n
                output = str(batch_size) + " , " + str(gpu_data[n])
                print(output, file=f)

    def __init__(self, root_dir = "./", hardware="nvidia_gtx_1080_ti"):
        self.root_dir = root_dir
        self.hardware = hardware

        directory = self.root_dir + self.hardware + "/raw_data/"

        self.wnd_exec_time   = np.asarray(self.parse_gpu(directory + "results_wnd.txt"))[:,5]
        self.rm1_exec_time   = np.asarray(self.parse_gpu(directory + "results_rm1.txt"))[:,5]
        self.rm2_exec_time   = np.asarray(self.parse_gpu(directory + "results_rm2.txt"))[:,5]
        self.rm3_exec_time   = np.asarray(self.parse_gpu(directory + "results_rm3.txt"))[:,5]
        self.ncf_exec_time   = np.asarray(self.parse_gpu(directory + "results_ncf.txt"))[:,5]
        self.mtwnd_exec_time = np.asarray(self.parse_gpu(directory + "results_mtwnd.txt"))[:,5]
        self.din_exec_time   = np.asarray(self.parse_gpu(directory + "results_din.txt"))[:,5]
        self.dien_exec_time = np.asarray(self.parse_gpu( directory + "results_dien.txt"))[:,5]

# Predicts GPU Execution Time based on characterization statistics
# Inputs: Model Name, Batch Size, Characterization Data
# Output: Total Execution Time (ms)
def predict_time(model_name = "wnd", input_batch_size = 1, gpu_data = None, gpu_type = "1080_ti"):

    if gpu_type == "1080_ti":
        input_batch_size_log = math.log(input_batch_size,4)
        batch_sizes_log      = np.arange(6)

        # Wide and Deep Case
        if model_name == "wnd":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.wnd_exec_time)
        # RM1 Case
        elif model_name == "rm1":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.rm1_exec_time)
        # RM2 Case
        elif model_name == "rm2":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.rm2_exec_time)
        # RM3 Case
        elif model_name == "rm3":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.rm3_exec_time)
        # NCF Case
        elif model_name == "ncf":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.ncf_exec_time)
        # Multi-Task Wide and Deep Case
        elif model_name == "mtwnd":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.mtwnd_exec_time)
        # Deep Interest Case
        elif model_name == "din":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.din_exec_time)
        # Deep Interest Evolution Case
        elif model_name == "dien":
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.dien_exec_time)

    elif gpu_type == "960":
        input_batch_size_log = np.log2(input_batch_size)

        # Wide and Deep Case
        if model_name == "wnd":
            batch_sizes_log = np.arange(gpu_data.wnd_exec_time_960.size)
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.wnd_exec_time_960)
        # RM1 Case
        elif model_name == "rm1":
            batch_sizes_log = np.arange(gpu_data.rm1_exec_time_960.size)
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.rm1_exec_time_960)
        # RM2 Case
        elif model_name == "rm2":
            batch_sizes_log = np.arange(gpu_data.rm2_exec_time_960.size)
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.rm2_exec_time_960)
        # RM3 Case
        elif model_name == "rm3":
            batch_sizes_log = np.arange(gpu_data.rm3_exec_time_960.size)
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.rm3_exec_time_960)
        # NCF Case
        elif model_name == "ncf":
            batch_sizes_log = np.arange(gpu_data.ncf_exec_time_960.size)
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.ncf_exec_time_960)
        # Multi-Task Wide and Deep Case
        elif model_name == "mtwnd":
            batch_sizes_log = np.arange(gpu_data.mtwnd_exec_time_960.size)
            exec_time = np.interp(input_batch_size_log, batch_sizes_log, gpu_data.mtwnd_exec_time_960)

    # if (input_batch_size_log > np.amax(batch_sizes_log)):
        # print("WARNING: Maximum input point for {GPU Type: " + gpu_type + " Model Name: " + model_name + "} is " + str(4**np.amax(batch_sizes_log)))

    return exec_time

if __name__ == "__main__":
    gpu_data = GPU_Data()

    gpu_data.gen_out("wnd")
    gpu_data.gen_out("rm1")
    gpu_data.gen_out("rm2")
    gpu_data.gen_out("rm3")
    gpu_data.gen_out("ncf")
    gpu_data.gen_out("mtwnd")
    gpu_data.gen_out("din")
    gpu_data.gen_out("dien")

    wnd_time   = predict_time("wnd", 10000, gpu_data)
    rm1_time   = predict_time("rm1", 10000, gpu_data)
    rm2_time   = predict_time("rm2", 10000, gpu_data)
    rm3_time   = predict_time("rm3", 10000, gpu_data)
    ncf_time   = predict_time("ncf", 10000, gpu_data)
    mtwnd_time = predict_time("mtwnd", 10000, gpu_data)
    din_time   = predict_time("din", 10000, gpu_data)
    dien_time  = predict_time("dien", 10000, gpu_data)

    print("[Nvidia GTX 1080 Ti] Wide and Deep GPU Execution Time at Batch Size 10000: ", wnd_time)
    print("[Nvidia GTX 1080 Ti] RM1 GPU Execution Time at Batch Size 10000: ", rm1_time)
    print("[Nvidia GTX 1080 Ti] RM2 GPU Execution Time at Batch Size 10000: ", rm2_time)
    print("[Nvidia GTX 1080 Ti] RM3 GPU Execution Time at Batch Size 10000: ", rm3_time)
    print("[Nvidia GTX 1080 Ti] NCF GPU Execution Time at Batch Size 10000: ", ncf_time)
    print("[Nvidia GTX 1080 Ti] Multi-Task Wide and Deep GPU Execution Time at Batch Size 10000: ", mtwnd_time)
    print("[Nvidia GTX 1080 Ti] DIN GPU Execution Time at Batch Size 10000: ", din_time)
    print("[Nvidia GTX 1080 Ti] DIEN GPU Execution Time at Batch Size 10000: ", dien_time)

