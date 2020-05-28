from __future__ import print_function
import os, sys, json
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rgb(r,g,b):
    return (float(r)/256.,float(g)/256.,float(b)/256.)

def parse_operations(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    op_durations = {}
    op_durations_cpu = {}
    op_durations_gpu = {}

    for line in lines:
        if "ms." in line:
            op_type = line.rstrip().split()[3]
            value   = float(line.rstrip().split()[0])
            if op_type in op_durations.keys():
                op_durations[op_type].append(value)
            else:
                op_durations[op_type] = [value]

    for op_type in op_durations.keys():
        op_durations_cpu[op_type] = {}
        op_durations_gpu[op_type] = {}
        for i in range(len(batchsizes)):
            cpu_start = i * num_iterations
            cpu_end   = cpu_start + 400
            gpu_start = cpu_end
            gpu_end   = gpu_start + 400
            op_durations_cpu[op_type][batchsizes[i]] = np.mean(op_durations[op_type][cpu_start:cpu_end])
            op_durations_gpu[op_type][batchsizes[i]] = np.mean(op_durations[op_type][gpu_start:gpu_end])

    return op_durations_cpu, op_durations_gpu

def sweep_cmd(config_file, model_file, output_file):

    num_epochs  = 100
    num_batches = 4

    rt_config_cpu = "--inference_only --inter_op_workers 1 --caffe2_net_type simple --enable_profiling "
    rt_config_gpu = "--inference_only --inter_op_workers 1 --caffe2_net_type simple --use_accel --enable_profiling "
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
    if not os.path.exists("pdf"):
        os.mkdir("pdf")
    if not os.path.exists("png"):
        os.mkdir("png")
    
    sweep_cmd("wide_and_deep.json", "wide_and_deep.py", "raw_data/results_wnd_p.txt")
    sweep_cmd("dlrm_rm1.json", "dlrm_s_caffe2.py", "raw_data/results_rm1_p.txt")
    sweep_cmd("dlrm_rm2.json", "dlrm_s_caffe2.py", "raw_data/results_rm2_p.txt")
    sweep_cmd("dlrm_rm3.json", "dlrm_s_caffe2.py", "raw_data/results_rm3_p.txt")
    sweep_cmd("ncf.json", "ncf.py", "raw_data/results_ncf_p.txt")
    sweep_cmd("mtwnd.json", "multi_task_wnd.py", "raw_data/results_mtwnd_p.txt")
    sweep_cmd("din.json", "din.py", "raw_data/results_din_p.txt")
    sweep_cmd("dien.json", "dien.py", "raw_data/results_dien_p.txt")
    
    print("Operator Breakdown Sweeps Successful!\n")
    
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

    num_epochs     = 100
    num_batches    = 4
    num_devices    = 2 # CPU and GPU
    num_iterations = num_epochs * num_batches * num_devices

    batchsizes  = 4**np.arange(6)
    models      = ['rm1', 'rm2', 'rm3', 'ncf', 'wnd', 'mtwnd', 'din', 'dien']
    ops         = ['FC', 'SparseLengthsSum', 'Concat', 'Relu', 'Sum', 'RecurrentNetwork', 'Softmax']

    batchsizes_plot  = np.array((1,16,64,1024))
    models_plot      = models

    dataset = {}
    dataset['cpu'] = {}
    dataset['gpu'] = {}

    for model in models:
        
        filename = 'raw_data/results_'+model+"_p.txt"
        dataset['cpu'][model], dataset['gpu'][model] = parse_operations(filename)
        
        for batchsize in batchsizes:
            total_duration_cpu = 0
            total_duration_gpu = 0
            for op in dataset['cpu'][model].keys():
                if op in ops:
                    total_duration_cpu += dataset['cpu'][model][op][batchsize]
                    total_duration_gpu += dataset['gpu'][model][op][batchsize]
            print("Total Duration Found (CPU): ", model, batchsize, total_duration_cpu)
            print("Total Duration Found (GPU): ", model, batchsize, total_duration_gpu)
            for op in dataset['cpu'][model].keys():
                dataset['cpu'][model][op][batchsize] = (dataset['cpu'][model][op][batchsize]/total_duration_cpu)*100
                dataset['gpu'][model][op][batchsize] = (dataset['gpu'][model][op][batchsize]/total_duration_gpu)*100
                print("CPU: ", model, batchsize, op, dataset['cpu'][model][op][batchsize])
                print("GPU: ", model, batchsize, op, dataset['gpu'][model][op][batchsize])

    dataset_plot = {}
    dataset_plot['cpu'] = {}
    dataset_plot['gpu'] = {}

    for model in models_plot:
        for op in ops:
            if op not in dataset_plot['cpu'].keys():
                dataset_plot['cpu'][op] = []
                dataset_plot['gpu'][op] = []
                
    for model in models_plot:
        for batchsize in batchsizes_plot:
            for op in ops:
                if op in dataset['cpu'][model].keys():
                    dataset_plot['cpu'][op].append(dataset['cpu'][model][op][batchsize])
                    dataset_plot['gpu'][op].append(dataset['gpu'][model][op][batchsize])
                else:
                    dataset_plot['cpu'][op].append(0)
                    dataset_plot['gpu'][op].append(0)

    N = len(models_plot) * len(batchsizes_plot)

    fig, ax = plt.subplots(figsize = (16,8))
    ind = np.arange(0, N) # x locations for groups

    for i in range(len(models_plot)):
        for j in range(len(batchsizes_plot)):
            ind[i * len(batchsizes_plot) + j] = i * (len(batchsizes_plot)+1) + j
            
    width = 0.75

    running_bottom = np.zeros(N)
    ps = []

    i = 0
    for op in ops:
        a = 1.0
        if (i%2 == 1):
            a = 0.5
        ps.append(plt.bar(ind,
                         dataset_plot['cpu'][op],
                         width,
                         yerr = 0,
                         bottom = running_bottom,
                         color = clr[i],
                         edgecolor = 'black',
                         alpha = a)
                 )
        running_bottom += dataset_plot['cpu'][op]
        i = i + 1
    handle_ps = [p[0] for p in ps]

    ax.set_ylim(0, 100.)
    ax.set_ylabel('CPU Operator Breakdown (% runtime)', fontsize = 24)
    ax.legend(handle_ps, ops, loc='upper center', ncol=5,
            bbox_to_anchor=(0.5, 1.15), fontsize=18, fancybox=None, frameon=False)

    model_labels = ["RMC1", "RMC2", "RMC3", "NCF", "WND", "MT-WND", "DIN", "DIEN"]
    labels=()
    for model in model_labels:
        for j in range(len(batchsizes_plot)+1):
            if j == (len(batchsizes_plot)+1)//2:
                labels += ('\n' + model, )
            elif j < (len(batchsizes_plot)+1)//2:
                labels += (str(batchsizes_plot[j]),)
            else:
                labels += (str(batchsizes_plot[j-1]),)

    ind_xticks = np.sort(np.append(ind,np.arange(len(models_plot))*(len(batchsizes_plot)+1)+(len(batchsizes_plot)-1)/float(2)))
    plt.xticks(ind_xticks+width/2, (labels), fontsize = 12)
    plt.yticks(fontsize = 24)

    plt.tick_params(axis='x', length = 0)

    fig.tight_layout()

    plt.savefig('pdf/op_breakdown_cpu.pdf', bbox_inches="tight")
    plt.savefig('png/op_breakdown_cpu.png', bbox_inches="tight")

    N = len(models_plot) * len(batchsizes_plot)

    fig, ax = plt.subplots(figsize = (16,8))
    ind = np.arange(0, N) # x locations for groups

    for i in range(len(models_plot)):
        for j in range(len(batchsizes_plot)):
            ind[i * len(batchsizes_plot) + j] = i * (len(batchsizes_plot)+1) + j
            
    width = 0.75

    running_bottom = np.zeros(N)
    ps = []

    i = 0
    for op in ops:
        a = 1.0
        if (i%2 == 1):
            a = 0.5
        ps.append(plt.bar(ind,
                         dataset_plot['gpu'][op],
                         width,
                         yerr = 0,
                         bottom = running_bottom,
                         color = clr[i],
                         edgecolor = 'black',
                         alpha = a)
                 )
        running_bottom += dataset_plot['gpu'][op]
        i = i + 1
    handle_ps = [p[0] for p in ps]

    # op_names[1] = 'Arg'; op_names[7] = 'FusedMatMul'; op_names[10] = 'RetVal'

    ax.set_ylim(0, 100.)
    ax.set_ylabel('GPU Operator Breakdown (% runtime)', fontsize = 24)
    ax.legend(handle_ps, ops, loc='upper center', ncol=5,
            bbox_to_anchor=(0.5, 1.15), fontsize=18, fancybox=None, frameon=False)

    model_labels = ["RMC1", "RMC2", "RMC3", "NCF", "WND", "MT-WND", "DIN", "DIEN"]
    labels=()
    for model in model_labels:
        for j in range(len(batchsizes_plot)+1):
            if j == (len(batchsizes_plot)+1)//2:
                labels += ('\n' + model, )
            elif j < (len(batchsizes_plot)+1)//2:
                labels += (str(batchsizes_plot[j]),)
            else:
                labels += (str(batchsizes_plot[j-1]),)

    ind_xticks = np.sort(np.append(ind,np.arange(len(models_plot))*(len(batchsizes_plot)+1)+(len(batchsizes_plot)-1)/float(2)))
    plt.xticks(ind_xticks+width/2, (labels), fontsize = 12)
    plt.yticks(fontsize = 24)

    plt.tick_params(axis='x', length = 0)

    fig.tight_layout()

    plt.savefig('pdf/op_breakdown_gpu.pdf', bbox_inches="tight")
    plt.savefig('png/op_breakdown_gpu.png', bbox_inches="tight")
