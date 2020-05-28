
from __future__ import absolute_import, division, print_function, unicode_literals

from utils.utils     import cli
from functools import reduce
import operator

from inferenceEngine import inferenceEngine
from accelInferenceEngine import accelInferenceEngine
from loadGenerator   import loadGenerator

from multiprocessing import Process, Queue
import csv
import sys
import os
import time
import numpy as np

import signal

def DeepRecSys():
  print("Running DeepRecSys")

  # ######################################################################
  # Get and print command line arguments for this experiment
  # ######################################################################
  args = cli()

  arg_keys = [str(key) for key in vars(args)]
  print("============================================================")
  print("DeepRecSys configuration")
  for key in arg_keys:
    print(key, getattr(args, key))
  print("============================================================")

  if args.queue == True:

    if args.model_accel:
      args.inference_engines += 1

    print("[DeepRecSys] total inference engine ", args.inference_engines)

    # Setup single request Queue and multiple response queues
    requestQueue    = Queue(maxsize=1024)
    accelRequestQueue = Queue(maxsize=32)
    pidQueue        = Queue()
    responseQueues  = []
    inferenceEngineReadyQueue = Queue()

    for _ in range(args.inference_engines):
      responseQueues.append(Queue())

    # Create load generator to mimic per-server load
    loadGeneratorReturnQueue = Queue()
    DeepRecLoadGenerator = Process( target = loadGenerator,
                        args   = (args, requestQueue, loadGeneratorReturnQueue, inferenceEngineReadyQueue, pidQueue, accelRequestQueue)
                      )

    # Create backend inference engines that consume requests from load
    # generator
    DeepRecEngines = []
    for i in range(args.inference_engines):
      if (args.model_accel) and (i == (args.inference_engines - 1)):
        p = Process( target = accelInferenceEngine,
                     args   = (args, accelRequestQueue, i, responseQueues[i], inferenceEngineReadyQueue)
                   )
      else:
        p = Process( target = inferenceEngine,
                     args   = (args, requestQueue, i, responseQueues[i], inferenceEngineReadyQueue)
                   )
      p.daemon = True
      DeepRecEngines.append(p)

    # Start all processes
    for i in range(args.inference_engines):
      DeepRecEngines[i].start()

    DeepRecLoadGenerator.start()

    responses_list = []
    inference_engines_finished = 0

    response_sets = {}
    response_latencies = []
    final_response_latencies = []

    request_granularity = int(args.req_granularity)

    while inference_engines_finished != args.inference_engines:
      for i in range(args.inference_engines):
        if (responseQueues[i].qsize()):
          response = responseQueues[i].get()

          # Process responses to determine what the running tail latency is and
          # send new batch-size to loadGenerator
          if response == None:
            inference_engines_finished += 1
            print("Joined ", inference_engines_finished, " inference engines")
            sys.stdout.flush()
          else:
            key = (response.epoch, response.batch_id, response.exp_packet)
            if key in response_sets.keys(): # Response already in the list
              curr_val = response_sets[key]

              val      = (response.arrival_time,
                          response.inference_end_time,
                          response.total_sub_batches)

              arr = min(curr_val[0], val[0])
              inf = max(curr_val[1], val[1])
              remain = curr_val[2]-1
              response_sets[ (response.epoch, response.batch_id, response.exp_packet) ] = (arr, inf, remain)
            else: # New response!
              arr = response.arrival_time
              inf = response.inference_end_time
              remain = response.total_sub_batches - 1

              response_sets[ (response.epoch, response.batch_id, response.exp_packet) ] = (arr, inf, remain)

            # If this request is over then we can go ahead and compute the
            # request latency in order to guide batch-scheduler
            if remain == 0:
              response_latencies.append( inf - arr )

              # If we are done finding the optimum batching and accelerator
              # partitioning threshold then we log the response latency to
              # measure packets later
              if not response.exp_packet:
                  final_response_latencies.append( inf - arr )

              if len(response_latencies) % request_granularity == 0:
                print("Running latency: ", np.percentile(response_latencies[int(-1 * request_granularity):], 95) * 1000.)
                sys.stdout.flush()
                # Add
                pidQueue.put ( np.percentile(response_latencies[int(-1 * request_granularity):], 95) * 1000. )

            # Add responses to final list
            responses_list.append(response.__dict__)

    print("Finished runing over the inference engines")
    sys.stdout.flush()

    log_dir = reduce(lambda x, y: x + y, args.log_file.split("/")[:-1])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(args.log_file, "w") as f:
        for response in responses_list:
          f.writelines(str(response) + "\n")

    # Join/end all processes
    DeepRecLoadGenerator.join()
    total_requests = loadGeneratorReturnQueue.get()

    cpu_sub_requests = total_requests[0]
    cpu_requests     = total_requests[1]
    accel_requests   = total_requests[2]

    agg_requests     = cpu_sub_requests + accel_requests

    print("Exiting DeepRecSys after printing ", len(responses_list), "/" , agg_requests)

    print("CPU sub requests ", cpu_sub_requests, "/" , agg_requests)
    print("CPU requests ", cpu_requests)
    print("Accel requests ", accel_requests, "/" , agg_requests)

    meas_qps_responses = list(filter(lambda x: (not x['exp_packet']) and (x['sub_id'] == 0), responses_list))

    initial_time = meas_qps_responses[0]['inference_end_time']
    end_time     = meas_qps_responses[-1]['inference_end_time']

    print("Measured QPS: ",  (len(meas_qps_responses)) / (end_time - initial_time))
    print("Measured p95 tail-latency: ",  np.percentile(final_response_latencies, 95) * 1000., " ms")
    print("Measured p99 tail-latency: ",  np.percentile(final_response_latencies, 99) * 1000., " ms")

    sys.stdout.flush()

    for i in range(args.inference_engines):
      DeepRecEngines[i].terminate()

  else: # No queue, run DeepRecSys in standalone mode
    inferenceEngine(args)

  return


if __name__=="__main__":
  DeepRecSys()
