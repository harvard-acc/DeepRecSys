from __future__ import absolute_import, division, print_function, unicode_literals

import queue as Q # import Python's Queue class for exception handling only
from multiprocessing import Queue, Process
from utils.packets   import ServiceRequest
from utils.utils  import debugPrint
import time
import numpy as np
import sys
import math
from scheduler import Scheduler


def model_arrival_times(args):
  arrival_time_delays = np.random.poisson(lam  = args.avg_arrival_rate,
                                          size = args.nepochs * args.num_batches)
  return arrival_time_delays


def model_batch_size_distribution(args):
  if args.batch_size_distribution == "normal":
    batch_size_distributions = np.random.normal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "lognormal":
    batch_size_distributions = np.random.lognormal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "fixed":
    batch_size_distributions = np.array([args.avg_mini_batch_size for _ in range(args.num_batches) ])

  elif args.batch_size_distribution == "file":
    percentiles = []
    batch_size_distributions = []
    with open(args.batch_dist_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        percentiles.append(float(line.rstrip()))

      for _ in range(args.num_batches):
        batch_size_distributions.append( int(percentiles[ int(np.random.uniform(0, len(percentiles))) ]) )

  for i in range(args.num_batches):
    batch_size_distributions[i] = int(max(min(batch_size_distributions[i], args.max_mini_batch_size), 1))
  return batch_size_distributions


def partition_requests(args, batch_size):
  batch_sizes = []

  while batch_size > 0:
    mini_batch_size = min(args.sub_task_batch_size, batch_size)
    batch_sizes.append(mini_batch_size)
    batch_size -= mini_batch_size

  return batch_sizes


def loadGenSleep( sleeptime ):
  if sleeptime > 0.0055:
    time.sleep(sleeptime)
  else:
    startTime = time.time()
    while (time.time() - startTime) < sleeptime:
      continue
  return


def loadGenerator(args,
                  requestQueue,
                  loadGeneratorReturnQueue,
                  inferenceEngineReadyQueue,
                  pidQueue,
                  accelRequestQueue):

  ready_engines = 0

  while ready_engines < args.inference_engines:
    inferenceEngineReadyQueue.get()
    ready_engines += 1

  arrival_time_delays = model_arrival_times(args)

  batch_size_distributions = model_batch_size_distribution(args)

  cpu_sub_requests = 0
  cpu_requests = 0
  accel_requests = 0

  minarr = args.min_arr_range
  maxarr = args.max_arr_range
  steps  = args.arr_steps

  # If we are tuning inference QPS we must generate a range of query arrival
  # rates (Poisson distribution)
  args_tune_batch_qps = args.tune_batch_qps
  args_tune_accel_qps = args.tune_accel_qps

  tuning_batch_qps = args.tune_batch_qps
  tuning_accel_qps   = False
  if tuning_batch_qps:
      possible_arrival_rates = np.logspace(math.log(minarr, 10), math.log(maxarr, 10), num=steps)
      arr_id = np.argmin(np.abs(possible_arrival_rates-args.avg_arrival_rate))

      print("Arrival rates to try: ", possible_arrival_rates)
      sys.stdout.flush()

  arrival_rate = args.avg_arrival_rate
  fixed_arrival_rate = None

  batch_configs = np.fromstring(args.batch_configs, dtype=int, sep="-")
  batch_config_attempt = 0

  if tuning_batch_qps:
      args.sub_task_batch_size = batch_configs[batch_config_attempt]

      # To start with lets not run with the Accel sweeps
      args.accel_request_size_thres = 1024

  epoch = 0
  exp_epochs = 0

  query_scheduler     = Scheduler(args, requestQueue, accelRequestQueue, pidQueue, mode="cpu")
  accel_query_scheduler = Scheduler(args, requestQueue, accelRequestQueue, pidQueue, mode="accel")

  while tuning_batch_qps or (exp_epochs < args.nepochs):
    for batch_id in range(args.num_batches):
      # absolute request ID
      request_id = epoch * args.num_batches + batch_id

      # ####################################################################
      # Batch size hill climbing
      # ####################################################################
      # If a new update from the controller arrives then update the
      # scheduler based on Hill-climbing. Here you can instantiate your own
      # scheduler to implement different scheduling algorithms for find optimal
      # task- and data-level parallelism configurations
      if tuning_batch_qps and (pidQueue.qsize() > 0):
          running_latency = pidQueue.get()
          out = query_scheduler.run(running_latency)

          args, arrival_rate, tuning_batch_qps = out

          if tuning_batch_qps == False:
              print("Finished batch size scheduler ")
              if args.model_accel and args.tune_accel_qps:
                  print("Starting accel scheduler")
                  tuning_accel_qps = True
              continue # start next batch_id to flush out response_latency queue

      # ####################################################################
      # Accel partition hill climbing
      # ####################################################################
      if args.model_accel and tuning_accel_qps and (pidQueue.qsize() > 0):
          running_latency = pidQueue.get()
          out = accel_query_scheduler.run(running_latency)
          args, arrival_rate, tuning_accel_qps = out

          if tuning_accel_qps == False:
              continue # start next batch_id to flush out response_latency queue

      request_size = int(batch_size_distributions[batch_id])

      if args.model_accel and (request_size >= args.accel_request_size_thres):
        # If request size is larger  than the threshold then we want to send it
        # over to the Accel.
        request = ServiceRequest(batch_id = batch_id,
                                 epoch = epoch,
                                 batch_size = request_size,
                                 sub_id = 0,
                                 total_sub_batches = 1,
                                 exp_packet = (tuning_batch_qps or tuning_accel_qps) )

        accel_requests += 1

        # add timestamp for request arriving onto server
        request.arrival_time = time.time()

        accelRequestQueue.put(request)

      else:

        batch_sizes = partition_requests(args, request_size)
        for i, batch_size in enumerate(batch_sizes):
          # create request

          request = ServiceRequest(batch_id = batch_id,
                                   epoch = epoch,
                                   batch_size = batch_size,
                                   sub_id = i,
                                   total_sub_batches = len(batch_sizes),
                                   exp_packet = (tuning_batch_qps or tuning_accel_qps))
          cpu_sub_requests += 1

          # add timestamp for request arriving onto server
          request.arrival_time = time.time()
          requestQueue.put(request)
        cpu_requests += 1

      arrival_time = np.random.poisson(lam = arrival_rate, size = 1)
      loadGenSleep( arrival_time[0] / 1000.   )

    epoch += 1

    if (tuning_batch_qps == False) and (tuning_accel_qps == False):
      exp_epochs += 1


  # Signal to the backend consumers that we are done
  for i in range(args.inference_engines):
    if args.model_accel and (i == (args.inference_engines-1)):
      debugPrint(args, "Load Generator", "sending done signal to " + str(i) + " accel engine")
      accelRequestQueue.put(None)
    else:
      debugPrint(args, "Load Generator", "sending done signal to " + str(i) + " cpu engine")
      requestQueue.put(None)

  # Return total number of sub-tasks simulated
  loadGeneratorReturnQueue.put( (cpu_sub_requests, cpu_requests, accel_requests) )

  if args_tune_batch_qps and not args_tune_accel_qps:
      print("Scheduler's Optimal batch_size configuration: ", query_scheduler.args.sub_task_batch_size, " @ arrival rate of ", query_scheduler.arrival_rate, "ms")
  elif args_tune_batch_qps and args_tune_accel_qps:
      print("Scheduler's Optimal batch_size configuration: ", query_scheduler.args.sub_task_batch_size )
      print("Scheduler's Optimal accel_size configuration: ", accel_query_scheduler.args.accel_request_size_thres , " @ arrival rate of", accel_query_scheduler.arrival_rate, "ms")

  sys.stdout.flush()

  return


if __name__=="__main__":
  main()
