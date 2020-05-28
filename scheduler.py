
import queue as Q # import Python's Queue class for exception handling only
import numpy as np
import math
import time
import sys


class Scheduler(object):
    def __init__(self, args, requestQueue, accelRequestQueue, pidQueue, mode="cpu"):
        self.args = args
        if not ((mode == "cpu") or (mode == "accel")):
            print("Unsupport scheduling backend")
            sys.exit()

        self.mode = mode

        ########################################################################
        # Range of input query arrival rates to try
        self.minarr = args.min_arr_range # min. arrival rate
        self.maxarr = args.max_arr_range # max. arrival rate
        self.steps  = args.arr_steps     # number of steps to try between min and max.

        # Generate possible arrival rate in logspace for more equal distribution
        self.possible_arrival_rates = np.logspace(math.log(self.minarr, 10), math.log(self.maxarr, 10), num=self.steps)
        self.arr_id = np.argmin(np.abs(self.possible_arrival_rates-args.avg_arrival_rate))
        ########################################################################

        self.qps_tried           = 0
        self.tried_arrival_rates = []

        self.config_qps     = []
        self.config_attempt = 0
        self.tuning_qps     = True

        if self.mode == "cpu":
            self.configs        = np.fromstring(args.batch_configs, dtype = int, sep = "-")
        if self.mode == "accel":
            self.configs        = np.fromstring(args.accel_configs, dtype = int, sep = "-")
            self.accel_config_attempt = 0

        self.requestQueue    = requestQueue
        self.accelRequestQueue = accelRequestQueue
        self.pidQueue        = pidQueue

        return

    def run(self, running_latency):
        ########################################################################
        # Computing new arrival rate based on current tail-latency and target tail-latency  (SLA)
        # We separate the region into three spaces:
        #     Increase arrival rate | Maintain arrival rate | Decrease Arrival Rate
        # Increase arrival rate if running_latency < target / (1 + stable_region)
        # Maintain arrival rate if target < running_latency > target / (1 + stable_region)
        # Decrease arrival rate if target > runn_latency
        # where stable_region is between 0 and 1

        if running_latency > self.args.target_latency:
          # if running latency is too high then we increase inter-arrival time (decrease QPS)
          self.arr_id = min( len(self.possible_arrival_rates) - 1, self.arr_id + 1)
        elif running_latency >= self.args.target_latency:
          self.arr_id = self.arr_id
        elif running_latency < self.args.target_latency / (1 + self.args.stable_region):
          # if running latency is too low then we increase inter-arrival time (decrease QPS)
          self.arr_id = max( 0, self.arr_id - 1)
        else:
          # if running latency is too low then we increase inter-arrival time (decrease QPS)
          self.arr_id = self.arr_id

        # Based on new arr_id lookup updated arrival_rates
        self.arrival_rate =  self.possible_arrival_rates[self.arr_id]
        self.tried_arrival_rates.append(self.arrival_rate)
        ########################################################################

        self.qps_tried += 1 # Number of input query arrival rates (or qps's) tried

        if self.qps_tried > self.args.sched_timeout:
          # Already tried `sched_timeout` number of input query arrival rates so now
          # we will determine the best one based on recent history.

          # Look back at previously attempted `arr_steps` number of attempts and pick
          # the median of those input query arrival rates
          self.arrival_rate = np.median(self.tried_arrival_rates[-1 * self.args.arr_steps:])
          print("Found fixed arrival rate:::", self.arrival_rate, "ms")
          sys.stdout.flush()

          self.config_qps.append(self.arrival_rate)
          self.config_attempt += 1

          if (len(self.config_qps) >= 2) and (self.config_qps[-1] > self.config_qps[-2]):
            # The most recent per-core query-size (or batch-size) is sub-optimal
            # compared to the previous attempt. Given hill-climbing we will back off
            # to the previous attempt and pick that as our optimal

            # We have found optimal configuration
            self.arrival_rate =  self.config_qps[self.config_attempt - 2]
            self.qps_tried = 0
            if self.tuning_qps:
              self.tuning_qps = False
              if self.mode == "cpu":
                  self.args.sub_task_batch_size = self.configs[self.config_attempt - 2]
                  print("[found opt] Optimal batch_size configuration: ", self.args.sub_task_batch_size, " @ arrival rate of ", self.arrival_rate, "ms")
                  sys.stdout.flush()
              elif self.mode == "accel":
                  self.args.accel_request_size_thres = self.configs[self.config_attempt - 2]
                  print("[found opt] Optimal accel configuration: ", self.args.accel_request_size_thres, " @ arrival rate of ", self.arrival_rate, "ms")
                  sys.stdout.flush()

              else:
                  print("Unsupport scheduling backend")
                  sys.exit()

          elif (len(self.config_qps) == len(self.configs)):
            # If we tried all possible configurations and the last one was
            # not worse than the 2nd to last then we know that the optimal configuration is the last one

            # We have found optimal configuration
            self.arrival_rate = min(self.config_qps)
            best_attempt = np.argmin(self.config_qps)

            self.qps_tried = 0
            if self.tuning_qps:
              self.tuning_qps = False
              if self.mode == "cpu":
                  self.args.sub_task_batch_size = self.configs[best_attempt]
                  print("[tried all ] Optimal batch_size configuration: ", self.args.sub_task_batch_size, " @ arrival rate of ", self.arrival_rate, "ms")
                  sys.stdout.flush()
              elif self.mode == "accel":
                  self.args.accel_request_size_thres = self.configs[best_attempt]
                  print("[tried all ] Optimal accel configuration: ", self.args.accel_request_size_thres, " @ arrival rate of ", self.arrival_rate, "ms")
                  sys.stdout.flush()
              else:
                  print("Unsupport scheduling backend")
                  sys.exit()

          else:
            # Else we find that that the achievable QPS has gone up so we
            # need to keep trying optimal batch-size configurations. We
            # should not be equal to the length  of configs as that
            # would have been caught by the previous condition
            if self.tuning_qps:
                if self.mode == "cpu":
                    self.args.sub_task_batch_size = self.configs[self.config_attempt]
                elif self.mode == "accel":
                    self.args.accel_request_size_thres = self.configs[self.config_attempt]
                else:
                    print("Unsupport scheduling backend")
                    sys.exit()

            # Need to try the next batch-size configuration so have to
            # reset the hill-climbing
            self.tried_arrival_rates = []
            self.qps_tried = 0
            self.arrival_rate = self.args.avg_arrival_rate
            self.arr_id = np.argmin(np.abs(self.possible_arrival_rates-self.args.avg_arrival_rate))

          # Drain the request and wait for the next iteration queue
          while self.requestQueue.qsize():
              try:
                  self.requestQueue.get(False)
              except Q.Empty: # if queue is empty we are done drain and can pass along
                  pass

          # Drain the request and wait for the next iteration queue
          while self.accelRequestQueue.qsize():
              try:
                  self.accelRequestQueue.get(False)
              except Q.Empty: # if queue is empty we are done drain and can pass along
                  pass

          time.sleep(3)

          while self.pidQueue.qsize() > 0:
            self.pidQueue.get()

        out = (self.args, self.arrival_rate, self.tuning_qps)

        return out

