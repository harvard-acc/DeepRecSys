
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from utils.packets   import ServiceResponse
from utils.utils   import debugPrint

# data generation
import threading
from multiprocessing import Queue
from accelerator.predict_execution import *

import time
import sys


def accelInferenceEngine(args,
                    requestQueue=None,
                    engine_id=None,
                    responseQueue=None,
                    inferenceEngineReadyQueue=None):

  ### some basic setup ###
  np.random.seed(args.numpy_rand_seed)
  np.set_printoptions(precision=args.print_precision)

  if requestQueue == None:
    print("If you want to run Accel in isolation please use the DeepRecBench/models/ directory directly")
    sys.stdout.flush()
    sys.exit()

  else:
    inferenceEngineReadyQueue.put(True)

    model_name = args.model_name
    if not (model_name in ["wnd", "rm1", "rm2", "rm3", "din", "dien", "mtwnd"]):
      print("Model not found in ones supported")
      sys.stdout.flush()
      sys.exit()

    accel_data = GPU_Data(root_dir = args.accel_root_dir, hardware="nvidia_gtx_1080_ti")

    while True:
      debugPrint(args, "Accel", "Trying to pull request")
      request = requestQueue.get()
      debugPrint(args, "Accel", "Pulled request")

      if request is None:
        debugPrint(args, "Accel", "Sending final done signal")
        responseQueue.put(None)
        debugPrint(args, "Accel", "Sent final done signal")
        return

      batch_id   = request.batch_id
      batch_size = request.batch_size

      start_time = time.time()

      # Model Accel execution time
      # For GPUs this based on real measured hardware performance (inference and dataloading)
      # based on accelerator/predict_execution.py
      eval_time = predict_time(model_name, batch_size, accel_data)
      time.sleep(eval_time / 1000. ) # Eval time is in milli-seconds

      end_time = time.time()

      response = ServiceResponse( consumer_id = engine_id,
                                  epoch = request.epoch,
                                  batch_id = request.batch_id,
                                  batch_size = request.batch_size,
                                  arrival_time = request.arrival_time,
                                  process_start_time = start_time,
                                  queue_end_time = end_time,
                                  inference_end_time = end_time,
                                  out_batch_size = request.batch_size,
                                  total_sub_batches = request.total_sub_batches,
                                  exp_packet = request.exp_packet,
                                  sub_id = request.sub_id,
                                )

      debugPrint(args, "Accel", "Sending response back")
      responseQueue.put(response)
      debugPrint(args, "Accel", "Sent response back")

  return

