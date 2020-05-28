from __future__ import absolute_import, division, print_function, unicode_literals

from models.dlrm_s_caffe2 import DLRM_Wrapper
from models.wide_and_deep import Wide_and_Deep_Wrapper
from models.ncf import NCF_Wrapper
from models.din import DIN_Wrapper
from models.dien import DIEN_Wrapper
from models.multi_task_wnd import MT_Wide_and_Deep_Wrapper

import numpy as np

from utils.packets   import ServiceResponse

# data generation
from data_generator.dlrm_data_caffe2 import DLRMDataGenerator
import threading
from multiprocessing import Queue

from caffe2.proto import caffe2_pb2
from caffe2.python import core,workspace
import time
import sys

import caffe2.python._import_c_extension as C

def run_model(model, args, internal_logging, responseQueue):

    top_fc_layers = args.arch_mlp_top.split("-")
    fc_tag = "top:::fc" + str(len(top_fc_layers)-1) + "_z"
    while True:

        if args.model_type == "dlrm":
          model.dlrm.run()
        elif args.model_type == "wnd":
          model.wnd.run()
        elif args.model_type == "ncf":
          model.ncf.run()
        elif args.model_type == "din":
          model.din.run()
        elif args.model_type == "mtwnd":
          model.mtwnd.run()
        elif args.model_type == "dien":
          model.dien.run()

        response                    = internal_logging.get()

        if response == None:
            return

        inference_end_time          = time.time()
        response.inference_end_time = inference_end_time
        #out_size = np.array(workspace.FetchBlob(fc_tag)).size / int(top_fc_layers[-2])
        if args.model_type == "ncf":
            ln_top = np.fromstring(args.arch_mlp_top, dtype=int, sep="-")
            out_size = np.array(workspace.FetchBlob(model.ncf.tout)).size / ln_top[-1]
        else:
            out_size = np.array(workspace.FetchBlob(fc_tag)).size / int(top_fc_layers[-2])
        response.out_batch_size = out_size
        responseQueue.put(response)


def inferenceEngine(args,
                    requestQueue=None,
                    engine_id=None,
                    responseQueue=None,
                    inferenceEngineReadyQueue=None):

  q_inference_logging = Queue()
  q_inference_done    = Queue()

  ### some basic setup ###
  np.random.seed(args.numpy_rand_seed)
  np.set_printoptions(precision=args.print_precision)

  # #########################################################################
  # Data generation
  # - with multiple model implementations this should instantiate the
  # particular model class' data generator
  # #########################################################################
  if args.model_type == "dlrm":
    datagen                    = DLRMDataGenerator(args)

    (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
    (nbatches, lT)             = datagen.generate_output_data()
    # construct the neural network specified by command line arguments ###

    model = DLRM_Wrapper( args )
    model.create(lX[0], lS_l[0], lS_i[0], lT[0])

  elif args.model_type == "wnd":
    datagen                    = DLRMDataGenerator(args)

    (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
    (nbatches, lT)             = datagen.generate_output_data()

    model = Wide_and_Deep_Wrapper( args )
    model.create(lX[0], lS_l[0], lS_i[0], lT[0])

  elif args.model_type == "mtwnd":
    datagen                    = DLRMDataGenerator(args)

    (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
    (nbatches, lT)             = datagen.generate_output_data()
    # construct the neural network specified by command line arguments ###

    model = MT_Wide_and_Deep_Wrapper( args )
    model.create(lX[0], lS_l[0], lS_i[0], lT[0])

  elif args.model_type == "ncf":
    datagen                    = DLRMDataGenerator(args)

    (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
    (nbatches, lT)             = datagen.generate_output_data()

    model = NCF_Wrapper( args )
    model.create(lX[0], lS_l[0], lS_i[0], lT[0])

  elif args.model_type == "din":
    datagen                    = DLRMDataGenerator(args)

    (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
    (nbatches, lT)             = datagen.generate_output_data()

    model = DIN_Wrapper( args )
    model.create(lX[0], lS_l[0], lS_i[0], lT[0])

  elif args.model_type == "dien":
    datagen                    = DLRMDataGenerator(args)

    (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
    (nbatches, lT)             = datagen.generate_output_data()
    # construct the neural network specified by command line arguments ###

    model = DIEN_Wrapper( args )
    model.create(lX[0], lS_l[0], lS_i[0], lT[0])

  if requestQueue == None:
    total_time = 0
    dload_time = 0

    time_start = time.time()
    for k in range(args.nepochs):
      for j in range(nbatches):
        if args.model_type == "dlrm":
          time_load_start = time.time()
          time_load_end   = model.dlrm.run(lX[j], lS_l[j], lS_i[j])
          dload_time     += (time_load_end - time_load_start)
        elif args.model_type == "wnd":
          time_load_start = time.time()
          time_load_end   = model.wnd.run(lX[j], lS_l[j], lS_i[j])
          dload_time     += (time_load_end - time_load_start)
        elif args.model_type == "mtwnd":
          time_load_start = time.time()
          time_load_end   = model.mtwnd.run(lX[j], lS_l[j], lS_i[j])
          dload_time     += (time_load_end - time_load_start)
        elif args.model_type == "ncf":
          time_load_start = time.time()
          time_load_end   = model.ncf.run(lX[j], lS_l[j], lS_i[j])
          dload_time     += (time_load_end - time_load_start)
        elif args.model_type == "din":
          model.din.run(lX[j], lS_l[j], lS_i[j])
        elif args.model_type == "dien":
          model.dien.run(lX[j], lS_l[j], lS_i[j])

    time_end = time.time()
    dload_time *= 1000.
    total_time += (time_end - time_start) * 1000.
    print("Total data loading time: ***", dload_time, " ms")
    print("Total data loading time: ***", dload_time / (args.nepochs * nbatches), " ms/iter")
    print("Total computation time: ***", (total_time - dload_time), " ms")
    print("Total computation time: ***", (total_time - dload_time) / (args.nepochs * nbatches), " ms/iter")
    print("Total execution time: ***", total_time, " ms")
    print("Total execution time: ***", total_time / (args.nepochs * nbatches), " ms/iter")

  else:
    # Run DLRM model inferences in a separate thread in order to decouple input
    # and inference run-times (non-blocking FeedBlob() caffe2 call)
    inference_thread = threading.Thread( target=run_model,
                                         args = (model,
                                                 args,
                                                 q_inference_logging,
                                                 responseQueue
                                                )
                                       )
    inference_thread.daemon = True
    inference_thread.start()

    total_time = 0

    while True:
      inferenceEngineReadyQueue.put(True)
      request = requestQueue.get()

      if request is None:
        time.sleep(4)
        q_inference_logging.put(None)
        responseQueue.put(None)
        return

      batch_id   = request.batch_id
      lS_l_curr  = np.transpose(np.array(lS_l[batch_id]))
      lS_l_curr  = np.transpose(np.array(lS_l_curr[:request.batch_size]))

      lS_ids_curr = np.array(lS_i[batch_id])

      lS_ids_curr = np.array(lS_ids_curr[:][:, :request.batch_size * args.num_indices_per_lookup])

      start_time = time.time()

      # Parameterized on batch_size
      model.run_queues(lS_ids_curr,
                       lS_l_curr,
                       lX[batch_id][:request.batch_size],
                       request.batch_size
                      )

      end_time = time.time()
      response = ServiceResponse( consumer_id = engine_id,
                                  epoch = request.epoch,
                                  batch_id = request.batch_id,
                                  batch_size = request.batch_size,
                                  arrival_time = request.arrival_time,
                                  process_start_time = start_time,
                                  queue_end_time = end_time,
                                  total_sub_batches = request.total_sub_batches,
                                  exp_packet = request.exp_packet,
                                  sub_id = request.sub_id
                                )

      q_inference_logging.put(response)

  return


if __name__=="__main__":
  from utils.utils import cli
  args = cli()

  inferenceEngine(args)
