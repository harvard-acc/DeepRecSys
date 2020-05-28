from __future__ import absolute_import, division, print_function, unicode_literals

import functools

# others
import operator
import time

# numpy
import numpy as np

# cProfile
import cProfile

# caffe2
from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, dyndep, model_helper, net_drawer, workspace
from numpy import random as ra

import caffe2.python._import_c_extension as C
import sys

# =============================================================================
# define wrapper for wnd in Caffe2
# This is to decouple input queues for NCF network and the NCF network itself
# =============================================================================
class NCF_Wrapper(object):
    def FeedBlobWrapper(self, tag, val):
        if self.accel_en:
            _d = core.DeviceOption(caffe2_pb2.CUDA, 0)
            with core.DeviceScope(_d):
                workspace.FeedBlob(tag, val, device_option=_d)
        else:
            workspace.FeedBlob(tag, val)

    def __init__(
        self,
        cli_args,
        model=None,
        tag=None,
        enable_prof=False,
    ):
        super(NCF_Wrapper, self).__init__()
        self.args = cli_args

        # Accel Enable Flags
        accel_en = self.args.use_accel

        if accel_en:
            device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)
            naccels = C.num_cuda_devices  # 1
            print("(Wrapper) Using {} Accel(s)...".format(naccels))
        else:
            device_opt = core.DeviceOption(caffe2_pb2.CPU)
            print("(Wrapper) Using CPU...")

        self.accel_en = accel_en

        num_tables = len(cli_args.arch_embedding_size.split("-"))

        # We require 3 datastructures in caffe2 to enable non-blocking inputs for NCF
        # At a high-level each input needs an input queue. Inputs are enqueued
        # when they arrive on the "server" or "core" and dequeued by the
        # model's inference engine
        # Input Blob -> Input Net -> ID Q ===>NCF model
        self.id_qs          = []
        self.id_input_blobs = []
        self.id_input_nets  = []

        # Same thing for the lengths inputs
        self.len_qs          = []
        self.len_input_blobs = []
        self.len_input_nets  = []

        for i in range(num_tables):

            q, input_blob, net = self.build_ncf_sparse_queue(tag="id", qid=i)
            self.id_qs.append(q)
            self.id_input_blobs.append(input_blob)
            self.id_input_nets.append(net)

            q, input_blob, net = self.build_ncf_sparse_queue(tag="len", qid=i)
            self.len_qs.append(q)
            self.len_input_blobs.append(input_blob)
            self.len_input_nets.append(net)

        if self.args.queue:
            with core.DeviceScope(device_opt):
                self.ncf = NCF(cli_args, model, tag, enable_prof,
                                     id_qs = self.id_qs,
                                     len_qs = self.len_qs)
        else:
            with core.DeviceScope(device_opt):
                self.ncf  = NCF(cli_args, model, tag, enable_prof)


    def create(self, X, S_lengths, S_indices, T):
        if self.args.queue:
            self.ncf.create(X, S_lengths, S_indices, T,
                             id_qs = self.id_qs,
                             len_qs = self.len_qs)
        else:
            self.ncf.create(X, S_lengths, S_indices, T)


    # Run the Queues to provide inputs toNCF model
    def run_queues(self, ids, lengths, fc, batch_size):
        # Sparse features
        num_tables = len(self.args.arch_embedding_size.split("-"))
        for i in range(num_tables):
           self.FeedBlobWrapper( self.id_input_blobs[i], ids[i])
           workspace.RunNetOnce( self.id_input_nets[i].Proto() )

           self.FeedBlobWrapper( self.len_input_blobs[i], lengths[i])
           workspace.RunNetOnce( self.len_input_nets[i].Proto() )
    # =========================================================================
    # Helper  functions to build queues forNCF inputs (IDs, Lengths, FC)
    # in order to decouple blocking input operations
    # =========================================================================
    def build_ncf_sparse_queue(self, tag = "id", qid = None):
        q_net_name = tag + '_q_init_' + str(qid)
        q_net = core.Net(q_net_name)

        q_input_blob_name = tag + '_q_blob_' + str(qid)

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            q = q_net.CreateBlobsQueue([], q_input_blob_name,
                                           num_blobs=1,
                                           capacity=8)

        workspace.RunNetOnce(q_net)

        input_blob_name = tag + '_inputs_' + str(qid)
        input_net = core.Net(tag + '_input_net_' + str(qid))
        input_net.EnqueueBlobs([q, input_blob_name], [input_blob_name])

        return q, input_blob_name, input_net


class NCF (object):
    def FeedBlobWrapper(self, tag, val):
        if self.accel_en:
            _d = core.DeviceOption(caffe2_pb2.CUDA, 0)
            # with core.DeviceScope(_d):
            workspace.FeedBlob(tag, val, device_option=_d)
        else:
            workspace.FeedBlob(tag, val)

    def create_mlp(self, ln, model, tag):
        (tag_layer, tag_in, tag_out) = tag

        # build MLP layer by layer
        layers = []
        weights = []
        for i in range(1, ln.size):
            n = ln[i - 1]
            m = ln[i]

            # create tags
            tag_fc_w = tag_layer + ":::" + "fc" + str(i) + "_w"
            tag_fc_b = tag_layer + ":::" + "fc" + str(i) + "_b"
            tag_fc_y = tag_layer + ":::" + "fc" + str(i) + "_y"
            tag_fc_z = tag_layer + ":::" + "fc" + str(i) + "_z"
            if i == ln.size - 1:
                tag_fc_z = tag_out
            weights.append(tag_fc_w)
            weights.append(tag_fc_b)

            # initialize the weights
            # approach 1: custom Xavier input, output or two-sided fill
            mean = 0.0 # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n)) # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m) # np.sqrt(2 / (m + 1))
            b = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            self.FeedBlobWrapper(tag_fc_w, W)
            self.FeedBlobWrapper(tag_fc_b, b)

            # print("FC layer: ", i , m, n)

            # approach 1: construct fully connected operator using model.net
            fc = model.net.FC([tag_in, tag_fc_w, tag_fc_b], tag_fc_y,
                              engine=self.args.engine,
                              max_num_tasks=self.args.fc_workers)

            layers.append(fc)

            layer = model.net.Relu(tag_fc_y, tag_fc_z)

            tag_in = tag_fc_z
            layers.append(layer)

        # WARNING: the dependency between layers is implicit in the tags,
        # so only the last layer is added to the layers list. It will
        # later be used for interactions.
        return layers, weights

    def create_emb(self, m, ln, model, tag, id_qs = None, len_qs = None):
        (tag_layer, tag_in, tag_out) = tag
        mf_emb_l = []
        mlp_emb_l = []
        mf_weights_l = []
        mlp_weights_l = []

        assert(ln.size == 4)

        # Matrix factorization embedding tables
        for i in range(0, 2):
            n = ln[i]

            # create tags
            len_s = tag_layer + ":::" + "mf_sls" + str(i) + "_l"
            ind_s = tag_layer + ":::" + "mf_sls" + str(i) + "_i"
            tbl_s = tag_layer + ":::" + "mf_sls" + str(i) + "_w"
            sum_s = tag_layer + ":::" + "mf_sls" + str(i) + "_z"
            mf_weights_l.append(tbl_s)

            # initialize the weights
            # approach 1a: custom
            W = np.random.uniform(low=-np.sqrt(1 / n),
                                  high=np.sqrt(1 / n),
                                  size=(n, m)).astype(np.float32)
            # approach 1b: numpy rand
            # W = ra.rand(n, m).astype(np.float32)
            self.FeedBlobWrapper(tbl_s, W)

            if self.args.queue:
                # If want to have non-blocking IDs we have to dequue the input
                # ID blobs on the model side
                model.net.DequeueBlobs(id_qs[i], ind_s + "_pre_cast")
                model.net.Cast(ind_s + "_pre_cast", ind_s,
                               to=core.DataType.INT32)
                # Operator Mod is not found in Caffe2 latest build
                #model.net.Mod(ind_s + "_pre_mod", ind_s, divisor = n)

                # Dequeue lengths vector as well
                model.net.DequeueBlobs(len_qs[i], len_s)

            # create operator
            if self.accel_en:
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
                    EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                        engine=self.args.engine,
                        max_sls_workers=self.args.sls_workers)
            else:
                EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s])
                EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                    engine=self.args.engine,
                    max_sls_workers=self.args.sls_workers)

            mf_emb_l.append(EE)

        # MLP embedding tables
        for i in range(2, int(ln.size)):
            n = ln[i]
            # print(i)

            # create tags
            len_s = tag_layer + ":::" + "mlp_sls" + str(i-2) + "_l"
            ind_s = tag_layer + ":::" + "mlp_sls" + str(i-2) + "_i"
            tbl_s = tag_layer + ":::" + "mlp_sls" + str(i-2) + "_w"
            sum_s = tag_layer + ":::" + "mlp_sls" + str(i-2) + "_z"
            mlp_weights_l.append(tbl_s)

            # initialize the weights
            # approach 1a: custom
            W = np.random.uniform(low=-np.sqrt(1 / n),
                                  high=np.sqrt(1 / n),
                                  size=(n, m)).astype(np.float32)
            # approach 1b: numpy rand
            # W = ra.rand(n, m).astype(np.float32)
            self.FeedBlobWrapper(tbl_s, W)

            if self.args.queue:
                # If want to have non-blocking IDs we have to dequue the input
                # ID blobs on the model side
                model.net.DequeueBlobs(id_qs[i], ind_s + "_pre_cast")
                model.net.Cast(ind_s + "_pre_cast", ind_s,
                               to=core.DataType.INT32)
                # Operator Mod is not found in Caffe2 latest build
                #model.net.Mod(ind_s + "_pre_mod", ind_s, divisor = n)

                # Dequeue lengths vector as well
                model.net.DequeueBlobs(len_qs[i], len_s)

            # create operator
            if self.accel_en:
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
                    EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                        engine=self.args.engine,
                        max_num_tasks=self.args.sls_workers)
            else:
                EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                    engine=self.args.engine,
                    max_num_tasks=self.args.sls_workers)

            mlp_emb_l.append(EE)

        return mf_emb_l, mf_weights_l, mlp_emb_l, mlp_weights_l

    def create_mf_interaction(self, x, model, tag):
        # NCF multiplies both vectors from MF embedding tables
        R = model.net.Sum ( x, tag)

        return R


    def create_mlp_interaction(self, x, model, tag):
        # NCF multiplies both vectors from MF embedding tables
        tag_info = tag + "_info"

        R, R_info = model.net.Concat ( x, [tag, tag_info], axis=1)

        return R


    def create_sequential_forward_ops(self, id_qs = None, len_qs = None):
        # embeddings
        tag = (self.temb, self.tsin, self.tsout)
        emb_out = self.create_emb(self.m_spa, self.ln_emb,
                                  self.model, tag,
                                  id_qs = id_qs,
                                  len_qs = len_qs)
        self.mf_emb_l, self.mf_emb_w, self.mlp_emb_l, self.mlp_emb_w = emb_out


        Zmf  = self.create_mf_interaction(self.mf_emb_l, self.model, self.mf)
        Zmlp = self.create_mlp_interaction(self.mlp_emb_l, self.model, self.mlp)

        # top mlp
        tag = (self.mlpfc, Zmlp, self.mlpfc_out)
        self.top_l, self.top_w = self.create_mlp(self.ln_top[:-1], self.model, tag)

        ## Combine output of MF and MLP
        tag = 'feat_int'
        tag_info = 'feat_int_info'
        R, R_info = self.model.net.Concat ( [Zmf] + [self.top_l[-1]], [tag, tag_info], axis=1)

        int_size = self.args.arch_sparse_feature_size + self.ln_top[-2]
        tag = (self.final, R, self.tout)
        self.final_l, self.final_w = self.create_mlp( np.array([int_size] + [self.ln_top[-1]]),
                                                      self.model,
                                                      tag)

        ## setup the last output variable
        self.last_output = self.final_l

    def check_args(self, args):
      assert(args.arch_interaction_op == "cat" and "Sparse and dense features must be concatenated in NCF")

      ln_emb = np.fromstring(self.args.arch_embedding_size, dtype=int, sep="-")
      assert(len(ln_emb) == 4 and "NCF only have 4 embedding tables")

      assert(args.num_indices_per_lookup == 1 and "NCF has 1 lookup per table")

      return

    def __init__(
        self,
        cli_args,
        model=None,
        tag=None,
        enable_prof=False,
        id_qs = None,
        len_qs = None
    ):
        super(NCF, self).__init__()
        self.args = cli_args

        # Check to ensure we are configure wide and deep networks correctly
        self.check_args(self.args)

        ### parse command line arguments ###

        m_spa = cli_args.arch_sparse_feature_size
        ln_emb = np.fromstring(cli_args.arch_embedding_size, dtype=int, sep="-")
        num_fea = ln_emb.size + 1  # num sparse + num dense features
        # print("num features ", num_fea)

        accel_en = self.args.use_accel

        # Size of input dimension to TopFC layers is 2 * sparse feture size
        # The 2 comes from 2 embedding tables in MLP branch
        num_int = 2 * int(m_spa)

        arch_mlp_top_adjusted = str(num_int) + "-" + cli_args.arch_mlp_top
        # print("mlp_top is: ", arch_mlp_top_adjusted)
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

        ### initialize the model ###
        if model is None:
            global_init_opt = ["caffe2", "--caffe2_log_level=0"]
            if enable_prof:
                global_init_opt += [
                    "--logtostderr=0",
                    "--log_dir=$HOME",
                    #"--caffe2_logging_print_net_summary=1",
                ]
            workspace.GlobalInit(global_init_opt)
            self.set_tags()
            self.model = model_helper.ModelHelper(name="NCF", init_params=True)

            if cli_args:
              self.model.net.Proto().type = cli_args.caffe2_net_type
              self.model.net.Proto().num_workers = cli_args.inter_op_workers

        # save arguments
        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.ln_top = ln_top
        self.arch_interaction_op = cli_args.arch_interaction_op
        self.arch_interaction_itself = cli_args.arch_interaction_itself
        self.sigmoid_bot = -1 # TODO: Lets not hard-code this going forward
        self.sigmoid_top = ln_top.size - 1
        self.accel_en = accel_en

        return self.create_sequential_forward_ops(id_qs, len_qs)

    def set_tags(
        self,
        _tag_layer_top_mlp="top",
        _tag_layer_bot_mlp="bot",
        _tag_layer_embedding="emb",
        _tag_feature_dense_in="dense_in",
        _tag_feature_dense_out="dense_out",
        _tag_feature_sparse_in="sparse_in",
        _tag_feature_sparse_out="sparse_out",
        _tag_interaction="interaction",
        _tag_dense_output="prob_click",
        _tag_dense_target="target",
    ):
        # layer tags
        self.ttop = _tag_layer_top_mlp
        self.tbot = _tag_layer_bot_mlp
        self.temb = _tag_layer_embedding

        # MF
        self.mf = "mf"

        # MLP
        self.mlp = "mlp"
        self.mlpfc = "mlpfc"
        self.mlpfc_out = "mlpfc_out"


        # Final predictor
        self.final = "final"

        # sparse feature tags
        self.tsin = _tag_feature_sparse_in
        self.tsout = _tag_feature_sparse_out

        # output and target tags
        self.tint = _tag_interaction
        self.ttar = _tag_dense_target
        self.tout = _tag_dense_output

    def parameters(self):
        return self.model

    def create(self, X, S_lengths, S_indices, T, id_qs = None, len_qs=None):
        self.create_input(X, S_lengths, S_indices, T)
        self.create_model(X, S_lengths, S_indices, T)

    def create_input(self, X, S_lengths, S_indices, T):

        for i in range(2):
            len_s = self.temb + ":::" + "mf_sls" + str(i) + "_l"
            ind_s = self.temb + ":::" + "mf_sls" + str(i) + "_i"
            self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))
            self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

        for i in range(2):
            len_s = self.temb + ":::" + "mlp_sls" + str(i) + "_l"
            ind_s = self.temb + ":::" + "mlp_sls" + str(i) + "_i"
            self.FeedBlobWrapper(len_s, np.array(S_lengths[i+2]))
            self.FeedBlobWrapper(ind_s, np.array(S_indices[i+2]))

        # feed target data to blobs
        if T is not None:
            zeros_fp32 = np.zeros(T.shape).astype(np.float32)
            self.FeedBlobWrapper(self.ttar, zeros_fp32)


    def create_model(self, X, S_lengths, S_indices, T):
        #setup tril indices for the interactions
        offset = 1 if self.arch_interaction_itself else 0

        # create compute graph
        print("Trying to run NCF for the first time")
        if T is not None:
            # WARNING: RunNetOnce call is needed only if we use brew and ConstantFill.
            # We could use direct calls to self.model functions above to avoid it
            workspace.RunNetOnce(self.model.param_init_net)
            workspace.CreateNet(self.model.net)
            print("Ran NCF for the first time")


    def run(self, X=None, S_lengths=None, S_indices=None, enable_prof=False):
        # feed input data to blobs
        if not self.args.queue:
            # sparse features
            for i in range(2):
                ind_s = self.temb + ":::" + "mf_sls" + str(i) + "_i"
                self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

                len_s = self.temb + ":::" + "mf_sls" + str(i) + "_l"
                self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))

            for i in range(2):
                ind_s = self.temb + ":::" + "mlp_sls" + str(i) + "_i"
                self.FeedBlobWrapper(ind_s, np.array(S_indices[i+2]))

                len_s = self.temb + ":::" + "mlp_sls" + str(i) + "_l"
                self.FeedBlobWrapper(len_s, np.array(S_lengths[i+2]))

        load_time = time.time()
        # execute compute graph
        if enable_prof:
            workspace.C.benchmark_net(self.model.net.Name(), 0, 1, True)
        else:
            workspace.RunNet(self.model.net)
        return load_time


if __name__ == "__main__":
    ### import packages ###
    import sys
    import argparse

    sys.path.append("..")
    sys.path.append("../..")
    # data generation
    from data_generator.wnd_data_caffe2 import Wide_and_DeepDataGenerator

    from utils.utils import cli

    args = cli()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    use_accel = args.use_accel
    if use_accel:
        device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)
        naccels = C.num_cuda_devices  # 1
        print("Using {} Accel(s)...".format(naccels))
    else:
        device_opt = core.DeviceOption(caffe2_pb2.CPU)
        print("Using CPU...")

    # TODO: Make this NCF generator
    dc = Wide_and_DeepDataGenerator (args)
    if args.data_generation == "dataset":
        print("Error we have disabled this function currently....")
        sys.exit()
        # input and target data
        #(nbatches, lX, lS_l, lS_i, lT,
        # nbatches_test, lX_test, lS_l_test, lS_i_test, lT_test,
        # ln_emb, m_den) = dc.read_dataset(
        #    args.data_set, args.mini_batch_size, args.data_randomize, args.num_batches,
        #    True, args.raw_data_file, args.processed_data_file)
    else:
        # input data
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        if args.data_generation == "random":
            (nbatches, lX, lS_l, lS_i) = dc.generate_input_data()
        elif args.data_generation == "synthetic":
            (nbatches, lX, lS_l, lS_i) = dc.generate_synthetic_input_data(
                args.data_size, args.num_batches, args.mini_batch_size,
                args.round_targets, args.num_indices_per_lookup,
                args.num_indices_per_lookup_fixed, 32, ln_emb,
                args.data_trace_file, args.data_trace_enable_padding)
        else:
            sys.exit("ERROR: --data-generation="
                     + args.data_generation + " is not supported")

        # target data
        print("Generating output dataset")
        (nbatches, lT) = dc.generate_output_data()

    ### construct the neural network specified above ###
    print("Trying to initialize NCF")
    with core.DeviceScope(device_opt):
        ncf = NCF ( args )
    print("Initialized NCF Net")
    ncf.create(lX[0], lS_l[0], lS_i[0], lT[0])
    print("Created network")

    total_time = 0
    dload_time = 0
    k = 0

    time_start = time.time()
    print("Running networks")
    for k in range(args.nepochs):
        for j in range(nbatches):
            # forward and backward pass, where the latter runs only
            # when gradients and loss have been added to the net
            time_load_start = time.time()
            time_load_end   = ncf.run(lX[j], lS_l[j], lS_i[j], args.enable_profiling) # args.enable_profiling
            dload_time     += (time_load_end - time_load_start)

    time_end = time.time()
    dload_time *= 1000.
    total_time += (time_end - time_start) * 1000.
    print("Total data loading time: ***", dload_time, " ms")
    print("Total data loading time: ***", dload_time / (args.nepochs * nbatches), " ms/iter")
    print("Total computation time: ***", (total_time - dload_time), " ms")
    print("Total computation time: ***", (total_time - dload_time) / (args.nepochs * nbatches), " ms/iter")
    print("Total execution time: ***", total_time, " ms")
    print("Total execution time: ***", total_time / (args.nepochs * nbatches), " ms/iter")
