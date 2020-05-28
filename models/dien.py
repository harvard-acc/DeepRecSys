# Following model architecture specified by Deep Interest Network paper
# (Alibaba)
# https://arxiv.org/pdf/1706.06978.pdf
from __future__ import absolute_import, division, print_function, unicode_literals

import functools

# others
import operator
import time

# numpy
import numpy as np

# caffe2
from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, model_helper, net_drawer, workspace, rnn_cell
# import rnn
from numpy import random as ra
import caffe2.python._import_c_extension as C
import sys

# =============================================================================
# define wrapper for dien in Caffe2
# This is to decouple input queues for DIEN network and the DIEN network itself
# =============================================================================
class DIEN_Wrapper(object):
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
        super(DIEN_Wrapper, self).__init__()
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

        # We require 3 datastructures in caffe2 to enable non-blocking inputs for DIEN
        # At a high-level each input needs an input queue. Inputs are enqueued
        # when they arrive on the "server" or "core" and dequeued by the
        # model's inference engine
        # Input Blob -> Input Net -> ID Q ===> DIEN model
        self.id_qs          = []
        self.id_input_blobs = []
        self.id_input_nets  = []

        # Same thing for the lengths inputs
        self.len_qs          = []
        self.len_input_blobs = []
        self.len_input_nets  = []

        self.seq_q, self.seq_input_blob, self.seq_input_net = self.build_dien_seq_queue()
        self.hid_q, self.hid_input_blob, self.hid_input_net = self.build_dien_hid_queue()

        for i in range(num_tables):

            q, input_blob, net = self.build_dien_sparse_queue(tag="id", qid=i)
            self.id_qs.append(q)
            self.id_input_blobs.append(input_blob)
            self.id_input_nets.append(net)

            q, input_blob, net = self.build_dien_sparse_queue(tag="len", qid=i)
            self.len_qs.append(q)
            self.len_input_blobs.append(input_blob)
            self.len_input_nets.append(net)

        if self.args.queue:
            with core.DeviceScope(device_opt):
                self.dien = DIEN_Net(cli_args, model, tag, enable_prof,
                                     id_qs = self.id_qs,
                                     len_qs = self.len_qs,
                                     seq_q = self.seq_q,
                                     hid_q = self.hid_q)
        else:
            with core.DeviceScope(device_opt):
                self.dien = DIEN_Net(cli_args, model, tag, enable_prof)


    def create(self, X, S_lengths, S_indices, T):
        if self.args.queue:
            self.dien.create(X, S_lengths, S_indices, T,
                             id_qs = self.id_qs,
                             len_qs = self.len_qs)
        else:
            self.dien.create(X, S_lengths, S_indices, T)


    # Run the Queues to provide inputs to DIEN model
    def run_queues(self, ids, lengths, fc, batch_size):
        # Sparse features
        ln_emb = np.fromstring(self.args.arch_embedding_size, dtype=int, sep="-")
        num_tables = len(self.args.arch_embedding_size.split("-"))
        sequence_lengths = np.zeros(batch_size).astype(np.int32)
        sequence_lengths += (len(ln_emb) - 3)

        initial_h_data = np.zeros( (batch_size, self.args.hidden_size) ).astype(np.float32)

        self.FeedBlobWrapper( self.seq_input_blob, sequence_lengths)
        self.FeedBlobWrapper( self.hid_input_blob, initial_h_data)

        workspace.RunNetOnce( self.seq_input_net.Proto() )
        workspace.RunNetOnce( self.hid_input_net.Proto() )

        for i in range(num_tables):
           self.FeedBlobWrapper( self.id_input_blobs[i], ids[i])
           workspace.RunNetOnce( self.id_input_nets[i].Proto() )

           self.FeedBlobWrapper( self.len_input_blobs[i], lengths[i])
           workspace.RunNetOnce( self.len_input_nets[i].Proto() )
    # =========================================================================
    # Helper  functions to build queues for DIEN inputs (IDs, Lengths, FC)
    # in order to decouple blocking input operations
    # =========================================================================
    def build_dien_sparse_queue(self, tag = "id", qid = None):
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

    def build_dien_seq_queue(self, ):
        seq_q_net_name = 'seq_q_init'
        seq_q_net = core.Net(seq_q_net_name)

        seq_q_input_blob_name = 'seq_q_blob'

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            seq_q = seq_q_net.CreateBlobsQueue([],
                                             seq_q_input_blob_name,
                                             num_blobs=1,
                                             capacity=8)

        workspace.RunNetOnce(seq_q_net)

        seq_input_blob_name = 'seq_inputs'
        seq_input_net = core.Net('seq_input_net')
        seq_input_net.EnqueueBlobs([seq_q, seq_input_blob_name], [seq_input_blob_name])

        return seq_q, seq_input_blob_name, seq_input_net

    def build_dien_hid_queue(self, ):
        hid_q_net_name = 'hid_q_init'
        hid_q_net = core.Net(hid_q_net_name)

        hid_q_input_blob_name = 'hid_q_blob'

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            hid_q = hid_q_net.CreateBlobsQueue([],
                                             hid_q_input_blob_name,
                                             num_blobs=1,
                                             capacity=8)

        workspace.RunNetOnce(hid_q_net)

        hid_input_blob_name = 'hid_inputs'
        hid_input_net = core.Net('hid_input_net')
        hid_input_net.EnqueueBlobs([hid_q, hid_input_blob_name], [hid_input_blob_name])

        return hid_q, hid_input_blob_name, hid_input_net


class DIEN_Net(object):
    def FeedBlobWrapper(self, tag, val):
        if self.accel_en:
            _d = core.DeviceOption(caffe2_pb2.CUDA, 0)
            # with core.DeviceScope(_d):
            workspace.FeedBlob(tag, val, device_option=_d)
        else:
            workspace.FeedBlob(tag, val)

    def create_mlp(self, ln, model, tag, fc_q = None):
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

            # approach 1: construct fully connected operator using model.net
            if self.args.queue and (fc_q is not None) and (i == 1):
                # Dequeue lengths vector as well
                model.net.DequeueBlobs(fc_q, tag_in)
                fc = model.net.FC([tag_in, tag_fc_w, tag_fc_b], tag_fc_y,
                                  engine=self.args.engine,
                                  max_num_tasks=self.args.fc_workers)
            else:
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
        emb_l = []
        weights_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # create tags
            len_s = tag_layer + ":::" + "sls" + str(i) + "_l"
            ind_s = tag_layer + ":::" + "sls" + str(i) + "_i"
            tbl_s = tag_layer + ":::" + "sls" + str(i) + "_w"
            sum_s = tag_layer + ":::" + "sls" + str(i) + "_z"
            weights_l.append(tbl_s)

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
            emb_l.append(EE)

        return emb_l, weights_l

    def create_gru_unit(self, emb_ls, user_emb_ids, model, tag, seq_q, hid_q):
        (tag_layer, tag_in, tag_out) = tag

        emb_ls_str = []
        for user_emb_id in user_emb_ids:
            emb_ls_str.append( emb_ls[user_emb_id] )

        tag_cat = tag_layer + ":::_rnn_inputs"
        tag_cat_info = tag_cat + "_info"
        rnn_inputs, info = model.net.Concat( emb_ls_str, [tag_cat, tag_cat_info])
        rnn_shape = model.net.Reshape(rnn_inputs, [tag_layer + ":::rnn_shape", "old_shape"],
                shape=(len(user_emb_ids), -1, self.input_size))

        gates_t_w_data = np.random.randn( self.args.hidden_size, self.args.hidden_size).astype(np.float32)
        gates_t_b_data = np.random.randn( self.args.hidden_size).astype(np.float32)
        i2h_w_data     = np.random.randn( self.args.hidden_size, self.input_size).astype(np.float32)
        i2h_b_data     = np.random.randn( self.args.hidden_size).astype(np.float32)

        self.FeedBlobWrapper( 'rnn_0/gates_t_w', gates_t_w_data)
        self.FeedBlobWrapper( 'rnn_0/gates_t_b', gates_t_b_data)
        self.FeedBlobWrapper( 'rnn_0/i2h_w', i2h_w_data)
        self.FeedBlobWrapper( 'rnn_0/i2h_b', i2h_b_data)

        if  seq_q:
            model.net.DequeueBlobs(seq_q, "seq_lengths")
        if  hid_q:
            model.net.DequeueBlobs(hid_q, "initial_h")

        rnn_0_out, _ = rnn_cell.BasicRNN(model,
                            tag_layer + ":::rnn_shape",
                            'seq_lengths',
                            ['initial_h'],
                            self.input_size,
                            self.args.hidden_size,
                            "rnn_0",
                            activation="tanh",
                            forward_only=True)

        output = brew.fc( self.model,
                          rnn_0_out,
                          None,
                          dim_in = self.args.hidden_size,
                          dim_out= self.args.hidden_size,
                          axis=2,
                          engine=self.args.engine,
                          max_num_tasks=self.args.fc_workers)

        output = brew.softmax(self.model, output, axis=2)
        output = brew.sum(self.model, rnn_0_out, output, axis=2)

        # TODO: Need to make input_h_data an input to the overall model due to
        # batch-size
        gates_t_w_data = np.random.randn( self.args.hidden_size, self.args.hidden_size).astype(np.float32)
        gates_t_b_data = np.random.randn( self.args.hidden_size).astype(np.float32)
        i2h_w_data     = np.random.randn( self.args.hidden_size, self.args.hidden_size).astype(np.float32)
        i2h_b_data     = np.random.randn( self.args.hidden_size).astype(np.float32)

        self.FeedBlobWrapper( 'rnn_1/gates_t_w', gates_t_w_data)
        self.FeedBlobWrapper( 'rnn_1/gates_t_b', gates_t_b_data)
        self.FeedBlobWrapper( 'rnn_1/i2h_w', i2h_w_data)
        self.FeedBlobWrapper( 'rnn_1/i2h_b', i2h_b_data)

        rnn_1_all_out, rnn_1_out = rnn_cell.BasicRNN(model,
                                                    output,
                                                    'seq_lengths',
                                                    ['initial_h'],
                                                    self.args.hidden_size,
                                                    self.args.hidden_size,
                                                    "rnn_1",
                                                    activation="tanh",
                                                    forward_only=True)

        return rnn_1_out


    def create_sequential_forward_ops(self, id_qs = None, len_qs = None, seq_q = None, hid_q = None):
        self.input_size  = self.args.arch_sparse_feature_size

        # embeddings
        tag = (self.temb, self.tsin, self.tsout)
        self.emb_l, self.emb_w = self.create_emb(self.m_spa, self.ln_emb,
                                                    self.model, tag,
                                                    id_qs = id_qs,
                                                    len_qs = len_qs)

        # Deep Interest network has 4 types of features, user profile, user
        # behavior, candidate ad, context features
        user_profile_emb      = 0
        user_behavior_emb     = list(range(1, len(self.ln_emb) - 2))
        candidate_ad_emb      = len(self.ln_emb) - 2
        context_features_emb  = len(self.ln_emb) - 1
        '''
        print(user_profile_emb)
        print(user_behavior_emb)
        print(candidate_ad_emb)
        print(context_features_emb)
        '''

        tag = (self.tgru, self.tsout, self.tgruout)
        gru_out = self.create_gru_unit(self.emb_l,
                                       user_behavior_emb,
                                       self.model,
                                       tag,
                                       seq_q = seq_q,
                                       hid_q = hid_q)

        out = self.model.net.Flatten(gru_out, axis=2)

        concat_inputs = [out] + [self.emb_l[user_profile_emb]]
        concat_inputs += [self.emb_l[candidate_ad_emb]]
        concat_inputs += [self.emb_l[context_features_emb]]
        tag = self.tgru + ":::concat"
        tag_info = tag + "_info"
        topFC_in, _ = self.model.net.Concat( concat_inputs, [tag, tag_info])

        # There are 4 sets out output in DIEN: User profile, user behavior,
        # candidate ad, context features
        # As aresult we must have at least 4 embedding tables
        num_int = self.args.hidden_size + (3 * self.args.arch_sparse_feature_size)

        arch_mlp_top_adjusted = str(num_int) + "-" + self.args.arch_mlp_top
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
        tag = (self.ttop, topFC_in, self.tout)
        self.top_l, self.top_w = self.create_mlp(ln_top, self.model, tag)
        ## setup the last output variable
        self.last_output = self.top_l[-1]


    def __init__(
        self,
        cli_args,
        model=None,
        tag=None,
        enable_prof=False,
        id_qs = None,
        len_qs = None,
        seq_q = None,
        hid_q = None
    ):
        super(DIEN_Net, self).__init__()
        self.args = cli_args

        m_spa = cli_args.arch_sparse_feature_size
        ln_emb = np.fromstring(cli_args.arch_embedding_size, dtype=int, sep="-")
        num_fea = ln_emb.size + 1  # num sparse + num dense features

        accel_en = self.args.use_accel

        assert( len(ln_emb) >= 4 )

        global_init_opt = ["caffe2", "--caffe2_log_level=1"]
        workspace.GlobalInit(global_init_opt)
        self.set_tags()
        self.model = model_helper.ModelHelper(name="DIEN", init_params=True)

        if cli_args:
          self.model.net.Proto().type = cli_args.caffe2_net_type
          self.model.net.Proto().num_workers = cli_args.inter_op_workers

        # save arguments
        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.accel_en = accel_en

        return self.create_sequential_forward_ops(id_qs, len_qs, seq_q, hid_q)

    def set_tags(
        self,
        _tag_layer_top_mlp="top",
        _tag_layer_embedding="emb",
        _tag_layer_gru="gru",
        _tag_feature_sparse_in="sparse_in",
        _tag_feature_sparse_out="sparse_out",
        _tag_feature_gru_out="gru_out",
        _tag_dense_output="prob_click",
    ):
        # layer tags
        self.ttop = _tag_layer_top_mlp
        self.temb = _tag_layer_embedding
        self.tgru = _tag_layer_gru

        # sparse feature tags
        self.tsin = _tag_feature_sparse_in
        self.tsout = _tag_feature_sparse_out
        self.tgruout = _tag_feature_gru_out

        # output and target tags
        self.tout = _tag_dense_output

    def parameters(self):
        return self.model

    def create(self, X, S_lengths, S_indices, T, id_qs = None, len_qs=None):
        self.create_input(X, S_lengths, S_indices, T)
        self.create_model(X, S_lengths, S_indices, T)

    def create_input(self, X, S_lengths, S_indices, T):
        for i in range(len(self.emb_l)):
            len_s = self.temb + ":::" + "sls" + str(i) + "_l"
            ind_s = self.temb + ":::" + "sls" + str(i) + "_i"
            self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))
            self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

        sequence_lengths = np.zeros(X.shape[0]).astype(np.int32)
        sequence_lengths += (len(self.ln_emb) - 3)
        self.FeedBlobWrapper( 'seq_lengths', sequence_lengths )

        initial_h_data = np.zeros( (X.shape[0], self.args.hidden_size) ).astype(np.float32)
        self.FeedBlobWrapper( 'initial_h', initial_h_data)

    def create_model(self, X, S_lengths, S_indices, T):
        #setup tril indices for the interactions
        num_fea = len(self.emb_l) + 1

        # create compute graph
        print("Trying to run DIEN for the first time")
        if T is not None:
            # WARNING: RunNetOnce call is needed only if we use brew and ConstantFill.
            # We could use direct calls to self.model functions above to avoid it
            workspace.RunNetOnce(self.model.param_init_net)
            workspace.CreateNet(self.model.net)
        print("Ran DIEN for the first time")


    def run(self, X=None, S_lengths=None, S_indices=None, enable_prof=False):
        # feed input data to blobs
        # TODO: Need to make sequence_lengths an input to the overall model due
        # to batch-size

        if not self.args.queue:
            # sparse features
            for i in range(len(self.emb_l)):
                ind_s = self.temb + ":::" + "sls" + str(i) + "_i"
                self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

                len_s = self.temb + ":::" + "sls" + str(i) + "_l"
                self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))

            batch_size = np.array(S_lengths).shape[1]
            sequence_lengths = np.zeros(np.array(S_lengths).shape[1]).astype(np.int32)
            sequence_lengths += len(self.ln_emb)
            self.FeedBlobWrapper( 'seq_lengths', sequence_lengths )

            initial_h_data = np.zeros( (np.array(S_lengths).shape[1], self.args.hidden_size) ).astype(np.float32)
            self.FeedBlobWrapper( 'initial_h', initial_h_data)

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
    from data_generator.dlrm_data_caffe2 import DLRMDataGenerator

    from utils.utils import cli

    args = cli()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    use_accel = args.use_accel
    if use_accel:
        device_opt = core.DeviceOption(workspace.caffe2_pb2.CUDA, 0)
        naccels = C.num_cuda_devices # 1
        print("Using {} Accel(s)...".format(naccels))
    else:
        device_opt = core.DeviceOption(caffe2_pb2.CPU)
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    dc = DLRMDataGenerator (args)
    if args.data_generation == "dataset":
        print("Error we have disabled this function currently....")
        sys.exit()
        # input and target data
        #(nbatches, lX, lS_l, lS_i, lT,
        # nbatches_test, lX_test, lS_l_test, lS_i_test, lT_test,
        # ln_emb, m_den) = dc.read_dataset(
        #    args.data_set, args.mini_batch_size, args.data_randomize, args.num_batches,
        #    True, args.raw_data_file, args.processed_data_file)
        #ln_bot[0] = m_den
    else:
        # input data
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        if args.data_generation == "random":
            (nbatches, lX, lS_l, lS_i) = dc.generate_input_data()
        elif args.data_generation == "synthetic":
            (nbatches, lX, lS_l, lS_i) = dc.generate_synthetic_input_data(
                args.data_size, args.num_batches, args.mini_batch_size,
                args.round_targets, args.num_indices_per_lookup,
                args.num_indices_per_lookup_fixed, m_den, ln_emb,
                args.data_trace_file, args.data_trace_enable_padding)
        else:
            sys.exit("ERROR: --data-generation="
                     + args.data_generation + " is not supported")

        # target data
        print("Generating output dataset")
        (nbatches, lT) = dc.generate_output_data()

    ### construct the neural network specified above ###
    print("Trying to initialized DIEN")
    with core.DeviceScope(device_opt):
        dien = DIEN_Net( args )
    print("Initialized DIEN Net")

    dien.create(lX[0], lS_l[0], lS_i[0], lT[0])
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
            time_load_end   = dien.run(lX[j], lS_l[j], lS_i[j], args.enable_profiling) # args.enable_profiling
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
