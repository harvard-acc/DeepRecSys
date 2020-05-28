
# Packet to log request information from the DeepRecLoadGenerator
from __future__ import absolute_import, division, print_function, unicode_literals


class ServiceRequest(object):
  def __init__( self, batch_id=None,
                      epoch = None,
                      arrival_time=None,
                      batch_size = None,
                      sub_id     = None,
                      total_sub_batches = None,
                      exp_packet = None):

    self.batch_id          = batch_id
    self.batch_size        = batch_size
    self.epoch             = epoch
    self.arrival_time      = arrival_time

    self.total_sub_batches = total_sub_batches
    self.sub_id            = sub_id
    self.exp_packet = exp_packet

  def __str__(self, ):
    string  = "Request[" + (self.epoch, self.batch_id, self.batch_size) + "] ->"
    string += "arrival_time " + self.arrival_time
    return string


# Packet to log output information for the request from the DeepRec inference
# engines
class ServiceResponse(object):
  def __init__( self, consumer_id=None,
                      epoch = None,
                      batch_id = None,
                      batch_size = None,
                      arrival_time=None,
                      process_start_time = None,
                      queue_end_time = None,
                      inference_end_time = None,
                      out_batch_size = None,
                      sub_id = None,
                      total_sub_batches = None,
                      exp_packet = None):

    self.consumer_id = consumer_id
    self.epoch = epoch
    self.batch_id = batch_id
    self.batch_size = batch_size

    self.arrival_time       = arrival_time
    self.queue_start_time   = process_start_time
    self.queue_end_time     = queue_end_time
    self.inference_end_time = inference_end_time

    self.out_batch_size = out_batch_size
    self.total_sub_batches = total_sub_batches
    self.exp_packet = exp_packet
    self.sub_id = sub_id

  def __str__(self, ):
    string  = "Response[" + str((self.epoch, self.batch_id, self.batch_size, self.consumer_id)) + "]"
    string += " ->arrival_time " + str(self.arrival_time * 10.) # multiply by 10 for printing purposes
    string += " ->queue_start_time " + str(self.process_start_time * 10.)
    string += " ->queue_end_time " + str(self.queue_end_time * 10.)
    string += " ->inference_end_time " + str(self.inference_end_time * 10.)

    return string

