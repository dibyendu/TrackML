from __future__ import absolute_import, division, print_function

from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt

NUM_LAYERS = 2    # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.

def make_mlp_model():
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
      snt.LayerNorm()
  ])


class MLPGraphIndependent(snt.AbstractModule):
  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=None,
          node_model_fn=make_mlp_model,
          global_model_fn=make_mlp_model)

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.DeepSets(make_mlp_model,
                                       make_mlp_model)

  def _build(self, inputs):
    return self._network(inputs)


class EncodeProcessDecode(snt.AbstractModule):
  """
                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent()
    self._core = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    with self._enter_variable_scope():
      self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn)

  def _build(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops