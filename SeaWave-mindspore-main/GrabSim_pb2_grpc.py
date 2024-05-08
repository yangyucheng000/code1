# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import GrabSim_pb2 as GrabSim__pb2


class GrabSimStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.AcquireTypes = channel.unary_unary(
        '/GrabSim.GrabSim/AcquireTypes',
        request_serializer=GrabSim__pb2.NUL.SerializeToString,
        response_deserializer=GrabSim__pb2.TypeList.FromString,
        )
    self.Init = channel.unary_unary(
        '/GrabSim.GrabSim/Init',
        request_serializer=GrabSim__pb2.Count.SerializeToString,
        response_deserializer=GrabSim__pb2.World.FromString,
        )
    self.ObserveAll = channel.unary_unary(
        '/GrabSim.GrabSim/ObserveAll',
        request_serializer=GrabSim__pb2.NUL.SerializeToString,
        response_deserializer=GrabSim__pb2.World.FromString,
        )
    self.Reset = channel.unary_unary(
        '/GrabSim.GrabSim/Reset',
        request_serializer=GrabSim__pb2.ResetParams.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.MakeObjects = channel.unary_unary(
        '/GrabSim.GrabSim/MakeObjects',
        request_serializer=GrabSim__pb2.ObjectList.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.RemoveObjects = channel.unary_unary(
        '/GrabSim.GrabSim/RemoveObjects',
        request_serializer=GrabSim__pb2.RemoveList.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.CleanObjects = channel.unary_unary(
        '/GrabSim.GrabSim/CleanObjects',
        request_serializer=GrabSim__pb2.SceneID.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.MakeAnchors = channel.unary_unary(
        '/GrabSim.GrabSim/MakeAnchors',
        request_serializer=GrabSim__pb2.AnchorList.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.Observe = channel.unary_unary(
        '/GrabSim.GrabSim/Observe',
        request_serializer=GrabSim__pb2.SceneID.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.Do = channel.unary_unary(
        '/GrabSim.GrabSim/Do',
        request_serializer=GrabSim__pb2.Action.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.SetLidar = channel.unary_unary(
        '/GrabSim.GrabSim/SetLidar',
        request_serializer=GrabSim__pb2.LidarParams.SerializeToString,
        response_deserializer=GrabSim__pb2.LidarParams.FromString,
        )
    self.MoveHand = channel.unary_unary(
        '/GrabSim.GrabSim/MoveHand',
        request_serializer=GrabSim__pb2.HandTarget.SerializeToString,
        response_deserializer=GrabSim__pb2.ArmSequence.FromString,
        )
    self.Capture = channel.unary_unary(
        '/GrabSim.GrabSim/Capture',
        request_serializer=GrabSim__pb2.CameraList.SerializeToString,
        response_deserializer=GrabSim__pb2.CameraData.FromString,
        )
    self.MakeObstacles = channel.unary_unary(
        '/GrabSim.GrabSim/MakeObstacles',
        request_serializer=GrabSim__pb2.ObstacleList.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.SetMovement = channel.unary_unary(
        '/GrabSim.GrabSim/SetMovement',
        request_serializer=GrabSim__pb2.MovementList.SerializeToString,
        response_deserializer=GrabSim__pb2.Scene.FromString,
        )
    self.GetAction = channel.unary_unary(
        '/GrabSim.GrabSim/GetAction',
        request_serializer=GrabSim__pb2.Scene.SerializeToString,
        response_deserializer=GrabSim__pb2.ActionList.FromString,
        )


class GrabSimServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def AcquireTypes(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Init(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ObserveAll(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Reset(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def MakeObjects(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def RemoveObjects(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CleanObjects(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def MakeAnchors(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Observe(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Do(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SetLidar(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def MoveHand(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Capture(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def MakeObstacles(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SetMovement(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetAction(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_GrabSimServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'AcquireTypes': grpc.unary_unary_rpc_method_handler(
          servicer.AcquireTypes,
          request_deserializer=GrabSim__pb2.NUL.FromString,
          response_serializer=GrabSim__pb2.TypeList.SerializeToString,
      ),
      'Init': grpc.unary_unary_rpc_method_handler(
          servicer.Init,
          request_deserializer=GrabSim__pb2.Count.FromString,
          response_serializer=GrabSim__pb2.World.SerializeToString,
      ),
      'ObserveAll': grpc.unary_unary_rpc_method_handler(
          servicer.ObserveAll,
          request_deserializer=GrabSim__pb2.NUL.FromString,
          response_serializer=GrabSim__pb2.World.SerializeToString,
      ),
      'Reset': grpc.unary_unary_rpc_method_handler(
          servicer.Reset,
          request_deserializer=GrabSim__pb2.ResetParams.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'MakeObjects': grpc.unary_unary_rpc_method_handler(
          servicer.MakeObjects,
          request_deserializer=GrabSim__pb2.ObjectList.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'RemoveObjects': grpc.unary_unary_rpc_method_handler(
          servicer.RemoveObjects,
          request_deserializer=GrabSim__pb2.RemoveList.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'CleanObjects': grpc.unary_unary_rpc_method_handler(
          servicer.CleanObjects,
          request_deserializer=GrabSim__pb2.SceneID.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'MakeAnchors': grpc.unary_unary_rpc_method_handler(
          servicer.MakeAnchors,
          request_deserializer=GrabSim__pb2.AnchorList.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'Observe': grpc.unary_unary_rpc_method_handler(
          servicer.Observe,
          request_deserializer=GrabSim__pb2.SceneID.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'Do': grpc.unary_unary_rpc_method_handler(
          servicer.Do,
          request_deserializer=GrabSim__pb2.Action.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'SetLidar': grpc.unary_unary_rpc_method_handler(
          servicer.SetLidar,
          request_deserializer=GrabSim__pb2.LidarParams.FromString,
          response_serializer=GrabSim__pb2.LidarParams.SerializeToString,
      ),
      'MoveHand': grpc.unary_unary_rpc_method_handler(
          servicer.MoveHand,
          request_deserializer=GrabSim__pb2.HandTarget.FromString,
          response_serializer=GrabSim__pb2.ArmSequence.SerializeToString,
      ),
      'Capture': grpc.unary_unary_rpc_method_handler(
          servicer.Capture,
          request_deserializer=GrabSim__pb2.CameraList.FromString,
          response_serializer=GrabSim__pb2.CameraData.SerializeToString,
      ),
      'MakeObstacles': grpc.unary_unary_rpc_method_handler(
          servicer.MakeObstacles,
          request_deserializer=GrabSim__pb2.ObstacleList.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'SetMovement': grpc.unary_unary_rpc_method_handler(
          servicer.SetMovement,
          request_deserializer=GrabSim__pb2.MovementList.FromString,
          response_serializer=GrabSim__pb2.Scene.SerializeToString,
      ),
      'GetAction': grpc.unary_unary_rpc_method_handler(
          servicer.GetAction,
          request_deserializer=GrabSim__pb2.Scene.FromString,
          response_serializer=GrabSim__pb2.ActionList.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'GrabSim.GrabSim', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))