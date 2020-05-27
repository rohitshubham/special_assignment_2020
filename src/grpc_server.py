import grpc
import gRPC_module.inference_pb2 as inference_pb2
import gRPC_module.inference_pb2_grpc as inference_pb2_grpc
from concurrent import futures
import partial_inference_server
import time
import json


class ServerServicer(inference_pb2_grpc.ServerServicer):
    """
    This class provides the server stub interface for partial inference
    """
    def Partial(self, request, context):
        """
        Parameters:
        request (Tensor) : Tensor object request data coming from edge
        """
        print("Received processing request")
        response = inference_pb2.Result()
        data = json.loads(request.tensor)
        response.result = partial_inference_server.partial_inference(data, request.start_layer)
        print("Returning response")
        return response


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

inference_pb2_grpc.add_ServerServicer_to_server(
    ServerServicer(), server
)

print('Starting server. Listening on port 50051.')
server.add_insecure_port('[::]:50051')
server.start()

# since server.start() will not block,
# a sleep-loop is added to keep alive
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
