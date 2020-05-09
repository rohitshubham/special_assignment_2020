import grpc
import inference_pb2
import inference_pb2_grpc
from concurrent import futures
import partial_inference_server
import time


class ServerServicer(inference_pb2_grpc.ServerServicer):
    def Partial(self, request, context):
        response = inference_pb2.Result()
        response.result = partial_inference_server.partial_inference(request.tensor, request.start_layer)
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
