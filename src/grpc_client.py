import grpc

import inference_pb2
import inference_pb2_grpc

# open a gRPC channel
channel = grpc.insecure_channel('cloud:50051')

stub = inference_pb2_grpc.ServerStub(channel)


def send_grpc_msg(out, split_layer):
    request = inference_pb2.Tensor(tensor=out, start_layer=split_layer)

    response = stub.Partial(request)

    print(response.result)
