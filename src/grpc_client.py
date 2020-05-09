import grpc
import json
import inference_pb2
import inference_pb2_grpc

server_name = "cloud"
server_port = "50051"
# open a gRPC channel
channel = grpc.insecure_channel(f'{server_name}:{server_port}')

stub = inference_pb2_grpc.ServerStub(channel)


def send_grpc_msg(tensor_data, split_layer):
    data = json.dumps(tensor_data)
    request = inference_pb2.Tensor(tensor=data, start_layer=split_layer)
    print(f'Attempting to perform gRPC call to {server_name}:{server_port}')
    response = stub.Partial(request)
    print(f"gRPC call to {server_name}:{server_port} successful!")
    print(f"RESPONSE RECEIEVED : {response.result}")
