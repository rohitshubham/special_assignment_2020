import grpc
import json
import gRPC_module.inference_pb2 as inference_pb2
import gRPC_module.inference_pb2_grpc as inference_pb2_grpc

server_name = "cloud"
server_port = "50051"
# open a gRPC channel
channel = grpc.insecure_channel(f'{server_name}:{server_port}')

stub = inference_pb2_grpc.ServerStub(channel)


def send_grpc_msg(tensor_data, split_layer):
    """
    gRPC call to send the partial inference data to cloud server

    Parameters:
    tensor_data (list): Partial inference data
    split_layer (int): (int) : the layer form which to begin inference on cloud
    """
    data = json.dumps(tensor_data)
    request = inference_pb2.Tensor(tensor=data, start_layer=split_layer)
    print(f'Attempting to perform gRPC call to {server_name}:{server_port}')
    response = stub.Partial(request)
    print(f"gRPC call to {server_name}:{server_port} successful!")
    print(f"RESPONSE RECEIEVED : {response.result}")
