import sys
import random
import logging
import argparse
import numpy as np

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune


from utils import get_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantization(model):
    """ Convert the model to TorchScript (quantization-aware) """
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    return quantized_model

def prune_model(opt, model):
    # Convert the model to CPU to perform pruning
    model.cpu()

   # Identify the convolutional layers (2D Conv) in the model
    conv_layers = [module for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]

    # Calculate the number of channels to prune based on the pruning rate
    total_channels = sum([module.weight.data.shape[0] for module in conv_layers])
    channels_to_prune = int(total_channels * opt.p_rate)

    # Ensure that the number of channels to prune is not greater than the total number of channels
    channels_to_prune = min(channels_to_prune, total_channels)

    # Sort the convolutional layers based on their L1-norms of weights (ascending order)
    sorted_conv_layers = sorted(conv_layers, key=lambda x: torch.norm(x.weight.data, 1))

    # Prune the least important channels
    for i in range(channels_to_prune):
        # Calculate the number of channels to prune for this layer
        channels_to_prune_layer = min(sorted_conv_layers[i].weight.data.shape[0], channels_to_prune)

        # Prune the channels for this layer
        prune.l1_unstructured(sorted_conv_layers[i], name='weight', amount=channels_to_prune_layer)

        # Update the remaining channels to prune
        channels_to_prune -= channels_to_prune_layer

        if channels_to_prune == 0:
            break

    # Remove the pruning re-parametrization buffers
    for module in conv_layers:
        prune.remove(module, 'weight')

    return model

def compress(opt):

    # setup model
    model = get_network(opt)
    # load pretrain
    model.load_state_dict(torch.load(opt.weights))

    model_name = "compression/resnet50"

    logging.info("Starting compressing")
    # quantization
    if (opt.q):
        model = quantization(model)
        model_name += "_quantize"

    # pruning
    if (opt.p):
        model = prune_model(opt, model)
        model_name += f"_prune{opt.p_rate}"
    logging.info("Finishing compressing")

    torch.save(model.state_dict(), f"{model_name}.pth")


if __name__ == '__main__':
    """ Arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet50", help='net type')
    parser.add_argument('-weights', type=str, default="resnet50.pth", help='the weights file you want to compress')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-q', action='store_true', default=False, help='use quantization or not')
    parser.add_argument('-p', action='store_true', default=False, help='use pruning or not')
    parser.add_argument('-p_rate', type=int, default=1, help='pruning rate')
    parser.add_argument(
        "--manual_seed", type=int, default=111, help="for random seed setting"
    )

    opt = parser.parse_args()

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # It fasten training.
    cudnn.deterministic = True

    if sys.platform == "win32":
        opt.workers = 0

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience

    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )

    # Compression
    compress(opt)