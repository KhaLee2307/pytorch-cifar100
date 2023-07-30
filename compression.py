import sys
import logging
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from utils import get_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantization(model):
    """ Convert the model to TorchScript (quantization-aware) """
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    return quantized_model

def prune_model(opt, model):
    """ Convert the model to NumPy for ease of handling """
    model_np = model.state_dict()
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = model_np[name].cpu().numpy()
            threshold = np.percentile(np.abs(weights), opt.p_rate)
            mask = np.abs(weights) > threshold
            weights[~mask] = 0
            model_np[name] = torch.from_numpy(weights).to(device)
    model.load_state_dict(model_np)

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

    opt = parser.parse_args()

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