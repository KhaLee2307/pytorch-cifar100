import sys
import random
import logging
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch_pruning as tp

from utils import get_network


def quantization(model):
    """ Convert the model to TorchScript (quantization-aware) """
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def prune_model(model, prune_rate):
    # Importance criteria
    example_inputs = torch.randn(1,3,32,32)
    imp = tp.importance.TaylorImportance()

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 100:
            ignored_layers.append(m) # DO NOT prune the final classifier!
    
    iterative_steps = 5 # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=prune_rate, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):

        # Taylor expansion requires gradients for importance estimation
        if isinstance(imp, tp.importance.TaylorImportance):
            # A dummy loss, please replace it with your loss function and data!
            loss = model(example_inputs).sum() 
            loss.backward() # before pruner.step()

        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        
def compress(opt):

    # setup model
    model = get_network(opt)
    # load pretrain
    model.load_state_dict(torch.load(opt.weights))

    # compression
    logging.info("Starting compressing")
    model_name = "compressed/model"

    # quantization
    if (opt.q):
        quantization(model)
        model_name += "_quantize"

    # pruning
    if (opt.p):
        prune_model(model, opt.p_rate / 100)
        model_name += f"_prune{opt.p_rate}"
    logging.info("Finishing compressing")
    
    # save model
    model.zero_grad()
    torch.save(model.state_dict(), f"{model_name}.pth")


if __name__ == '__main__':
    """ Arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet50", help='net type')
    parser.add_argument('-weights', type=str, default="resnet50.pth", help='the weights file you want to compress')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-q', action='store_true', default=False, help='use quantization or not (only for cpu)')
    parser.add_argument('-p', action='store_true', default=False, help='use pruning or not')
    parser.add_argument('-p_rate', type=int, default=10, help='pruning rate')
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