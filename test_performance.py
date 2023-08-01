import os
import sys
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from conf import settings
from utils import get_network, get_test_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, dataloader):
    
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        for (image, label) in tqdm(dataloader):

            if opt.gpu:
                image = image.cuda()
                label = label.cuda()

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    top1_err = 1 - correct_1 / len(dataloader.dataset)
    top5_err = 1 - correct_5 / len(dataloader.dataset)
    
    return top1_err, top5_err


def measure_performance(opt):
    # load model
    model = torch.load(opt.weights, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # load data
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=opt.b,
    )

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 3
    timings=np.zeros((repetitions,1))

    # GPU-WARM-UP
    top1_err, top5_err = test(model, cifar100_test_loader)

    # MEASURE PERFORMANCE
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            test(model, cifar100_test_loader)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            memory_used = torch.cuda.max_memory_allocated()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time    

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    name_model = os.path.splitext(os.path.basename(opt.weights))[0]
    result = name_model
    result += "\nParameter numbers: {}".format(sum(p.numel() for p in model.parameters()))
    result += f"\nTop 1 error: {top1_err}"
    result += f"\nTop 5 error: {top5_err}"
    result += f"\nMean_syn: {mean_syn}"
    result += f"\nStd_syn: {std_syn}"
    result += f"\nInference time (ms)/image: {mean_syn/len(cifar100_test_loader.dataset)}"
    result += f"\nRuntime memory: {memory_used}"

    # save result
    with open(f"experiment/{name_model}.txt", "w") as file:
        file.write(result)

    # print result
    print(result)


if __name__ == '__main__':
    """ Arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="resnet50", help='net type')
    parser.add_argument('-weights', type=str, default="resnet50.pth", help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=512, help='batch size for dataloader')
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

    measure_performance(opt)