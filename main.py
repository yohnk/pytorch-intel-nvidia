import os.path
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process

import torch
from torch.nn import Sequential, Linear, CrossEntropyLoss
from torch.optim import SGD, Adam
import json


def create(device):
    model = Sequential(
        Linear(in_features=100, out_features=10),
        Linear(in_features=10, out_features=10),
        Linear(in_features=10, out_features=10),
        Linear(in_features=10, out_features=10),
        Linear(in_features=10, out_features=1)
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss().to(device)
    return model, optimizer, criterion


def run(len_rows, device):
    try:
        import intel_extension_for_pytorch as ipex
        torch.xpu.init()
        torch.xpu.set_device(0)
    except ImportError:
        pass

    if os.path.exists(f"report_{device}.json"):
        with open(f"report_{device}.json", "r") as f:
            report = json.load(f)
    else:
        report = {}

    print(f"Attempting {len_rows} rows")
    try:
        x = torch.rand(len_rows, 100, dtype=torch.float32, device=device)
        y = torch.rand(len_rows, 1, dtype=torch.float32, device=device)
        model, optimizer, criterion = create(device)

        times = []
        for _ in range(100):
            start = time.time()
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            times.append(time.time() - start)

        report[len_rows] = {
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times)
        }

        with open(f"report_{device}.json", "w") as f:
            json.dump(report, f)

    except RuntimeError as e:
        traceback.print_exc()
        print(f"Error encountered on row {len_rows}")
        sys.exit(1)

    sys.exit(0)


def test():
    torch.manual_seed(0)

    print("PyTorch Version:", torch.__version__)
    print("CUDA Version: ", torch.version.cuda)
    device = torch.device("xpu")
    print("PyTorch Device:", device, "\n")

    # Increase the number of rows until it fails
    for len_rows in range(1000000, 50000000, 1000000):
        p = Process(target=run, args=(len_rows, device,))
        p.start()
        p.join()

        try:
            import intel_extension_for_pytorch as ipex
            torch.xpu.empty_cache()
        except:
            traceback.print_exc()

        if p.exitcode != 0:
            break
        else:
            time.sleep(10)


if __name__ == '__main__':
    test()
