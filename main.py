import time
import traceback

import torch
from torch.nn import Sequential, Linear, CrossEntropyLoss
from torch.optim import SGD, Adam
import json

try:
    import intel_extension_for_pytorch as ipex
    ipex_loaded = True
except ImportError:
    ipex_loaded = False
    pass


def create(device):
    model = Sequential(
        Linear(in_features=100, out_features=10),
        Linear(in_features=10, out_features=10),
        Linear(in_features=10, out_features=10),
        Linear(in_features=10, out_features=10),
        Linear(in_features=10, out_features=1)
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    if ipex_loaded:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)

    criterion = CrossEntropyLoss().to(device)
    return model, optimizer, criterion


def test():
    torch.manual_seed(0)

    print("PyTorch Version:", torch.__version__)
    print("CUDA Version: ", torch.version.cuda)
    device = torch.device("xpu")
    print("PyTorch Device:", device, "\n")

    report = {}

    # Increase the number of rows until it fails
    for len_rows in range(500000, 50000000, 500000):
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

        except RuntimeError:
            traceback.print_exc()
            print(f"Error encountered on row {len_rows}")
            break


if __name__ == '__main__':
    test()
