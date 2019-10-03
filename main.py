import numpy as np
import json, math, time, os
from hyperopt import *
import gc

BATCH_SIZE = 300

mnist_train = torchvision.datasets.MNIST(
    "./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
)

mnist_test = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)

dl_train = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE, shuffle=False
)
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)


def test(model):
    for i, (features_, labels_) in enumerate(dl_test):
        features, labels = torch.reshape(features_, (10000, 28 * 28)), labels_
        pred = model.forward(features)
        return pred.argmax(dim=1).eq(labels).sum().item() / 10000 * 100


def train(model, epochs=3, height=1):
    stats = []
    for epoch in range(epochs):
        for i, (features_, labels_) in enumerate(dl_train):
            t0 = time.process_time()
            model.begin()
            features, labels = torch.reshape(features_, (BATCH_SIZE, 28 * 28)), labels_
            pred = model.forward(
                features
            )  # typo in https://www.groundai.com/project/gradient-descent-the-ultimate-optimizer/
            loss = F.nll_loss(pred, labels)
            model.zero_grad()
            loss.backward(create_graph=True)
            model.adjust()
            tf = time.process_time()
            data = {
                "time": tf - t0,
                "iter": epoch * len(dl_train) + i,
                "loss": loss.item(),
                "params": {
                    k: v.item()
                    for k, v in model.optimizer.parameters.items()
                    if "." not in k
                },
            }
            stats.append(data)
    return stats


def run(opt, name="out", usr={}, epochs=3, height=1):
    torch.manual_seed(0x42)
    model = MNIST_FullyConnected(28 * 28, 128, 10, opt)
    print("Running...", str(model))
    model.initialize()
    log = train(model, epochs, height)
    acc = test(model)
    out = {"acc": acc, "log": log, "usr": usr}
    with open("log/%s.json" % name, "w+") as f:
        json.dump(out, f, indent=True)
    times = [x["time"] for x in log]
    print("Times (ms):", np.mean(times), "+/-", np.std(times))
    print("Final accuracy:", acc)
    return out


def sgd_experiments():
    run(SGD(0.01), "sgd", epochs=1)
    out = run(SGD(0.01, optimizer=SGD(0.01)), "sgd+sgd", epochs=1)
    alpha = out["log"][-1]["params"]["alpha"]
    print(alpha)
    run(SGD(alpha), "sgd-final", epochs=1)


def adam_experiments():
    run(Adam(), "adam", epochs=1)
    print()
    mo = SGDPerParam(
        0.001, ["alpha", "beta1", "beta2", "log_eps"], optimizer=SGD(0.0001)
    )
    out = run(Adam(optimizer=mo), "adam+sgd", epochs=1)
    p = out["log"][-1]["params"]
    alpha = p["alpha"]
    beta1 = Adam.clamp(torch.tensor(p["beta1"])).item()
    beta2 = Adam.clamp(torch.tensor(p["beta2"])).item()
    log_eps = p["log_eps"]
    print(alpha, beta1, beta2, log_eps)
    print(mo)
    run(
        Adam(alpha=p["alpha"], beta1=beta1, beta2=beta2, log_eps=log_eps),
        "adam+sgd-final",
        epochs=1,
    )
    print()
    out = run(Adam(optimizer=Adam()), "adam2", epochs=1)
    p = out["log"][-1]["params"]
    alpha = p["alpha"]
    beta1 = Adam.clamp(torch.tensor(p["beta1"])).item()
    beta2 = Adam.clamp(torch.tensor(p["beta2"])).item()
    log_eps = p["log_eps"]
    print(alpha, beta1, beta2, log_eps)
    run(
        Adam(alpha=p["alpha"], beta1=beta1, beta2=beta2, log_eps=log_eps),
        "adam2-final",
        epochs=1,
    )
    print()
    mo = SGDPerParam(0.001, ["alpha"], optimizer=SGD(0.0001))
    out = run(AdamBaydin(optimizer=mo), "adambaydin+sgd", epochs=1)
    p = out["log"][-1]["params"]
    alpha = p["alpha"]
    print(alpha)
    print(mo)
    run(Adam(alpha=p["alpha"]), "adambaydin+sgd-final", epochs=1)
    print()
    out = run(AdamBaydin(optimizer=Adam()), "adambaydin2", epochs=1)
    p = out["log"][-1]["params"]
    alpha = p["alpha"]
    print(alpha)
    run(Adam(alpha=p["alpha"]), "adambaydin2-final", epochs=1)


def surface():
    run(SGD(10 ** -3, optimizer=SGD(10 ** -1)), "tst", epochs=1)
    for log_alpha in np.linspace(-3, 2, 10):
        run(SGD(10 ** log_alpha), "sgd@1e%+.2f" % log_alpha, epochs=1)


def make_sgd_stack(height, top):
    if height == 0:
        return SGD(alpha=top)
    return SGD(alpha=top, optimizer=make_sgd_stack(height - 1, top))


def make_adam_stack(height, top=0.0000001):
    if height == 0:
        return Adam(alpha=top)
    return Adam(alpha=top, optimizer=make_adam_stack(height - 1))


def stack_test():
    for top in np.linspace(-7, 3, 20):
        for height in range(6):
            print("height =", height, "to p=", top)
            opt = make_sgd_stack(height, 10 ** top)
            run(
                opt,
                "metasgd3-%d@%+.2f" % (height, top),
                {"height": height, "top": top},
                epochs=1,
                height=height,
            )
            gc.collect()


def perf_test():
    for h in range(51):
        print("height:", h)
        # opt = make_sgd_stack(h, 0.01)
        opt = make_adam_stack(h)
        run(opt, "adamperf-%d" % h, {"height": h}, epochs=1)
        gc.collect()


if __name__ == "__main__":
    try:
        os.mkdir("log")
    except:
        print("log/ exists already")

    surface()
    sgd_experiments()
    adam_experiments()
    stack_test()
    perf_test()
