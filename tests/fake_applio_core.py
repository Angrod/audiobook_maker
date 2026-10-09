"""Stand-in for Applio's core.py used by tests/test_rvc_training.py.

Mimics what the app relies on: the commands and flags, `<name> | epoch=N | step=S` lines,
`logs/<name>/<name>_<E>e_<S>s.pth` checkpoints, `logs/<name>/<name>.index`, resuming from
`G_*.pth`, and exiting 0 when training fails (as Applio does).
Environment: FAKE_FAIL=<command> to fail a step, FAKE_EPOCH_SECONDS to slow epochs down.
"""
import argparse
import glob
import os
import sys
import time


def log_call():
    with open("calls.log", "a", encoding="utf-8") as f:
        f.write(" ".join(sys.argv[1:]) + "\n")


def touch(path, content="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    log_call()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("prerequisites")
    p = sub.add_parser("preprocess")
    p.add_argument("--model-name"); p.add_argument("--dataset-path"); p.add_argument("--sample-rate")
    p.add_argument("--cut-preprocess"); p.add_argument("--noise-reduction", action="store_true")
    e = sub.add_parser("extract")
    for flag in ("--model-name", "--f0-method", "--gpu", "--sample-rate", "--embedder-model"):
        e.add_argument(flag)
    t = sub.add_parser("train")
    for flag in ("--model-name", "--vocoder", "--sample-rate", "--gpu", "--index-algorithm"):
        t.add_argument(flag)
    for flag in ("--save-every-epoch", "--total-epoch", "--batch-size"):
        t.add_argument(flag, type=int)
    args = parser.parse_args()
    fail = os.environ.get("FAKE_FAIL") == args.command

    if args.command == "prerequisites":
        for path in ("rvc/models/pretraineds/hifi-gan/f0G40k.pth", "rvc/models/pretraineds/hifi-gan/f0D40k.pth",
                     "rvc/models/predictors/rmvpe.pt", "rvc/models/embedders/contentvec/pytorch_model.bin"):
            touch(path)
        print("Prerequisites installed")
    elif args.command == "preprocess":
        if fail or not os.path.isdir(args.dataset_path):
            print("Preprocess failed")
            sys.exit(1)
        touch(f"logs/{args.model_name}/sliced_audios/0.wav")
        print("Preprocess completed")
    elif args.command == "extract":
        touch(f"logs/{args.model_name}/extracted/0.npy")
        print("Extraction completed")
    elif args.command == "train":
        name, folder = args.model_name, f"logs/{args.model_name}"
        if fail:
            print(f"Training failed for model {name}. Please check the console logs for more details.")
            return  # Applio prints the failure and still exits 0
        start = 0
        for g in glob.glob(f"{folder}/G_*.pth"):
            with open(g, encoding="utf-8") as f:
                start = max(start, int(f.read()))
        delay = float(os.environ.get("FAKE_EPOCH_SECONDS", "0"))
        for epoch in range(start + 1, args.total_epoch + 1):
            step = epoch * 10
            print(f"{name} | epoch={epoch} | step={step} | time=00:00:01 | lowest_value=1.0", flush=True)
            if epoch % args.save_every_epoch == 0 or epoch == args.total_epoch:
                touch(f"{folder}/G_{step}.pth", str(epoch))
                touch(f"{folder}/{name}_{epoch}e_{step}s.pth", f"weights {epoch}")
            time.sleep(delay)
        print(f"Training has been successfully completed with {args.total_epoch} epoch")
        if not os.path.exists(f"{folder}/{name}.index"):
            touch(f"{folder}/{name}.index", "index")
        print(f"Model {name} trained successfully.")


if __name__ == "__main__":
    main()
