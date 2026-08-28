import argparse
import torch
import hikp_pytorch_extension._C as kernel

def main(args):
    kernel.fuzz.dt_fuzz_test(args.operator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", type=str, required=True)
    args = parser.parse_args()
    main(args)