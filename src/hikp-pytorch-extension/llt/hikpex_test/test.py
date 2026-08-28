import importlib
import argparse

def main(args):
    import_file = args.module + "." + args.operator + "_test"
    model = importlib.import_module(import_file)
    test = model.get_test_obj(args.para, args.item, args.iteration)
    test.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--para", type=str, required=True)
    parser.add_argument("--module", type=str, required=True)
    parser.add_argument("--operator", type=str, required=True)
    parser.add_argument("--item", type=str, choices=["function", "performance"], default="function")
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    main(args)