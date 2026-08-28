"""
    Example:
      python ./test/dnn/llt/scripts/run_all_tests.py --platform KP920B --primitive binary gnorm

    Input Parameters: 
      1. --platform: Testing platform. Supports "KP920B" and "KP920F". Default to "KP920B".
      2. --primitive: Testing primitive. The input format is consistent with benchdnn. Default is "binary".
                      Multiple operators can be validated at once, separated by spaces.

    Test Workflow: 
      1. Build the release version.
      2. Run 3 types of tests in parallel: KuDNN UT, Open-source ctest and Open-source operator-related test cases.
      3. Without waiting for the above to finish, build the ASAN version, outputting to a separate out_asan directory.
      4. Run 2 types of ASAN tests in parallel: Operator fuzz testing and LLT test cases.
      5. Wait for all tests to complete, then return the final result.

    Usage Notes: 
      1. You can run the script from any path. For example, in the KPL_DNN directory:
            python ./test/dnn/llt/scripts/run_all_tests.py --platform KP920B --primitive binary gnorm
      2. Before testing on KP920F, ensure the current code is consistent with the code to be committed.
      3. When testing on KP920F, add the BiSheng ASAN library libclang_rt.asan.so (usually in
         BiShengCompiler/lib/clang/17/lib/aarch64-unknown-linux-gnu/) to the LD_LIBRARY_PATH. Since compilation
         takes a long time, you may bind additional NUMA nodes to speed it up.

"""

import os
import sys
import argparse
import subprocess
import multiprocessing
import re
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Example: Command-line arguments with default values")
    parser.add_argument(
        "--platform", "-p",
        default="KP920B",
        help="Specify the platform (default: KP920B)"
    )
    parser.add_argument(
    "--primitive", "-c",
    nargs='+',                     # Accept one or more primitives
    default=["binary"],            # Default value is binary
    help="Specify one or more primitives (default: binary)"
    )
    return parser.parse_args()


def get_parent_dir(path: str, levels: int = 1) -> str:
    """
    Returns the parent directory of the given path, levels levels up
    """
    parent = path
    for _ in range(levels):
        parent = os.path.dirname(parent)
    return parent


def run_command(cmd: str, log_file: str = "command.log") -> None:
    """
    Runs a command in shell mode silently, writing output to a log file.
    """
    log_path = Path("tests_log")
    log_path.mkdir(parents=True, exist_ok=True)
    with open(f"tests_log/{log_file}", "w") as log:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=log,
            stderr=log
        )

    return result.returncode


def compile_project(root_dir: str, platform: str = "KP920B", asan: str = "off") -> None:
    compiler = "gnu"
    output_dir = "out"
    version = "release"
    enable_hbm = "off"
    if platform == "KP920F":
        compiler = "clang"
        enable_hbm = "on"
    if asan == "on":
        output_dir = "out_asan"
        version = "ASAN"
    print(f"\033[94m[RUN]\033[0m Building {platform} {version} version")
    result = run_command(f"sh build/build.sh --target_platform={platform} --enable_asan={asan} --enable_hbm={enable_hbm} --compiler={compiler} --output_dir={root_dir}/{output_dir}", "build.log")
    if result != 0:
        print(f"\033[91m[FAILED]\033[0m Failed to build {platform} {version} version, see tests_log/build.log for details")
        sys.exit(1)

    print(f"\033[92m[PASSED]\033[0m Successfully built {platform} {version} version")

def check_allowed_failures(log_path: str) -> bool:
    allowed_failures = {
        "test_shuffle",
        "test_graph_unit_dnnl_conv_usm_cpu",
        "test_graph_unit_dnnl_large_partition_usm_cpu",
    }

    found_failures = set()

    pattern = re.compile(r"^\s*(\d+)\s*-\s*(\S+)\s*\((.+)\)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue

            test_id, test_name, status = m.groups()
            status = status.strip()
            if "Failed" in status or "Subprocess aborted" in status or "SEGFAULT" in status:
                found_failures.add(test_name)

    unexpected = found_failures - allowed_failures
    if unexpected:
        print(f"\033[91m[FAILED]\033[0m Unexpected ctest failures found: {unexpected}", file=sys.stderr)
        return False

    print(f"Only allowed ctest failures found in log: {found_failures & allowed_failures}")
    return True

def test_ut(root_dir: str, env:str = "default_env"):
    if '=' in env:
        key, value = env.split('=', 1)
        os.environ[key] = value
    print(f"\033[94m[RUN]\033[0m Starting UT tests with {env}")
    result = run_command(f"sh out/test/dnn/llt/scripts/test_dnn.sh --test_type=ut --output_dir={root_dir}/out", f"ut_{env.replace('=', '_')}.log")
    if result != 0:
        print(f"\033[91m[FAILED]\033[0m UT tests failed with {env} see tests_log/ut_{env.replace('=', '_')}.log for details")
    else:
        print(f"\033[92m[PASSED]\033[0m UT tests passed with {env}")

def test_ctest(root_dir: str, env:str = "default_env"):
    if '=' in env:
        key, value = env.split('=', 1)
        os.environ[key] = value
    print(f"\033[94m[RUN]\033[0m Starting open-source ctest with {env}")
    result = run_command(f"cd ./out/test/dnn/oneDNN-3.4/build/ && OMP_NUM_THREADS=36 ctest", f"ctest_{env.replace('=', '_')}.log")
    if not check_allowed_failures(f"./tests_log/ctest_{env.replace('=', '_')}.log"):
        print(f"\033[91m[FAILED]\033[0m Open-source ctest failed with {env} see tests_log/ctest_{env.replace('=', '_')}.log for details")
    else:
        print(f"\033[92m[PASSED]\033[0m Open-source ctest passed with {env}")

def test_opensorce(root_dir: str, primitive: str, env:str = "default_env"):
    if '=' in env:
        key, value = env.split('=', 1)
        os.environ[key] = value
    print(f"\033[94m[RUN]\033[0m Starting open-source test cases for {primitive} with {env}")
    filenames = []
    success_flag = True
    # Comment out skip and reset in open-source test cases to avoid running them
    for name in os.listdir(f"{root_dir}/out/test/dnn/oneDNN-3.4/build/tests/benchdnn/inputs/{primitive}"):
        fullpath = os.path.join(f"{root_dir}/out/test/dnn/oneDNN-3.4/build/tests/benchdnn/inputs/{primitive}", name)
        if os.path.isfile(fullpath):
            with open(fullpath, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.replace('--skip-impl', '# --skip-impl')
            content = content.replace('--reset', '# --reset')
            with open(fullpath, 'w', encoding='utf-8') as f:
                f.write(content)

    for name in os.listdir(f"{root_dir}/out/test/dnn/oneDNN-3.4/build/tests/benchdnn/inputs/{primitive}"):
        fullpath = os.path.join(f"{root_dir}/out/test/dnn/oneDNN-3.4/build/tests/benchdnn/inputs/{primitive}", name)
        if os.path.isfile(fullpath):
            if "shape" in name: continue # Shape test cases are included in other tests, skip to avoid duplication
            if "perf" in name: continue # Performance test cases take too long, skip functional verification
            if "gpt" in name: continue # GPT test cases take too long, skip functional verification
            if "bert" in name: continue # BERT test cases take too long, skip functional verification
            if "test" in name and ("all" not in name or "all_" in name) : continue # test_xx_all test cases include all test cases, skip others
            cmd = f"ENABGLE_POINTWISE_TRH=off OMP_NUM_THREADS=36 {root_dir}/out/test/dnn/oneDNN-3.4/build/tests/benchdnn/benchdnn -v5 --skip-impl=acl,gemm,any,ref,uni,jit: --{primitive} --batch=" + fullpath
            log_name = f"{name}_{env.replace('=', '_')}"
            result = run_command(f"{cmd}", f"{log_name}.log")
            with open(f"tests_log/{log_name}.log", 'r', encoding='utf-8') as f:
                for line in f:
                    if 'FAILED' in line or result < 0 :
                        print(f"\033[91m[FAILED]\033[0m Open-source test case failed with {env} see tests_log/{log_name}.log for details")
                        success_flag = False
                        break
    if success_flag: print(f"\033[92m[PASSED]\033[0m Open-source test cases for {primitive} passed with {env}")

def test_fuzz(root_dir: str, primitive: str, env:str = "default_env"):
    if '=' in env:
        key, value = env.split('=', 1)
        os.environ[key] = value
    print(f"\033[94m[RUN]\033[0m Starting {primitive} fuzz testing with {env}")
    if primitive == "ip": primitive= "inner_product"
    if primitive == "conv": primitive= "convolution"
    if primitive == "deconv": primitive= "deconvolution"
    result = run_command(f"OMP_NUM_THREADS=36 ./out_asan/test/dnn/llt/fuzz/{primitive}_fuzz", f"{primitive}_fuzz_{env.replace('=', '_')}.log")
    if result != 0:
        print(f"\033[91m[FAILED]\033[0m {primitive} fuzz test failed with {env} see tests_log/{primitive}_fuzz_{env.replace('=', '_')}.log for details")
    else:
        print(f"\033[92m[PASSED]\033[0m {primitive} fuzz test passed with {env}")

def test_llt(root_dir: str, primitive: str, env:str = "default_env"):
    if '=' in env:
        key, value = env.split('=', 1)
        os.environ[key] = value
    print(f"\033[94m[RUN]\033[0m Starting {primitive} LLT + ASAN test cases with {env}")
    filenames = []
    primitive_file_path = primitive
    if primitive == "conv": primitive_file_path= "convolution"
    if primitive == "deconv": primitive_file_path= "deconvolution"
    for name in os.listdir(f"{root_dir}/out_asan/test/dnn/llt/inputs/{primitive_file_path}"):
        fullpath = os.path.join(f"{root_dir}/out_asan/test/dnn/llt/inputs/{primitive_file_path}", name)
        if os.path.isfile(fullpath):
            cmd = f"OMP_NUM_THREADS=36 {root_dir}/out_asan/test/dnn/oneDNN-3.4/build/tests/benchdnn/benchdnn -v5 --{primitive} --batch=" + fullpath
            result = run_command(f"{cmd}", f"{name}.log")
            if result != 0:
                print(f"\033[91m[FAILED]\033[0m {primitive} LLT + ASAN test case failed with {env} see tests_log/{name}.log for details")
                return
    print(f"\033[92m[PASSED]\033[0m {primitive} LLT + ASAN test cases passed with {env}")

def main():
    start = time.perf_counter()
    args = parse_args()
    platform = args.platform
    primitive = args.primitive
    this_file = os.path.abspath(__file__)
    root_dir = get_parent_dir(this_file, levels=5)

    print(f"----Starting {platform} platform {primitive} operator tests----")

    if not os.path.isdir(root_dir):
        print(f"KPL_DNN directory does not exist ⇒ {root_dir}", file=sys.stderr)
        sys.exit(1)

    # Switch to the KPL_DNN directory
    os.chdir(root_dir)

    compile_project(root_dir, platform, asan = "off")
    release_test_processes = []
    release_test_processes.append(multiprocessing.Process(target=test_ut, args=(root_dir,)))
    release_test_processes.append(multiprocessing.Process(target=test_ctest, args=(root_dir,)))
    if "matmul" in primitive or "ip" in primitive: release_test_processes.append(multiprocessing.Process(target=test_ctest, args=(root_dir, "KUDNN_FORCE_NEON=1")))
    for prim in primitive:
        release_test_processes.append(multiprocessing.Process(target=test_opensorce, args=(root_dir, prim)))
        if prim == "matmul" or prim == "ip":
            release_test_processes.append(multiprocessing.Process(target=test_opensorce, args=(root_dir, prim, "KUDNN_FORCE_NEON=1")))

    for process in release_test_processes:
        process.start()

    time.sleep(1)
    compile_project(root_dir, platform, asan = "on")


    asan_test_processes = []
    
    for prim in primitive:
        asan_test_processes.append(multiprocessing.Process(target=test_fuzz, args=(root_dir, prim)))
        asan_test_processes.append(multiprocessing.Process(target=test_llt, args=(root_dir, prim)))
        if prim == "matmul" or prim == "ip":
            asan_test_processes.append(multiprocessing.Process(target=test_fuzz, args=(root_dir, prim, "KUDNN_FORCE_NEON=1")))
            asan_test_processes.append(multiprocessing.Process(target=test_llt, args=(root_dir, prim, "KUDNN_FORCE_NEON=1")))

    for process in asan_test_processes:
        process.start()

    for process in asan_test_processes:
        process.join()

    for process in release_test_processes:
        process.join()

    end = time.perf_counter()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"----Test completed, elapsed time: {int(minutes)}m{seconds:.2f}s----")

if __name__ == "__main__":
    main()