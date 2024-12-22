import cupy as cp
import time
import os
import argparse
import gc
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stress_test(matrix_size, duration_seconds):
    """
    Performs matrix multiplications on the assigned GPU for a specified duration or until an error occurs.

    Args:
        matrix_size: The size (rows/cols) of the square matrices.
        duration_seconds: How long to run the stress test (in seconds).

    Returns:
        True if completed without CUDA errors, False otherwise.
    """
    device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    
    # Pre-declare to ensure these variables exist even if something fails early
    a = b = c = None
    
    try:
        with cp.cuda.Device(device_id):
            # Initialize matrices on the specified GPU
            a = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)
            b = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)

            logger.info(f"Starting stress test on GPU {device_id} with matrix size {matrix_size}x{matrix_size}...")

            start_time = time.time()
            end_time = start_time + duration_seconds

            iterations = 0
            while time.time() < end_time:
                # Perform matrix multiplication
                c = cp.matmul(a, b)

                # Synchronize
                cp.cuda.Stream.null.synchronize()

                iterations += 1

            elapsed_time = time.time() - start_time
            flops = (2 * matrix_size**3 * iterations) / elapsed_time
            gflops = flops / 1e9

            logger.info(f"GPU {device_id}: Matrix size {matrix_size}x{matrix_size} completed in {elapsed_time:.2f} seconds. Iterations: {iterations}")
            logger.info(f"GPU {device_id}: Approximate Performance: {gflops:.2f} GFLOPs")

            return True

    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.info(f"CUDA Error on GPU {device_id} with matrix size {matrix_size}x{matrix_size}: {e}")
        logger.info("This typically indicates an out-of-memory error on the GPU.")
        return False
    except Exception as e:
        logger.info(f"An unexpected error occurred on GPU {device_id} with matrix size {matrix_size}x{matrix_size}: {e}")
        return False
    finally:
        # Ensure we free GPU memory even if an error occurs
        try:
            del a, b, c
            gc.collect()
        except NameError:
            pass
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def get_gpu_info():
    """Logs information about the CUDA-enabled GPU."""
    try:
        device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

        with cp.cuda.Device(device_id):
            gpu_props = cp.cuda.runtime.getDeviceProperties(device_id)
            logger.info(f"\n--- GPU {device_id} ---")
            logger.info(f"  Name: {gpu_props['name'].decode()}")
            logger.info(f"  Total Memory: {gpu_props['totalGlobalMem'] / (1024**3):.2f} GB")
            logger.info(f"  Compute Capability: {gpu_props['major']}.{gpu_props['minor']}")

    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.info(f"CUDA Error: {e}")
        logger.info("This error often happens if there are no CUDA-capable devices or if the CUDA driver is not properly installed.")


def find_max_matrix_size(initial_size, duration_seconds):
    """
    Finds the maximum matrix size that can be processed without errors on a specific GPU.

    Args:
        initial_size: Starting matrix size for the search.
        duration_seconds: Duration to run each test.

    Returns:
        The maximum matrix size that ran without error, or None if the initial size itself fails.
    """
    device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

    max_size = None
    upper_bound = None
    lower_bound = initial_size

    # Handle initial failure separately
    if not stress_test(initial_size, duration_seconds):
        logger.info(f"Initial matrix size {initial_size}x{initial_size} failed on GPU {device_id}. Try reducing the initial size.")
        return None
    else:
        max_size = initial_size

    # Binary search until we can't find a bigger size that works
    while True:
        if upper_bound is None:
            current_size = int(lower_bound * 2)
        else:
            current_size = (lower_bound + upper_bound) // 2

        if current_size == lower_bound:
            break

        if stress_test(current_size, duration_seconds):
            lower_bound = current_size
            max_size = current_size
        else:
            upper_bound = current_size

    logger.info(f"\nMaximum matrix size without error on GPU {device_id}: {max_size}x{max_size}")
    return max_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA Stress Test using CuPy - Auto Size Finder")
    parser.add_argument("-s", "--initial_size", type=int, default=4096, help="Initial matrix size to start testing.")
    parser.add_argument("--test", action="store_true", help="Search for the maximum matrix size that can be processed without errors.")
    parser.add_argument("-d", "--duration", type=int, default=5, help="Duration of each stress test in seconds.")
    parser.add_argument("-r", "--run_duration", type=int, default=3600*10, help="Duration of the entire stress test in seconds.")
    args = parser.parse_args()

    get_gpu_info()
    if args.test:
        max_size = find_max_matrix_size(args.initial_size, args.duration)
    else:
        max_size = args.initial_size
    if max_size is not None:
        stress_test(max_size, args.run_duration)
    else:
        logger.info("No valid matrix size found. Exiting...")
