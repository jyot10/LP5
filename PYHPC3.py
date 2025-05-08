import random
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Function to find local min
def find_local_min(arr_slice, thread_id):
    print(f"\nThread {thread_id} processing min on slice: {arr_slice}")
    return min(arr_slice)

# Function to find local max
def find_local_max(arr_slice, thread_id):
    print(f"\nThread {thread_id} processing max on slice: {arr_slice}")
    return max(arr_slice)

# Function to calculate sum for average
def compute_partial_sum(arr_slice, thread_id):
    print(f"\nThread {thread_id} processing sum on slice: {arr_slice}")
    return sum(arr_slice)

# Main function
def main():
    n = int(input("Enter the number of elements in the array: "))
    arr = [random.randint(0, 99) for _ in range(n)]

    print("\nArray elements are:", arr)

    num_threads = 4
    chunk_size = (n + num_threads - 1) // num_threads
    slices = [arr[i:i + chunk_size] for i in range(0, n, chunk_size)]

    # Min calculation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        min_results = list(executor.map(find_local_min, slices, range(len(slices))))
    print("\n\nMin value:", min(min_results))

    # Max calculation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        max_results = list(executor.map(find_local_max, slices, range(len(slices))))
    print("\n\nMax value:", max(max_results))

    # Average calculation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        sum_results = list(executor.map(compute_partial_sum, slices, range(len(slices))))
    total_sum = sum(sum_results)
    print("\n\nSum:", total_sum)
    print("Average:", total_sum / n)

if __name__ == "__main__":
    main()
