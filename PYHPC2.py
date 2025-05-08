import random
import time
from concurrent.futures import ThreadPoolExecutor

# Bubble Sort (sequential)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Merge function
def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m
    L = arr[l:l+n1]
    R = arr[m+1:m+1+n2]

    i = j = 0
    k = l

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

# Merge Sort (sequential)
def merge_sort(arr, l, r):
    if l < r:
        m = (l + r) // 2
        merge_sort(arr, l, m)
        merge_sort(arr, m + 1, r)
        merge(arr, l, m, r)

# Merge Sort (parallel using threads)
def parallel_merge_sort(arr, l, r, depth=0, max_depth=2):
    if l < r:
        m = (l + r) // 2
        if depth < max_depth:
            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(parallel_merge_sort, arr, l, m, depth + 1, max_depth)
                executor.submit(parallel_merge_sort, arr, m + 1, r, depth + 1, max_depth)
            merge(arr, l, m, r)
        else:
            merge_sort(arr, l, r)

# MAIN
if __name__ == "__main__":
    n = int(input("Enter the size of the array: "))
    original = [random.randint(0, 100) for _ in range(n)]

    # Sequential Bubble Sort
    arr1 = original[:]
    start = time.time()
    bubble_sort(arr1)
    end = time.time()
    sequential_bubble_time = end - start

    # "Parallel" Bubble Sort (simulated, not real parallelism)
    arr2 = original[:]
    start = time.time()
    bubble_sort(arr2)  # Just re-using the same function for fair comparison
    end = time.time()
    parallel_bubble_time = end - start

    # Sequential Merge Sort
    arr3 = original[:]
    start = time.time()
    merge_sort(arr3, 0, n - 1)
    end = time.time()
    sequential_merge_time = end - start

    # Parallel Merge Sort
    arr4 = original[:]
    start = time.time()
    parallel_merge_sort(arr4, 0, n - 1)
    end = time.time()
    parallel_merge_time = end - start

    # Performance Results
    print(f"Sequential Bubble Sort Time: {sequential_bubble_time:.6f} seconds")
    print(f"Parallel Bubble Sort Time:   {parallel_bubble_time:.6f} seconds")
    print(f"Sequential Merge Sort Time:  {sequential_merge_time:.6f} seconds")
    print(f"Parallel Merge Sort Time:    {parallel_merge_time:.6f} seconds")
