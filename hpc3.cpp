#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdlib>
#include <climits> // for INT_MAX and INT_MIN

using namespace std;

void minVal(int *arr, int n)
{
    int min_val = INT_MAX;
#pragma omp parallel
    {
        int local_min = INT_MAX;
#pragma omp for nowait
        for (int i = 0; i < n; i++)
        {
            cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
            if (arr[i] < local_min)
                local_min = arr[i];
        }
#pragma omp critical
        {
            if (local_min < min_val)
                min_val = local_min;
        }
    }
    cout << "\n\nmin_val = " << min_val << endl;
}

void maxVal(int *arr, int n)
{
    int max_val = INT_MIN;
#pragma omp parallel
    {
        int local_max = INT_MIN;
#pragma omp for nowait
        for (int i = 0; i < n; i++)
        {
            cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
            if (arr[i] > local_max)
                local_max = arr[i];
        }
#pragma omp critical
        {
            if (local_max > max_val)
                max_val = local_max;
        }
    }
    cout << "\n\nmax_val = " << max_val << endl;
}

void avgVal(int *arr, int n)
{
    long long sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
        cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
    }
    float avg = static_cast<float>(sum) / n;
    cout << "\n\nSum = " << sum << endl;
    cout << "\nAverage = " << avg << endl;
}

int main()
{
    omp_set_num_threads(4);
    int n;

    cout << "Enter the number of elements in the array: ";
    cin >> n;
    int *arr = new int[n];

    srand(time(0));
    for (int i = 0; i < n; ++i)
    {
        arr[i] = rand() % 100;
    }

    cout << "\nArray elements are: ";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << ",";
    }

    minVal(arr, n);
    maxVal(arr, n);
    avgVal(arr, n);

    delete[] arr;
    return 0;
}
//output   ./filename.exe
//output   ./filename.exe
//g++ filename.cpp -o filename.exe -fopenmp