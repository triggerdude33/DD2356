#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SAMPLES 10
#define ARR_SIZE 10000000

#define MAX_THREADS 128

struct elem_t {
  double sum;
  char padding[120];
};

void generate_random(double *input, size_t size) {
  for (size_t i = 0; i < size; i++) {
    input[i] = rand() / (double)(RAND_MAX);
  }
}

double opt_local_sum(double *x, size_t size) {
  struct elem_t local_sum[MAX_THREADS];
  for (size_t i = 0; i < MAX_THREADS; ++i) {
    local_sum[i].sum = 0.0;
  }
#pragma omp parallel shared(local_sum)
  {
    int id = omp_get_thread_num();
#pragma omp for
    for (size_t i = 0; i < size; i++) {
      local_sum[id].sum += x[i];
    }
  }

  double sum_val = 0.0;
  for (size_t i = 0; i < MAX_THREADS; ++i) {
    sum_val += local_sum[i].sum;
  }

  return sum_val;
}

double omp_local_sum(double *x, size_t size) {
  double local_sum[MAX_THREADS];
  for (size_t i = 0; i < MAX_THREADS; ++i) {
    local_sum[i] = 0.0;
  }
#pragma omp parallel shared(local_sum)
  {
    int id = omp_get_thread_num();
#pragma omp for
    for (size_t i = 0; i < size; i++) {
      local_sum[id] += x[i];
    }
  }

  double sum_val = 0.0;
  for (size_t i = 0; i < MAX_THREADS; ++i) {
    sum_val += local_sum[i];
  }

  return sum_val;
}

double omp_critical_sum(double *x, size_t size) {
  double sum_val = 0.0;

#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
#pragma omp critical
    { sum_val += x[i]; }
  }

  return sum_val;
}

double omp_sum(double *x, size_t size) {
  double sum_val = 0.0;

#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    sum_val += x[i];
  }

  return sum_val;
}

double serial_sum(double *x, size_t size) {
  double sum_val = 0.0;

  for (size_t i = 0; i < size; i++) {
    sum_val += x[i];
  }

  return sum_val;
}

int main() {
  double *arr = malloc(sizeof(double) * ARR_SIZE);
  generate_random(arr, ARR_SIZE);

  double samples[SAMPLES];
  for (int i = 0; i < SAMPLES; ++i) {
    double t0 = omp_get_wtime();
    double sum = opt_local_sum(arr, ARR_SIZE);
    double t1 = omp_get_wtime();
    samples[i] = t1 - t0;
    printf("Sample %i: %f s, %f\n", i, t1 - t0, sum);
  }

  double average = 0.0;
  for (int i = 1; i < SAMPLES; ++i) {
    average += samples[i];
  }
  average /= ((double)(SAMPLES - 1));

  double stddev = 0.0;
  for (int i = 1; i < SAMPLES; ++i) {
    double e = samples[i] - average;
    stddev += e * e;
  }
  stddev /= ((double)(SAMPLES - 2));
  stddev = sqrt(stddev);

  printf("Average: %f, Stddev: %f\n", average, stddev);

  free(arr);
}
