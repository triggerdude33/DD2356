#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEED 921
#define NUM_ITER 1000000000

int main(int argc, char *argv[]) {
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand(SEED * rank);

  int slice = NUM_ITER / size;

  int count = 0;
  for (int iter = rank * slice; iter < rank * slice + slice; ++iter) {
    double x = (double)random() / (double)RAND_MAX;
    double y = (double)random() / (double)RAND_MAX;
    double z = sqrt((x * x) + (y * y));

    // Check if point is in unit circle
    if (z <= 1.0) {
      count++;
    }
  }

  int global_count;
  MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // Estimate Pi and display the result
    double pi = ((double)global_count / (double)NUM_ITER) * 4.0;
    printf("The result is %f\n", pi);
  }

  MPI_Finalize();
}
