#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10

int compare(const void *a, const void *b) {
  int x = *(const int *)a;
  int y = *(const int *)b;

  return (x > y) - (x < y);
}

int main(int argc, char *argv[]) {
  int rank;
  int size;

  // Start MPI
  MPI_Init(&argc, &argv);

  // Get this process's rank and the total number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // This simple version requires exactly two MPI processes
  if (size != 2) {
    if (rank == 0) {
      printf("This program requires exactly 2 MPI processes.\n");
    }

    MPI_Finalize();
    return 1;
  }

  // Each process is responsible for half of the array
  int local_n = N / size;
  int local_data[5];

  // Different seed for each MPI process
  srand(time(NULL) + rank);

  // Generate this process's five random numbers
  for (int i = 0; i < local_n; i++) {
    local_data[i] = rand() % 100;
  }

  if (rank == 0) {
    int data[N];

    // Copy rank 0's five numbers into the first half
    for (int i = 0; i < local_n; i++) {
      data[i] = local_data[i];
    }

    // Receive rank 1's five numbers into the second half
    MPI_Recv(&data[local_n], local_n, MPI_INT, 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    // Print the complete array
    printf("Original: ");

    for (int i = 0; i < N; i++) {
      printf("%d ", data[i]);
    }

    printf("\n");
  }

  if (rank == 1) {
    // Send rank 1's five numbers to rank 0
    MPI_Send(local_data, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  /* Now both ranks sort their own local arrays */
  qsort(local_data, local_n, sizeof(int), compare);

  if (rank == 1) {
    // Send the sorted half to rank 0
    MPI_Send(local_data, local_n, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    int other_data[5];

    // Receive rank 1's sorted half
    MPI_Recv(other_data, local_n, MPI_INT, 1, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    // Merge the two sorted arrays here
    int sorted[N];

    int i = 0;
    int j = 0;
    int k = 0;

    // Merge while both arrays still have elements
    while (i < local_n && j < local_n) {

      if (local_data[i] < other_data[j]) {
        sorted[k] = local_data[i];
        i++;
      } else {
        sorted[k] = other_data[j];
        j++;
      }

      k++;
    }

    // Copy anything remaining from rank 0's array
    while (i < local_n) {
      sorted[k] = local_data[i];
      i++;
      k++;
    }

    // Copy anything remaining from rank 1's array
    while (j < local_n) {
      sorted[k] = other_data[j];
      j++;
      k++;
    }

    // Print the final sorted array
    printf("Sorted: ");

    for (int i = 0; i < N; i++) {
      printf("%d ", sorted[i]);
    }

    printf("\n");
  }
  // Shut down MPI
  MPI_Finalize();

  return 0;
}
