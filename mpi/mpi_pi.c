#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int rank, size;
    unsigned long long total_points, local_points;
    unsigned long long local_inside = 0, global_inside = 0;
    unsigned long long i;
    unsigned int seed;
    double x, y;
    double start, end, pi;
    char *endptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpiexec -np <P> ./mpi_pi <total_points>\n");
            fflush(stderr);
        }
        MPI_Finalize();
        return 1;
    }

    total_points = strtoull(argv[1], &endptr, 10);
    if (*endptr != '\0' || total_points == 0) {
        if (rank == 0) {
            fprintf(stderr, "[ERROR] total_points must be a positive integer\n");
            fflush(stderr);
        }
        MPI_Finalize();
        return 1;
    }

    /* Prosty podział pracy */
    local_points = total_points / size;

    /* Inny seed dla każdego procesu */
    seed = (unsigned int)time(NULL) + rank * 12345;

    /* Synchronizacja przed pomiarem */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for (i = 0; i < local_points; i++) {
        x = (double)rand_r(&seed) / (double)RAND_MAX;
        y = (double)rand_r(&seed) / (double)RAND_MAX;

        if (x * x + y * y <= 1.0) {
            local_inside++;
        }
    }

    /* Sumowanie wyników */
    MPI_Reduce(&local_inside, &global_inside, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();

    if (rank == 0) {
        unsigned long long used_points = local_points * size;
        pi = 4.0 * (double)global_inside / (double)used_points;

        /* CSV: processes,total_points,used_points,pi,time_s */
        printf("%d,%llu,%llu,%.12f,%.6f\n",
               size, total_points, used_points, pi, end - start);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}