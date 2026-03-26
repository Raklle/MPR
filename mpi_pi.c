#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static void usage(int rank) {
    if (rank == 0) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  mpiexec -np <P> ./mpi_pi <total_points>\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "  mpiexec -np 4 ./mpi_pi 10000000\n");
        fflush(stderr);
    }
}

/* Prosty, szybki generator xorshift64* */
static inline uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

/* Losowanie double z [0,1) */
static inline double rand01(uint64_t *state) {
    /* bierzemy 53 bity do double */
    return (xorshift64star(state) >> 11) * (1.0 / 9007199254740992.0);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 2) {
        usage(rank);
        MPI_Finalize();
        return 1;
    }

    char *endptr = NULL;
    unsigned long long total_points = strtoull(argv[1], &endptr, 10);

    if (*argv[1] == '\0' || *endptr != '\0') {
        if (rank == 0) {
            fprintf(stderr, "[ERROR] total_points must be an integer\n");
            fflush(stderr);
        }
        MPI_Finalize();
        return 1;
    }

    if (total_points == 0ULL) {
        if (rank == 0) {
            fprintf(stderr, "[ERROR] total_points must be > 0\n");
            fflush(stderr);
        }
        MPI_Finalize();
        return 1;
    }

    /* Prosty podział pracy */
    unsigned long long local_points = total_points / (unsigned long long)size;

    /* Seed różny dla każdego procesu */
    uint64_t seed = (uint64_t)time(NULL) ^ ((uint64_t)rank * 0x9E3779B97F4A7C15ULL);
    if (seed == 0) seed = 88172645463325252ULL;

    /* Synchronizacja przed pomiarem */
    MPI_Barrier(comm);
    double start = MPI_Wtime();

    unsigned long long local_inside = 0ULL;

    for (unsigned long long i = 0; i < local_points; i++) {
        double x = rand01(&seed);
        double y = rand01(&seed);

        if (x * x + y * y <= 1.0) {
            local_inside++;
        }
    }

    unsigned long long global_inside = 0ULL;

    MPI_Reduce(
        &local_inside,
        &global_inside,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    double end = MPI_Wtime();

    if (rank == 0) {
        unsigned long long used_points = local_points * (unsigned long long)size;
        double pi_est = 4.0 * (double)global_inside / (double)used_points;
        double elapsed = end - start;

        /* CSV-friendly output:
           processes,total_points,used_points,pi,time_s
        */
        printf("%d,%llu,%llu,%.12f,%.6f\n",
               size,
               total_points,
               used_points,
               pi_est,
               elapsed);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}