#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <omp.h>

static size_t N         = 100000000ULL;
static size_t N_BUCKETS = 10000ULL;

constexpr double MIN_VAL = 0.0;
constexpr double MAX_VAL = 1.0;


// ===================== LOSOWANIE =====================
void fill_random(std::vector<double>& arr,
                 double min_val,
                 double max_val) {

    #pragma omp parallel
    {
        std::random_device rd;

        std::seed_seq seeds{
            rd(), rd(), rd(), rd(),
            static_cast<unsigned>(omp_get_thread_num())
        };

        std::mt19937_64 gen(seeds);

        std::uniform_real_distribution<double> dist(min_val, max_val);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < arr.size(); ++i) {
            arr[i] = dist(gen);
        }
    }
}


// ===================== ALGORYTM 1 =====================
// Każdy wątek:
// 1. czyta CAŁĄ tablicę,
// 2. posiada WŁASNE kubełki,
// 3. zapisuje tylko do swoich kubełków.
void bucket_sort_algo1(
        const std::vector<double>& arr,
        std::vector<std::vector<std::vector<double>>>& thread_buckets,
        double min_val,
        double max_val) {

    const int T = omp_get_max_threads();
    const size_t nb = thread_buckets[0].size();

    const double range = max_val - min_val;
    const double bucket_range = range / nb;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // poprawny podział kubełków
        size_t start_bucket = tid * nb / T;
        size_t end_bucket   = (tid + 1) * nb / T;

        auto& my_buckets = thread_buckets[tid];

        // każdy wątek czyta CAŁĄ tablicę
        for (size_t i = 0; i < arr.size(); ++i) {

            double val = arr[i];

            size_t idx =
                static_cast<size_t>((val - min_val) / bucket_range);

            if (idx >= nb)
                idx = nb - 1;

            // zapis tylko do własnych kubełków
            if (idx >= start_bucket && idx < end_bucket) {
                my_buckets[idx].push_back(val);
            }
        }
    }
}


// ===================== SORTOWANIE =====================
// Każdy wątek sortuje własne kubełki.
void sort_buckets(
        std::vector<std::vector<std::vector<double>>>& thread_buckets) {

    const int T = thread_buckets.size();

    #pragma omp parallel for schedule(static)
    for (int tid = 0; tid < T; ++tid) {

        for (size_t b = 0; b < thread_buckets[tid].size(); ++b) {

            std::sort(
                thread_buckets[tid][b].begin(),
                thread_buckets[tid][b].end()
            );
        }
    }
}


// ===================== ZAPIS =====================
// Każdy wątek zapisuje do INNEGO fragmentu tablicy.
void write_back_parallel(
        std::vector<double>& arr,
        const std::vector<std::vector<std::vector<double>>>& thread_buckets) {

    const int T = thread_buckets.size();
    const size_t nb = thread_buckets[0].size();

    // liczba elementów w każdym kubełku
    std::vector<size_t> bucket_sizes(nb, 0);

    for (int tid = 0; tid < T; ++tid) {
        for (size_t b = 0; b < nb; ++b) {
            bucket_sizes[b] += thread_buckets[tid][b].size();
        }
    }

    // prefix sum -> pozycja startowa kubełka
    std::vector<size_t> bucket_offsets(nb, 0);

    for (size_t b = 1; b < nb; ++b) {
        bucket_offsets[b] =
            bucket_offsets[b - 1] + bucket_sizes[b - 1];
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        size_t start_bucket = tid * nb / T;
        size_t end_bucket   = (tid + 1) * nb / T;

        size_t pos = bucket_offsets[start_bucket];

        for (size_t b = start_bucket; b < end_bucket; ++b) {

            for (double v : thread_buckets[tid][b]) {
                arr[pos++] = v;
            }
        }
    }
}


// ===================== POMIAR CZASU =====================
struct PhaseTimes {
    double t_a = 0;
    double t_b = 0;
    double t_c = 0;
    double t_d = 0;
    double t_e = 0;
};


// ===================== RUN =====================
PhaseTimes run(std::vector<double>& arr,
               size_t nbuckets) {

    PhaseTimes t;

    int T = omp_get_max_threads();

    // każdy wątek ma własny zestaw kubełków
    std::vector<std::vector<std::vector<double>>> thread_buckets(
        T,
        std::vector<std::vector<double>>(nbuckets)
    );

    // (a) losowanie
    double ta0 = omp_get_wtime();

    fill_random(arr, MIN_VAL, MAX_VAL);

    t.t_a = omp_get_wtime() - ta0;

    double te0 = omp_get_wtime();

    // (b) rozdział
    double tb0 = omp_get_wtime();

    bucket_sort_algo1(
        arr,
        thread_buckets,
        MIN_VAL,
        MAX_VAL
    );

    t.t_b = omp_get_wtime() - tb0;

    // (c) sortowanie
    double tc0 = omp_get_wtime();

    sort_buckets(thread_buckets);

    t.t_c = omp_get_wtime() - tc0;

    // (d) zapis
    double td0 = omp_get_wtime();

    write_back_parallel(arr, thread_buckets);

    t.t_d = omp_get_wtime() - td0;

    // (e) całość
    t.t_e = omp_get_wtime() - te0;

    return t;
}


// ===================== MAIN =====================
int main(int argc, char** argv) {

    if (argc >= 2)
        N = std::stoull(argv[1]);

    if (argc >= 3)
        N_BUCKETS = std::stoull(argv[2]);

    int nthreads = omp_get_max_threads();

    std::cerr << "=== ALG1 (FULL SCAN + PRIVATE BUCKETS) ===\n";

    std::cerr << "N = "
              << N
              << " buckets = "
              << N_BUCKETS
              << " threads = "
              << nthreads
              << "\n";

    std::vector<double> arr(N);

    PhaseTimes t = run(arr, N_BUCKETS);

    std::cerr << std::fixed << std::setprecision(4);

    std::cerr << "(a) losowanie:   "
              << t.t_a
              << " s\n";

    std::cerr << "(b) rozdzial:    "
              << t.t_b
              << " s\n";

    std::cerr << "(c) sortowanie:  "
              << t.t_c
              << " s\n";

    std::cerr << "(d) zapis:       "
              << t.t_d
              << " s\n";

    std::cerr << "(e) calosc:      "
              << t.t_e
              << " s\n";

    std::cout << "RESULT,"
              << nthreads << ","
              << N << ","
              << N_BUCKETS << ","
              << t.t_a << ","
              << t.t_b << ","
              << t.t_c << ","
              << t.t_d << ","
              << t.t_e << "\n";

    return 0;
}