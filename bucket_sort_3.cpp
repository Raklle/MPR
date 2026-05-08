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

        std::uniform_real_distribution<double>
            dist(min_val, max_val);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < arr.size(); ++i) {
            arr[i] = dist(gen);
        }
    }
}


// ===================== ALGORYTM 3 =====================
// Każdy wątek:
// 1. czyta WYŁĄCZNIE swój fragment tablicy,
// 2. posiada lokalne kubełki dla CAŁEGO zakresu,
// 3. zapisuje tylko do swoich kubełków.
void bucket_distribute_algo3(
        const std::vector<double>& arr,
        std::vector<std::vector<std::vector<double>>>& local,
        double min_val,
        double max_val,
        size_t nbuckets) {

    int T = omp_get_max_threads();

    size_t n = arr.size();

    double range = max_val - min_val;
    double bucket_range = range / nbuckets;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        size_t start = tid * n / T;
        size_t end   = (tid + 1) * n / T;

        // każdy wątek czyta WYŁĄCZNIE swój fragment
        for (size_t i = start; i < end; ++i) {

            double val = arr[i];

            size_t b =
                static_cast<size_t>(
                    (val - min_val) / bucket_range
                );

            if (b >= nbuckets)
                b = nbuckets - 1;

            // lokalne kubełki
            local[tid][b].push_back(val);
        }
    }
}


// ===================== MERGE + SORT =====================
// Każdy bucket:
// 1. scala kubełki lokalne wszystkich wątków,
// 2. sortuje wynik.
void merge_and_sort(
        const std::vector<std::vector<std::vector<double>>>& local,
        std::vector<std::vector<double>>& merged,
        size_t nbuckets) {

    int T = omp_get_max_threads();

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t b = 0; b < nbuckets; ++b) {

        // policz rozmiar docelowy
        size_t total_size = 0;

        for (int t = 0; t < T; ++t) {
            total_size += local[t][b].size();
        }

        merged[b].reserve(total_size);

        // merge / konkatenacja
        for (int t = 0; t < T; ++t) {

            merged[b].insert(
                merged[b].end(),
                local[t][b].begin(),
                local[t][b].end()
            );
        }

        // sortowanie kubełka
        std::sort(
            merged[b].begin(),
            merged[b].end()
        );
    }
}


// ===================== RÓWNOLEGŁY ZAPIS =====================
// Każdy wątek zapisuje INNY fragment tablicy.
void write_back_parallel(
        std::vector<double>& arr,
        const std::vector<std::vector<double>>& buckets,
        size_t nbuckets) {

    // rozmiary kubełków
    std::vector<size_t> offsets(nbuckets + 1, 0);

    for (size_t b = 0; b < nbuckets; ++b) {
        offsets[b + 1] =
            offsets[b] + buckets[b].size();
    }

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < nbuckets; ++b) {

        size_t pos = offsets[b];

        for (double v : buckets[b]) {
            arr[pos++] = v;
        }
    }
}


// ===================== POMIAR =====================
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

    // lokalne kubełki:
    // [thread][bucket]
    std::vector<std::vector<std::vector<double>>> local(
        T,
        std::vector<std::vector<double>>(nbuckets)
    );

    // kubełki po merge
    std::vector<std::vector<double>> merged(nbuckets);

    // ===================== (a) LOSOWANIE =====================

    double ta0 = omp_get_wtime();

    fill_random(arr, MIN_VAL, MAX_VAL);

    t.t_a = omp_get_wtime() - ta0;

    double te0 = omp_get_wtime();

    // ===================== (b) ROZDZIAŁ =====================

    double tb0 = omp_get_wtime();

    bucket_distribute_algo3(
        arr,
        local,
        MIN_VAL,
        MAX_VAL,
        nbuckets
    );

    t.t_b = omp_get_wtime() - tb0;

    // ===================== (c) MERGE + SORT =====================

    double tc0 = omp_get_wtime();

    merge_and_sort(
        local,
        merged,
        nbuckets
    );

    t.t_c = omp_get_wtime() - tc0;

    // ===================== (d) ZAPIS =====================

    double td0 = omp_get_wtime();

    write_back_parallel(
        arr,
        merged,
        nbuckets
    );

    t.t_d = omp_get_wtime() - td0;

    // ===================== (e) CAŁOŚĆ =====================

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

    std::cerr
        << "=== ALG3 (LOCAL BUCKETS + PARALLEL MERGE) ===\n";

    std::cerr
        << "N = " << N
        << " buckets = " << N_BUCKETS
        << " threads = " << nthreads
        << "\n";

    std::vector<double> arr(N);

    PhaseTimes t = run(arr, N_BUCKETS);

    std::cerr << std::fixed << std::setprecision(4);

    std::cerr
        << "(a) losowanie:   "
        << t.t_a
        << " s\n";

    std::cerr
        << "(b) rozdzial:    "
        << t.t_b
        << " s\n";

    std::cerr
        << "(c) merge+sort:  "
        << t.t_c
        << " s\n";

    std::cerr
        << "(d) zapis:       "
        << t.t_d
        << " s\n";

    std::cerr
        << "(e) calosc:      "
        << t.t_e
        << " s\n";

    std::cout
        << "RESULT,"
        << nthreads << ","
        << N << ","
        << N_BUCKETS << ","
        << t.t_a << ","
        << t.t_b << ","
        << t.t_c << ","
        << t.t_d << ","
        << t.t_e
        << "\n";

    return 0;
}