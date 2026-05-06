#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <omp.h>

static size_t N         = 100000000ULL;
static size_t N_BUCKETS = 10000ULL;

constexpr double MIN_VAL = 0.0;
constexpr double MAX_VAL = 1.0;


// ===================== LOSOWANIE =====================
void fill_random(std::vector<double>& arr, double min_val, double max_val) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::seed_seq seeds{rd(), rd(), rd(), rd(),
                            static_cast<unsigned>(omp_get_thread_num())};

        std::mt19937_64 gen(seeds);
        std::uniform_real_distribution<double> dist(min_val, max_val);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < arr.size(); ++i) {
            arr[i] = dist(gen);
        }
    }
}


// ===================== ALGORYTM 3 =====================
// podział danych + lokalne kubełki + brak synchronizacji
void bucket_distribute_algo3(const std::vector<double>& arr,
                             std::vector<std::vector<std::vector<double>>>& local,
                             double min_val, double max_val) {

    int T = omp_get_max_threads();
    size_t n = arr.size();

    double range = max_val - min_val;
    double bucket_range = range / T;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        size_t start = tid * (n / T);
        size_t end   = (tid == T - 1) ? n : (tid + 1) * (n / T);

        for (size_t i = start; i < end; ++i) {
            double val = arr[i];

            size_t b = static_cast<size_t>((val - min_val) / bucket_range);
            if (b >= (size_t)T) b = T - 1;

            local[tid][b].push_back(val);
        }
    }
}


// ===================== MERGE + SORT =====================
void merge_and_sort(std::vector<std::vector<std::vector<double>>>& local,
                    std::vector<std::vector<double>>& merged) {

    int T = omp_get_max_threads();

    // merge kubełków (każdy wątek robi swój bucket)
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < T; ++b) {

        for (int t = 0; t < T; ++t) {
            merged[b].insert(
                merged[b].end(),
                local[t][b].begin(),
                local[t][b].end()
            );
        }

        std::sort(merged[b].begin(), merged[b].end());
    }
}


// ===================== ZAPIS =====================
void write_back(std::vector<double>& arr,
                const std::vector<std::vector<double>>& buckets) {

    size_t pos = 0;

    for (size_t i = 0; i < buckets.size(); ++i) {
        for (double v : buckets[i]) {
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
PhaseTimes run(std::vector<double>& arr, size_t nbuckets) {

    PhaseTimes t;

    int T = omp_get_max_threads();

    std::vector<std::vector<std::vector<double>>> local(T,
        std::vector<std::vector<double>>(T));

    std::vector<std::vector<double>> merged(T);

    // (a) losowanie
    double ta0 = omp_get_wtime();
    fill_random(arr, MIN_VAL, MAX_VAL);
    t.t_a = omp_get_wtime() - ta0;

    double te0 = omp_get_wtime();

    // (b) rozdział
    double tb0 = omp_get_wtime();
    bucket_distribute_algo3(arr, local, MIN_VAL, MAX_VAL);
    t.t_b = omp_get_wtime() - tb0;

    // (c) merge + sort
    double tc0 = omp_get_wtime();
    merge_and_sort(local, merged);
    t.t_c = omp_get_wtime() - tc0;

    // (d) zapis
    double td0 = omp_get_wtime();
    write_back(arr, merged);
    t.t_d = omp_get_wtime() - td0;

    // (e) całość
    t.t_e = omp_get_wtime() - te0;

    return t;
}


// ===================== MAIN =====================
int main(int argc, char** argv) {

    if (argc >= 2) N = std::stoull(argv[1]);
    if (argc >= 3) N_BUCKETS = std::stoull(argv[2]);

    int nthreads = omp_get_max_threads();

    std::cerr << "=== ALG3 (LOCAL BUCKETS + MERGE) ===\n";
    std::cerr << "N = " << N << " buckets = " << N_BUCKETS
              << " threads = " << nthreads << "\n";

    std::vector<double> arr(N);

    PhaseTimes t = run(arr, N_BUCKETS);

    std::cerr << std::fixed << std::setprecision(4);
    std::cerr << "(a) losowanie:   " << t.t_a << " s\n";
    std::cerr << "(b) rozdzial:    " << t.t_b << " s\n";
    std::cerr << "(c) merge+sort:  " << t.t_c << " s\n";
    std::cerr << "(d) zapis:       " << t.t_d << " s\n";
    std::cerr << "(e) calosc:      " << t.t_e << " s\n";

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