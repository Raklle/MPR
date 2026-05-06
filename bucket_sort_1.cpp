#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>

static size_t N         = 100000000ULL;
static size_t N_BUCKETS = 10000ULL;

constexpr double MIN_VAL = 0.0;
constexpr double MAX_VAL = 1.0;


// ======================= GENEROWANIE =======================
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


// ======================= ALGORYTM 1 =======================
// każdy wątek czyta CAŁĄ tablicę + własne kubełki
void bucket_sort_algo1(std::vector<double>& arr,
                       std::vector<std::vector<double>>& buckets,
                       double min_val, double max_val) {

    int T = omp_get_max_threads();
    size_t nb = buckets.size();

    double range = max_val - min_val;
    double bucket_range = range / nb;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        size_t start = min_val + tid * (nb / T);
        size_t end   = (tid == T - 1) ? nb : (tid + 1) * (nb / T);

        // KAŻDY WĄTEK CZYTA CAŁĄ TABLICĘ
        for (size_t i = 0; i < arr.size(); ++i) {
            double val = arr[i];

            size_t idx = static_cast<size_t>((val - min_val) / bucket_range);
            if (idx >= nb) idx = nb - 1;

            // tylko wątek "właściciel" zakresu kubełków zapisuje
            if (idx >= start && idx < end) {
                buckets[idx].push_back(val);
            }
        }
    }

    // sortowanie kubełków
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < nb; ++i) {
        std::sort(buckets[i].begin(), buckets[i].end());
    }

    // zapis wyników
    size_t pos = 0;
    for (size_t i = 0; i < nb; ++i) {
        for (double v : buckets[i]) {
            arr[pos++] = v;
        }
    }
}


// ======================= WERYFIKACJA =======================
bool verify_sorted(const std::vector<double>& arr) {
    return std::is_sorted(arr.begin(), arr.end());
}


// ======================= BENCHMARK =======================
struct PhaseTimes {
    double t_a = 0.0;
    double t_b = 0.0;
    double t_c = 0.0;
    double t_d = 0.0;
    double t_e = 0.0;
};

PhaseTimes run_bucket_sort(std::vector<double>& arr, size_t n_buckets) {
    PhaseTimes t;

    std::vector<std::vector<double>> buckets(n_buckets);

    double ta0 = omp_get_wtime();
    fill_random(arr, MIN_VAL, MAX_VAL);
    t.t_a = omp_get_wtime() - ta0;

    double te0 = omp_get_wtime();

    double tb0 = omp_get_wtime();
    bucket_sort_algo1(arr, buckets, MIN_VAL, MAX_VAL);
    t.t_b = omp_get_wtime() - tb0;

    double tc0 = omp_get_wtime();
    // sortowanie już w algorytmie (dla spójności zostawiamy pusty etap)
    t.t_c = omp_get_wtime() - tc0;

    double td0 = omp_get_wtime();
    // zapis już w algorytmie
    t.t_d = omp_get_wtime() - td0;

    t.t_e = omp_get_wtime() - te0;

    return t;
}


// ======================= MAIN =======================
int main(int argc, char** argv) {

    if (argc >= 2) N         = std::stoull(argv[1]);
    if (argc >= 3) N_BUCKETS = std::stoull(argv[2]);

    int nthreads = omp_get_max_threads();

    std::cerr << "=== Bucket Sort ALG.1 (full scan per thread) ===\n"
              << "N         = " << N << "\n"
              << "N_BUCKETS = " << N_BUCKETS << "\n"
              << "Threads   = " << nthreads << "\n";

    std::vector<double> arr(N);

    PhaseTimes t = run_bucket_sort(arr, N_BUCKETS);

    bool sorted = verify_sorted(arr);

    std::cerr << std::fixed << std::setprecision(4);
    std::cerr << "(a) losowanie:   " << t.t_a << " s\n";
    std::cerr << "(b) rozdzial:    " << t.t_b << " s\n";
    std::cerr << "(c) sortowanie:  " << t.t_c << " s\n";
    std::cerr << "(d) zapis:       " << t.t_d << " s\n";
    std::cerr << "(e) calosc:      " << t.t_e << " s\n";
    std::cerr << "Posortowane:     " << (sorted ? "TAK" : "NIE") << "\n";

    std::cout << "RESULT,"
              << nthreads << ","
              << N << ","
              << N_BUCKETS << ","
              << t.t_a << ","
              << t.t_b << ","
              << t.t_c << ","
              << t.t_d << ","
              << t.t_e << ","
              << (sorted ? 1 : 0) << "\n";

    return sorted ? 0 : 1;
}