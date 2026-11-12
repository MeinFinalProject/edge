#include "byte_tracker.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace FaceRuntime::Tracking
{
    namespace
    {
        using Vec4 = std::array<double, 4>;
        using Vec8 = std::array<double, 8>;
        using Mat4 = std::array<Vec4, 4>;
        using Mat8 = std::array<Vec8, 8>;
        constexpr double kStdWeightPosition = 1.0 / 20.0;  // Toleransi Posisi 5%
        constexpr double kStdWeightVelocity = 1.0 / 160.0; // Toleransi Kecepatan 0.625%

        // Forward declarations:
        Vec4 BoxToXyah(Common::BoxXyxy const &b);
        Common::BoxXyxy XyahToBox(Vec8 const &m);

        // Kalman filter 8 dimensi: [cx, cy, aspect(w/h), h, vx, vy, va, vh]
        // CX, CY : posisi tengah kotak,
        // A : rasio aspek (w/h), proporsi ktak
        // H : tinggi kotak
        // vx: Kecepatan geser x. Kalau positif, orangnya sedang berjalan ke kanan.
        // vy: Kecepatan vertikal. Kalau orangnya melompat, nilainya berubah.
        // vh: Kecepatan "Zoom". Ini efek 3D di layar 2D. Kalau nilainya positif tinggi, artinya kotak wajah terus membesar (orangnya menuju kamera).
        // va: Kecepatan perubahan proporsi. Biasanya nol. Tapi kalau kamu menoleh dari hadap depan ke profil samping, rasio kotak wajahmu mengecil, dan kecepatan putaran kepalamu direkam di angka ini.
        struct KalmanState
        {
            Vec8 mean{}; // [Cx, Cy, aspect, h, vx, vy, va, vh]  posisi + kecepatan
            Mat8 cov{};  // matriks kovarians 8x8 ("Tingkat Keraguan")
        };

        // Struktur hasil dari algoritma Hungarian (Kuhn-Munkres) untuk masalah penugasan linier.
        struct Assignment
        {
            std::vector<std::pair<int, int>> matches; // (indeks baris, indeks kolom)
            std::vector<int> unmatchedRows;           // Track ID yang kehilangan jejak (wajahnya hilang dari frame)
            std::vector<int> unmatchedCols;           // Deteksi AI baru yang butuh Track ID (Akte Kelahiran) baru
        };

        enum class State
        {
            Tracked,
            Lost
        };

        struct Track
        {
            int id = 0;
            State state = State::Tracked;
            bool activated = false;
            KalmanState kalman;
            float score = 0.0F;
            int lastFrame = 0; // frame terakhir track ini cocok dengan deteksi
            int lastDetection = -1;

            [[nodiscard]] Common::BoxXyxy box() const
            {
                return XyahToBox(kalman.mean);
            }
        };

        // Invers matriks 4x4().
        Mat4 Invert4(Mat4 a)
        {
            Mat4 inv{};
            for (int i = 0; i < 4; ++i)
            {
                inv[i][i] = 1.0;
            }
            for (int col = 0; col < 4; ++col)
            {
                int pivot = col;
                for (int r = col + 1; r < 4; ++r)
                {
                    if (std::abs(a[r][col]) > std::abs(a[pivot][col]))
                    {
                        pivot = r;
                    }
                }
                std::swap(a[col], a[pivot]);
                std::swap(inv[col], inv[pivot]);
                const double diag = a[col][col];
                for (int c = 0; c < 4; ++c)
                {
                    a[col][c] /= diag;
                    inv[col][c] /= diag;
                }
                for (int r = 0; r < 4; ++r)
                {
                    if (r == col)
                    {
                        continue;
                    }
                    const double f = a[r][col];
                    for (int c = 0; c < 4; ++c)
                    {
                        a[r][c] -= f * a[col][c];
                        inv[r][c] -= f * inv[col][c];
                    }
                }
            }
            return inv;
        };

        Vec4 BoxToXyah(Common::BoxXyxy const &b)
        {
            const double w = b.width();
            const double h = b.height();
            return {b.topLeft.x + w / 2.0,
                    b.topLeft.y + h / 2.0,
                    w / std::max(h, 1e-6),
                    h};
        };

        Common::BoxXyxy XyahToBox(Vec8 const &m)
        {
            const double h = m[3];
            const double w = m[2] * h;
            Common::BoxXyxy b;
            b.topLeft.x = static_cast<float>(m[0] - w / 2.0);
            b.topLeft.y = static_cast<float>(m[1] - h / 2.0);
            b.bottomRight.x = static_cast<float>(m[0] + w / 2.0);
            b.bottomRight.y = static_cast<float>(m[1] + h / 2.0);
            return b;
        };

        // Inisialisasi Kalman filter untuk kotak baru.
        // Posisi awal = deteksi, kecepatan awal = 0.
        // z = zentrum(observed) = [cx, cy, aspect, h]
        KalmanState KalmanInitiate(Vec4 const &z)
        {
            KalmanState s;
            // mean = [cx, cy, aspect, h, 0, 0, 0, 0]
            for (int i = 0; i < 4; ++i)
            {
                s.mean[i] = z[i];
            }
            const double h = z[3];
            // SET STANDARD DEVIATION (BATAS TOLERANSI GETARAN)
            // Mengukur seberapa tidak yakinnya kita terhadap posisi dan kecepatan saat ini.
            const std::array<double, 8> std{
                2 * kStdWeightPosition * h,   // [0] Posisi X: Keraguan awal diperbesar 2x lipat.
                2 * kStdWeightPosition * h,   // [1] Posisi Y: Keraguan awal diperbesar 2x lipat.
                1e-2,                         // [2] Rasio (Aspect): Proporsi wajah manusia sangat padat, toleransi diset statis sangat kecil (0.01).
                2 * kStdWeightPosition * h,   // [3] Tinggi kotak: Keraguan awal diperbesar 2x lipat.
                10 * kStdWeightVelocity * h,  // [4] Kecepatan X: Keraguan awal diperbesar 10x lipat.
                10 * kStdWeightVelocity * h,  // [5] Kecepatan Y: Keraguan awal diperbesar 10x lipat.
                1e-5,                         // [6] Kecepatan Rasio: Proporsi wajah manusia sangat padat, toleransi diset statis sangat kecil (0.00001).
                10 * kStdWeightVelocity * h}; // [7] Kecepatan Tinggi: Keraguan awal diperbesar 10x lipat.

            // RAKIT MATRIKS KOVARIANSI (ZONA KERAGUAN)
            // Matriks ini memodelkan seberapa tidak presisi state kita terhadap kondisi real di dunia nyata.
            // Kita hanya mengisi garis diagonal utamanya dengan nilai Varians (Standard Deviasi dikuadratkan).
            // - Sel Diagonal (s.cov[i][i]): Ketidakpastian absolut dari tiap-tiap variabel (cx, cy, dsb) secara independen.
            // - Sel Non-Diagonal (s.cov[i][j] = 0): Mempresentasikan "Zero Initial Cross-Correlation".
            // Artinya, pada t=0, kita berasumsi bahwa deviasi/error pada posisi X tidak memiliki korelasi
            // matematis dengan deviasi pada posisi Y atau kecepatannya, karena kita belum memiliki data riwayat.
            // Nilai 0 ini tidak akan permanen; di frame berikutnya, perhitungan Kalman Gain (saat Update)
            // akan secara otomatis mengisi sel non-diagonal ini dengan nilai korelasi seiring terbentuknya vektor momentum.
            for (int i = 0; i < 8; ++i)
            {
                s.cov[i][i] = std[i] * std[i];
            }
            return s;
        }

        // Memprediksi status (posisi & keraguan) di masa depan (frame berikutnya).
        // Fungsi ini mengeksekusi model
        // kinematika dasar: Posisi_Baru = Posisi_Lama + (Kecepatan * Waktu)
        void KalmanPredict(KalmanState &s)
        {
            // 1. SET PROCESS NOISE (MATRIKS Q)
            // Mendefinisikan ketidakpastian intrinsik dari sistem (karena manusia bisa berubah pikiran/arah tiba-tiba).
            // Berbeda dengan Inisialisasi awal, kali ini kita menggunakan bobot wajar (tidak dikali 2 atau 10).
            const double h = s.mean[3];
            const std::array<double, 8> std{
                kStdWeightPosition * h,
                kStdWeightPosition * h,
                1e-2,
                kStdWeightPosition * h,
                kStdWeightVelocity * h,
                kStdWeightVelocity * h,
                1e-5,
                kStdWeightVelocity * h};

            // 2. PREDIKSI STATE MEAN (POSISI & KECEPATAN MASA DEPAN)
            // Rumus Matematika: mean' = F * mean
            // Di sini, F (State Transition Model) disederhanakan tanpa perkalian matriks penuh.
            // Karena asumsi Delta Time (dt) = 1 frame, kita cukup menjumlahkan kecepatan ke posisi.
            // cx_baru = cx_lama + vx,
            // cy_baru = cy_lama + vy,
            // dst.
            // Kecepatan (indeks 4-7) diasumsikan konstan.
            for (int i = 0; i < 4; ++i)
            {
                s.mean[i] += s.mean[i + 4];
            }

            // 3. RAKIT MATRIKS TRANSISI (F)
            // Matriks 8x8 yang menjabarkan hukum fisika: bagaimana status frame ini berdampak ke frame depan.
            Mat8 f{};

            // Mengisi garis diagonal dengan 1.0 (Identitas).
            // Maknanya: Posisi X di masa depan dipengaruhi oleh Posisi X saat ini, Kecepatan X dipengaruhi Kecepatan X saat ini.
            for (int i = 0; i < 8; ++i)
            {
                f[i][i] = 1.0;
            }

            // Mengisi kuadran kanan atas dengan 1.0 (Korelasi Kecepatan ke Posisi).
            // Maknanya: Posisi X di masa depan JUGA dipengaruhi oleh Kecepatan X saat ini.
            for (int i = 0; i < 4; ++i)
            {
                f[i][i + 4] = 1.0;
            }

            // 4. PREDIKSI MATRIKS KOVARIANSI (ZONA KERAGUAN MASA DEPAN)
            // Rumus Matematika: cov' = (F * cov * F^T) + Q
            // Keraguan selalu bertambah seiring berjalannya waktu saat memprediksi masa depan.

            // Langkah A: Hitung (F * cov)
            // Matriks Transisi dikalikan dengan Zona Keraguan saat ini.
            Mat8 tmp{};
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    for (int k = 0; k < 8; ++k)
                        tmp[i][j] += f[i][k] * s.cov[k][j];

            // Langkah B: Hitung (tmp * F^T)
            // Hasil Langkah A dikalikan dengan Transpose dari Matriks Transisi.
            // Transpose (F^T) dicapai secara elegan dengan menukar indeks saat membaca `f`: memanggil `f[j][k]` alih-alih `f[k][j]`.
            // Di tahap inilah sel Non-Diagonal (Cross-Correlation) yang tadinya 0 mulai terisi angka momentum.
            Mat8 next{};
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    for (int k = 0; k < 8; ++k)
                        next[i][j] += tmp[i][k] * f[j][k]; // f[j][k] = F^T[k][j]

            // Langkah C: Tambahkan Process Noise (Q)
            // Menambahkan keraguan intrinsik (dari array `std` di atas) ke garis diagonal.
            for (int i = 0; i < 8; ++i)
            {
                next[i][i] += std[i] * std[i];
            }

            // 5. UPDATE STATE KOVARIANSI
            // Terapkan zona keraguan yang sudah membesar ini sebagai acuan di frame berikutnya.
            s.cov = next;
        }

        // Mengoreksi tebakan (Predict) masa lalu menggunakan data nyata (z) dari AI saat ini.
        // Konsep Utama: The Kalman Gain (K).
        // Fungsi ini menghitung bobot kepercayaan (K). Jika sensor AI (R) lebih presisi dari tebakan kita (P),
        // maka K akan besar dan kita merevisi tebakan kita mengikuti AI. Sebaliknya, jika AI tidak stabil,
        // K akan kecil, dan kita tetap mempertahankan tebakan inersia pergerakan.
        void KalmanUpdate(KalmanState &s, Vec4 const &z)
        {
            const double h = s.mean[3];
            // 1. SET MEASUREMENT NOISE (R)
            // Mendefinisikan seberapa tidak percayanya kita pada mata kamera/AI (z).
            // Perhatikan bahwa di sini hanya ada 4 array (Posisi & Ukuran), karena AI (z)
            // tidak bisa mendeteksi Kecepatan secara instan, ia hanya mengirim letak piksel.
            const std::array<double, 4> std{
                kStdWeightPosition * h, kStdWeightPosition * h, 1e-1, kStdWeightPosition * h};

            // 2. MENGHITUNG INNOVATION COVARIANCE (S)
            // Rumus Matematika: S = (H * P * H^T) + R
            // Karena matriks observasi (H) kita hanya mengambil 4 data posisi pertama dari 8 data (P),
            // kita cukup memotong kuadran kiri atas (4x4) dari matriks Covariance (s.cov) saat ini.
            Mat4 S{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    S[i][j] = s.cov[i][j]; // H * P * H^T

            // Menambahkan Measurement Noise (R)
            for (int i = 0; i < 4; ++i)
            {
                S[i][i] += std[i] * std[i];
            }
            // Membalikkan matriks S (Inverse) untuk digunakan di rumus Kalman Gain nanti.
            // Operasi ini aman dan O(1) karena ukurannya hanya statis 4x4.
            const Mat4 sInv = Invert4(S);

            /// 3. MENGHITUNG KALMAN GAIN (K)
            // Rumus Matematika: K = P * H^T * S^-1
            // K (8x4) menentukan porsi "kekuasaan" koreksi.
            // Nilai K yang tinggi berarti: "percayai saja kordinat AI!".
            std::array<std::array<double, 4>, 8> K{};
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k)
                        K[i][j] += s.cov[i][k] * sInv[k][j];

            // 4. MENGHITUNG INNOVATION (SELISIH FAKTA vs PREDIKSI)
            // Seberapa meleset tebakan kita (mean) dibandingkan dengan deteksi AI yang baru (z)?
            std::array<double, 4> innovation{};
            for (int i = 0; i < 4; ++i)
            {
                innovation[i] = z[i] - s.mean[i];
            }

            // 5. UPDATE STATE MEAN (KOREKSI POSISI & KECEPATAN)
            // Rumus Matematika: mean' = mean + (K * innovation)
            // Bagian terindah: Selisih (innovation) posisi yang terjadi saat ini digunakan
            // tidak hanya untuk mengoreksi letak, tetapi juga *merevisi tebakan kecepatan kita (indeks 4-7)*
            // karena efek perkalian dot product dengan Gain (K).
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 4; ++j)
                    s.mean[i] += K[i][j] * innovation[j];

            // 6. UPDATE STATE COVARIANCE (REDUKSI KERAGUAN)
            // Rumus Matematika: P' = P - (K * S * K^T)
            // Karena kita mendapat informasi nyata baru dari AI, tingkat kebingungan sistem (P)
            // pasti menurun. Kita mengurangi matriks Covariance kita agar lingkarannya mengecil.
            // Disusun secara efisien sebagai: P' = P - (P * H^T * K^T)
            Mat8 next = s.cov;
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    for (int k = 0; k < 4; ++k)
                        next[i][j] -= s.cov[i][k] * K[j][k];
            s.cov = next;
        }

        // Implementasi Algoritma Hungarian (Kuhn-Munkres) O(N^3) untuk "Linear Assignment Problem".
        // Fungsi ini bertindak sebagai "Biro Jodoh Global" yang meminimalkan Total Cost secara keseluruhan,
        // bukan sekadar mencari pasangan terdekat secara serakah (greedy).
        //
        // Prinsip Kerja (Primal-Dual Method):
        // 1. Sistem memberikan "Pajak/Subsidi" pada tiap baris (u) dan kolom (v).
        // 2. Cost Relatif dihitung dengan rumus: (Cost Asli - u - v).
        // 3. Aturan Mutlak: Jodoh hanya sah terbentuk JIKA Cost Relatif == 0.
        // 4. Jika terjadi rebutan (konflik), algoritma menyeimbangkan ulang nilai u dan v (lewat nilai 'delta')
        //    untuk membatalkan ikatan lama dan memunculkan Cost Relatif 0 pada pasangan baru yang sebelumnya terabaikan.
        std::vector<int> SolveSquareAssignment(std::vector<std::vector<double>> const &cost)
        {
            const int n = static_cast<int>(cost.size());
            const double inf = std::numeric_limits<double>::max() / 4; // Hindari overflow saat penjumlahan

            // POTENSI DUAL (SHADOW PRICES)
            // u: Pajak untuk baris (Tracker masa lalu)
            // v: Pajak untuk kolom (Deteksi masa kini)
            std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);

            // MANAJEMEN JALUR (PATH TRACKING)
            // Menggunakan 1-based indexing. Indeks 0 dipakai sebagai "Node Dummy" atau ruang tunggu.
            // p[j]: Menyimpan ID baris yang saat ini dijodohkan dengan kolom j.
            // way[j]: Menyimpan kolom pendahulu (parent) untuk melacak balik "Jalur Augmentasi".
            std::vector<int> p(n + 1, 0), way(n + 1, 0);

            // Iterasi untuk mencoba menjodohkan setiap baris (Tracker) satu per satu.
            for (int i = 1; i <= n; ++i)
            {
                p[0] = i;   // Taruh baris i di ruang tunggu (kolom 0) sebagai titik awal pencarian.
                int j0 = 0; // j0 adalah kolom yang sedang kita evaluasi.

                // minv: Menyimpan Cost Relatif terkecil (Slack) untuk mencapai tiap kolom. Ini mirip jarak di algoritma Dijkstra.
                std::vector<double> minv(n + 1, inf);
                std::vector<char> used(n + 1, false); // Menandai kolom yang sudah dieksplorasi di putaran ini.

                // TAHAP 1: PENCARIAN JALUR AUGMENTASI (Mencari rentetan perjodohan alternatif)
                do
                {
                    used[j0] = true;
                    const int i0 = p[j0]; // i0 adalah baris yang saat ini terhubung ke kolom j0.
                    double delta = inf;   // delta adalah batas minimum untuk mengubah pajak (potensi).
                    int j1 = 0;           // j1 akan menjadi kolom target berikutnya.

                    // Evaluasi semua kolom untuk mencari jembatan (edge) dengan Cost Relatif paling murah dari baris i0.
                    for (int j = 1; j <= n; ++j)
                    {
                        if (used[j])
                        {
                            continue; // Abaikan jika kolom sudah dieksplorasi.
                        }

                        // Menghitung Cost Relatif (Reduced Cost).
                        // Rumus fisiknya: Biaya Asli dikurangi total pajak baris dan pajak kolom.
                        const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];

                        // Relaksasi Dijkstra: Update jarak terdekat (minv) ke kolom j.
                        if (cur < minv[j])
                        {
                            minv[j] = cur;
                            way[j] = j0; // Catat bahwa kita sampai ke j melewati j0.
                        }

                        // Cari nilai minimum absolut (delta) di antara semua kolom yang belum dieksplor.
                        if (minv[j] < delta)
                        {
                            delta = minv[j];
                            j1 = j;
                        }
                    }

                    // UPDATE POTENSI (MENYESUAIKAN PAJAK)
                    // Karena kita mungkin mentok (tidak ada Cost Relatif = 0), kita memaksa 0 baru muncul
                    // dengan cara mengurangi 'delta' dari slack dan menyesuaikan pajak u dan v.
                    for (int j = 0; j <= n; ++j)
                    {
                        if (used[j])
                        {
                            u[p[j]] += delta; // Naikkan pajak baris yang sudah dieksplor.
                            v[j] -= delta;    // Turunkan pajak kolom yang sudah dieksplor.
                        }
                        else
                        {
                            minv[j] -= delta; // Kurangi slack kolom yang belum dieksplor.
                        }
                    }
                    // Pindah ke kolom terbaik berikutnya.
                    // Loop ini berhenti jika p[j0] == 0, jika tidak artinya kita menemukan kolom yang masih JOMBLO!
                    j0 = j1;
                } while (p[j0] != 0);

                // TAHAP 2: AUGMENTASI JALUR (CHAIN REACTION)
                // Kita telah menemukan ujung rantai (kolom jomblo). Sekarang kita merunut balik (backtrack)
                // menggunakan array 'way', merombak semua status pasangan di sepanjang jalur tersebut.
                // Efeknya: jumlah pasangan yang sukses terbentuk bertambah 1.
                do
                {
                    const int j1 = way[j0];
                    p[j0] = p[j1]; // Ambil alih pasangan dari kolom pendahulu.
                    j0 = j1;
                } while (j0 != 0);
            }

            // TAHAP 3: TRANSLASI HASIL
            // Algoritma di atas memetakan kolom ke baris (1-based).
            // Kita balikkan menjadi array standar:
            // indeks = baris (0-based),
            // value = kolom (0-based).
            std::vector<int> rowToCol(n, -1);
            for (int j = 1; j <= n; ++j)
            {
                rowToCol[p[j] - 1] = j - 1;
            }
            return rowToCol;
        }

        // Linear Assignment Wrapper: Mempersiapkan matriks Cost (yang mungkin tidak persegi)
        // agar bisa dikunyah oleh Algoritma Hungarian murni (SolveSquareAssignment).
        Assignment LinearAssignment(
            std::vector<std::vector<double>> const &cost,
            int rows,
            int cols,
            double limit)
        {
            Assignment out;

            // 1. KONDISI TRIVIAL (KOSONG)
            if (rows == 0 || cols == 0)
            {
                for (int i = 0; i < rows; ++i)
                    out.unmatchedRows.push_back(i);
                for (int j = 0; j < cols; ++j)
                    out.unmatchedCols.push_back(j);
                return out;
            }

            // 2. TRIK AUGMENTASI MATRIKS (Menjadi Persegi N x N)
            // Algoritma Hungarian butuh matriks persegi sempurna.
            // Jika ada 5 Tracker (Rows) dan 3 Deteksi (Cols), kita jumlahkan ukurannya (N = 5 + 3 = 8).
            const int n = rows + cols;
            const double big = 1e9; // Cost super mahal (Penalti/Tidak mungkin dipilih)
            std::vector<std::vector<double>> square(n, std::vector<double>(n, 0.0));

            // Membangun Matriks Augmented:
            // [ Cost Asli ] [ Dummy Cols ]
            // [ Dummy Rows] [ Blok 0     ]
            for (int i = 0; i < rows; ++i)
            {
                // Kuadran Kiri Atas: Salin data Cost murni dari IoU/Kalman
                for (int j = 0; j < cols; ++j)
                    square[i][j] = cost[i][j];

                // Kuadran Kanan Atas: "Kamera Pengawas Oklusi"
                // Nilai limit/2.0 ditanam menyilang. Ini jaring pengaman agar kalau Cost Asli
                // ke sebuah deteksi sudah menembus batas kewajaran (> limit), sistem lebih
                // memilih menjodohkan Tracker ke Dummy ini (yang artinya: Wajah hilang/Oklusi).
                for (int j = 0; j < rows; ++j)
                    square[i][cols + j] = (i == j) ? limit / 2.0 : big;
            }
            for (int i = 0; i < cols; ++i)
            {
                // Kuadran Kiri Bawah: "Tempat Lahir Wajah Baru"
                // Menyilang limit/2.0. Sama seperti di atas, jika wajah deteksi baru ini posisinya
                // terlalu jauh dari Tracker mana pun, sistem akan lebih memilih menjodohkannya
                // dengan Dummy ini, menandakan dia butuh Akte Kelahiran (Track ID baru).
                for (int j = 0; j < cols; ++j)
                    square[rows + i][j] = (i == j) ? limit / 2.0 : big;
                // Kuadran Kanan Bawah (Dummy x Dummy)
                // Otomatis terisi 0.0 (dari inisialisasi awal vector `square`).
            }

            // 3. EKSEKUSI HUNGARIAN MURNI (O(N^3))
            const auto rowToCol = SolveSquareAssignment(square);
            std::vector<char> colUsed(cols, false);

            // 4. PEMILAHAN HASIL (PARSING)
            for (int i = 0; i < rows; ++i)
            {

                const int j = rowToCol[i];
                // Apakah Tracker ini berjodoh dengan Deteksi Asli (bukan Dummy)
                // DAN jaraknya masih masuk akal (<= limit)?
                if (j >= 0 && j < cols && cost[i][j] <= limit)
                {
                    out.matches.emplace_back(i, j); // Sah! Tracker dilanjutkan.
                    colUsed[j] = true;
                }
                else
                {
                    // Tracker terjebak dengan Dummy atau Cost terlalu besar.
                    // Artinya: Orangnya hilang dari layar (oklusi/keluar frame).
                    out.unmatchedRows.push_back(i);
                }
            }
            for (int j = 0; j < cols; ++j)
            {
                if (!colUsed[j])
                    out.unmatchedCols.push_back(j); // Ini orang baru, buatkan Akte (KalmanInitiate).
            }
            return out;
        }

        float Iou(Common::BoxXyxy const &a,
                  Common::BoxXyxy const &b)
        {
            const float ix1 = std::max(a.topLeft.x, b.topLeft.x);
            const float iy1 = std::max(a.topLeft.y, b.topLeft.y);
            const float ix2 = std::min(a.bottomRight.x, b.bottomRight.x);
            const float iy2 = std::min(a.bottomRight.y, b.bottomRight.y);

            // Area irisan
            const float iw = std::max(0.0F, ix2 - ix1);
            const float ih = std::max(0.0F, iy2 - iy1);

            // Area gabungan
            const float inter = iw * ih;
            const float uni = a.area() + b.area() - inter;
            return uni > 0.0F ? inter / uni : 0.0F;
        };
    };

    struct ByteTracker::Impl
    {
        ByteTrackerConfig config;
        std::vector<Track> tracks; // semua track yang masih hidup (Tracked atau Lost)
        int frameId = 0;
        int nextId = 1;

        explicit Impl(ByteTrackerConfig c) : config(c) {}

        // Membangun Matriks Biaya (Cost Matrix) untuk diserahkan ke Algoritma Hungarian.
        // Konversi: Hungarian mencari "Biaya Minimum", sedangkan IoU (Tumpang Tindih) adalah "Nilai Maksimum".
        // Karena itu kita membaliknya menggunakan rumus fisik: Cost = 1.0 - IoU.
        //
        // Parameter fuseScore: Ini adalah inovasi ByteTrack.
        // Alih-alih hanya percaya pada geometri kotak murni (IoU), kita meleburkan (fuse)
        // Confidence Score dari AI deteksi wajah ke dalam perhitungan jarak.
        std::vector<std::vector<double>> BuildCost(
            std::vector<int> const &trackIdx,               // Indeks Tracker dari masa lalu
            std::vector<int> const &detIdx,                 // Indeks Deteksi dari masa kini
            std::vector<Common::FaceDetection> const &dets, // Data deteksi mentah (untuk ambil skor)
            bool fuseScore) const
        {
            // Inisialisasi Matriks 2D (Baris x Kolom) dengan nilai default 1.0 (Biaya paling mahal).
            // Artinya: Jika dua benda tidak bersentuhan sama sekali (IoU = 0), jarak/biayanya maksimal.
            std::vector<std::vector<double>> cost(
                trackIdx.size(),
                std::vector<double>(detIdx.size(), 1.0));

            // Iterasi Baris: Ambil prediksi kotak masa depan (dari Kalman Filter) milik orang ke-i
            for (std::size_t i = 0; i < trackIdx.size(); ++i)
            {
                const auto trackBox = tracks[trackIdx[i]].box();
                // Iterasi Kolom: Ambil kotak deteksi wajah nyata milik orang ke-j dari AI
                for (std::size_t j = 0; j < detIdx.size(); ++j)
                {
                    // Hitung Kemiripan Geometris Murni (Intersection over Union).
                    // Nilai mendekati 1.0 = Sangat berdempetan.
                    double sim = Iou(trackBox, dets[detIdx[j]].box);
                    if (fuseScore)
                    {
                        // PELEBURAN SKOR (FUSION)
                        // Melemahkan nilai IoU (Similarity) dengan persentase keyakinan AI.
                        // Jika AI tidak terlalu yakin itu wajah (misal skor = 0.3),
                        // maka seberapa bagus pun letak kotaknya berdempetan, nilainya akan anjlok.
                        // Tujuan: Mencegah Jenderal Hungarian menjodohkan target penting dengan deteksi palsu.
                        sim *= dets[detIdx[j]].score;
                    }
                    // Rumus Kebalikan: Ubah kemiripan tinggi (sim besar) menjadi biaya murah (cost kecil).
                    cost[i][j] = 1.0 - sim;
                }
            }
            return cost;
        };

        void ApplyMatch(Track &t,
                        Common::FaceDetection const &det,
                        int detectionIndex)
        {
            KalmanUpdate(t.kalman, BoxToXyah(det.box));
            t.state = State::Tracked;
            t.activated = true;
            t.score = det.score;
            t.lastFrame = frameId;
            t.lastDetection = detectionIndex;
        };

        TrackerResult Update(std::vector<Common::FaceDetection> const &dets)
        {
            ++frameId;
            TrackerResult result;

            // 1) Pisahkan deteksi: high (tahap 1) dan low (tahap 2).
            std::vector<int> high, low;
            for (int i = 0; i < static_cast<int>(dets.size()); ++i)
            {
                if (dets[i].score >= config.highThreshold)
                    high.push_back(i);
                else if (dets[i].score > config.lowThreshold)
                    low.push_back(i);
            }

            // 2) Pisahkan track: unconfirmed (baru, belum terbukti) dan pool (Tracked+Lost).
            std::vector<int> unconfirmed, pool;
            for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
            {
                if (tracks[i].state == State::Tracked && !tracks[i].activated)
                    unconfirmed.push_back(i);
                else
                    pool.push_back(i);
            }

            // 3) Prediksi posisi semua track di pool (Kalman).
            // Track Lost: kecepatan tinggi di-nol-kan.
            for (int idx : pool)
            {
                if (tracks[idx].state != State::Tracked)
                    tracks[idx].kalman.mean[7] = 0.0;
                KalmanPredict(tracks[idx].kalman);
            }

            std::vector<char> removedFlag(tracks.size(), false);

            // 4) Asosiasi tahap 1: pool vs deteksi skor tinggi.
            auto first = LinearAssignment(
                BuildCost(pool, high, dets, true),
                static_cast<int>(pool.size()), static_cast<int>(high.size()),
                config.matchThreshold);
            for (auto const &[r, c] : first.matches)
                ApplyMatch(tracks[pool[r]], dets[high[c]], high[c]);

            // 5) Asosiasi tahap 2: track berstatus Tracked yang tersisa vs deteksi skor rendah.
            std::vector<int> leftoverTracked;
            for (int r : first.unmatchedRows)
                if (tracks[pool[r]].state == State::Tracked)
                    leftoverTracked.push_back(pool[r]);

            auto second = LinearAssignment(
                BuildCost(leftoverTracked, low, dets, false),
                static_cast<int>(leftoverTracked.size()), static_cast<int>(low.size()),
                config.lowMatchThreshold);
            for (auto const &[r, c] : second.matches)
                ApplyMatch(tracks[leftoverTracked[r]], dets[low[c]], low[c]);
            for (int r : second.unmatchedRows)
                tracks[leftoverTracked[r]].state = State::Lost; // tidak ketemu = hilang sementara

            // 6) Track unconfirmed vs deteksi skor tinggi yang belum terpakai.
            std::vector<int> remainingHigh;
            for (int c : first.unmatchedCols)
                remainingHigh.push_back(high[c]);

            auto third = LinearAssignment(
                BuildCost(unconfirmed, remainingHigh, dets, true),
                static_cast<int>(unconfirmed.size()), static_cast<int>(remainingHigh.size()),
                config.unconfirmedMatchThreshold);
            for (auto const &[r, c] : third.matches)
                ApplyMatch(tracks[unconfirmed[r]], dets[remainingHigh[c]], remainingHigh[c]);
            for (int r : third.unmatchedRows)
                removedFlag[unconfirmed[r]] = true; // track baru yang tak terbukti = buang

            // 7) Deteksi skor tinggi yang tetap tak terpakai -> track baru.
            std::vector<Track> fresh;
            for (int c : third.unmatchedCols)
            {
                const int detIndex = remainingHigh[c];
                if (dets[detIndex].score < config.newTrackThreshold)
                    continue;
                Track t;
                t.id = nextId++;
                t.state = State::Tracked;
                t.activated = (frameId == 1); // frame pertama langsung terkonfirmasi
                t.kalman = KalmanInitiate(BoxToXyah(dets[detIndex].box));
                t.score = dets[detIndex].score;
                t.lastFrame = frameId;
                t.lastDetection = detIndex;
                fresh.push_back(t);
            }

            // 8) Buang track Lost yang terlalu lama hilang.
            for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
            {
                if (tracks[i].state == State::Lost &&
                    frameId - tracks[i].lastFrame > config.maxFrameAge)
                    removedFlag[i] = true;
            }

            // 9) Rapikan penyimpanan + kumpulkan hasil.
            std::vector<Track> survivors;
            for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
            {
                if (removedFlag[i])
                    result.removedIds.push_back(tracks[i].id);
                else
                    survivors.push_back(tracks[i]);
            }
            for (auto &t : fresh)
                survivors.push_back(t);
            tracks = std::move(survivors);

            for (auto const &t : tracks)
            {
                if (t.state == State::Tracked && t.activated && t.lastFrame == frameId)
                {
                    TrackedFace out;
                    out.trackId = t.id;
                    out.box = t.box();
                    out.score = t.score;
                    out.detectionIndex = t.lastDetection;
                    result.active.push_back(out);
                }
            }
            return result;
        }
    };

    ByteTracker::ByteTracker(ByteTrackerConfig config) : impl(std::make_unique<Impl>(config))
    {
    }
    ByteTracker::~ByteTracker() = default;

    TrackerResult ByteTracker::update(std::vector<Common::FaceDetection> const &detections)
    {
        return impl->Update(detections);
    }

    void ByteTracker::reset()
    {
        impl = std::make_unique<Impl>(impl->config);
    }
}