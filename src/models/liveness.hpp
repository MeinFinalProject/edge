#pragma once
// SEMUA yang dibutuhkan liveness ada di file ini dan liveness.cpp:
//   1. kSpec, kTemplate     resep tensor input (crop 80x80) dan template 5 landmark
//   2. Classifier           crop wajah -> [WinML] -> probabilitas "wajah asli" (satu observasi)
//   3. LlrTable, SprtAccumulator   dari banyak observasi -> keputusan Live / Spoof / Undecided
//                                  (SPRT: akumulasi Log-Likelihood Ratio, berhenti adaptif)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "../gpu/gpu_context.hpp"
#include "../gpu/nv12_bridge.hpp"
#include "../inference/onnx_model.hpp"
#include "../preprocess/similarity_transform.hpp"
#include "../preprocess/warp.hpp"
#include "../common/types.hpp"

namespace FaceRuntime::Liveness
{
    inline constexpr Preprocess::TensorSpec kSpec{
        "liveness",
        80,
        80,
        1.0F / 255.0F,
        0.0F,
        Preprocess::OobMode::Clamp};

    inline constexpr std::array<Common::Point2f, 5> kTemplate{
        {{32.03F, 38.06F},
         {47.89F, 37.98F},
         {40.01F, 47.08F},
         {33.50F, 56.36F},
         {46.63F, 56.29F}}};

    // Output model liveness InsightFace (addons/liveness.py) = SATU angka [N] = probabilitas
    // "wajah asli" yang SUDAH final. Jangan diolah lagi (tidak sigmoid, tidak softmax).
    // Hanya jika nilainya di luar 0..1, dianggap logit mentah -> sigmoid.
    inline float ProbabilityFromOutput(std::vector<float> const &values)
    {
        if (values.empty())
        {
            throw std::runtime_error("Liveness: output kosong.");
        }
        for (float v : values)
        {
            if (!std::isfinite(v))
            {
                throw std::runtime_error("Liveness: output model bukan angka terhingga.");
            }
        }
        if (values.size() != 1)
        {
            throw std::runtime_error("Liveness: output model harus 1 angka, diterima " +
                                     std::to_string(values.size()) + ". Periksa file ONNX di Netron.");
        }
        const float v = values[0];
        if (v >= 0.0F && v <= 1.0F)
        {
            return v; // sudah probabilitas final
        }
        return 1.0F / (1.0F + std::exp(-v)); // logit mentah
    }

    // Classifier: satu observasi
    class Classifier
    {
    public:
        Classifier(Gpu::GpuContext &gpu,
                   Preprocess::WarpContext &warp,
                   std::filesystem::path const &modelPath,
                   Preprocess::FrameSize source);

        // Crop 80x80 dari frame memakai `transform` (landmark -> kTemplate), lalu probabilitas 0..1
        // bahwa wajah itu ASLI. Gerbang OOB dan pose dikerjakan pemanggil SEBELUM fungsi ini.
        [[nodiscard]] float realProbability(Gpu::Nv12Frame const &frame,
                                            Preprocess::SimilarityTransform2D const &transform);

        void printModelSignature() const
        {
            model.printSignature("Liveness");
        }

        // DEBUG: isi tensor crop 80x80 terakhir (lambat).
        [[nodiscard]] std::vector<float> debugReadback() const
        {
            return preprocessor.debugReadback();
        }

    private:
        Preprocess::WarpPreprocessor preprocessor;
        Inference::OnnxModel model;
    };

    // SPRT: dari banyak observasi ke satu keputusan
    // H1 = wajah ASLI (bonafide),
    // H0 = SPOOF.
    // Skor positif = bukti "asli".
    struct SprtConfig
    {
        float alpha = 0.01F;      // P(putuskan Live | sebenarnya Spoof)  = false accept
        float beta = 0.05F;       // P(putuskan Spoof | sebenarnya Live)  = false reject
        float weight = 0.5F;      // diskon korelasi antar-frame, 0 < weight <= 1
        float llrClip = 3.0F;     // batas |LLR| satu observasi (sebelum diskon)
        int maxObservations = 12; // lewat dari ini tanpa keputusan -> Undecided
    };

    enum class Verdict
    {
        Pending,
        Live,
        Spoof,
        Undecided
    };

    // LLR(p) = ln P(p | Live) - ln P(p | Spoof),
    // diperkirakan dari histogram pada sumbu
    // logit(p) yang dipotong ke [-range, +range]
    // (model liveness sering jenuh dekat 0 / 1).
    class LlrTable
    {
    public:
        LlrTable(
            std::vector<float> llrPerBin,
            float range_)
            : llr_(std::move(llrPerBin)),
              range_(range_)
        {
            if (llr_.empty() || range_ <= 0.0F)
            {
                throw std::invalid_argument("LlrTable: tabel kosong atau range <= 0");
            }
        }

        // Laplace smoothing (+1 per bin)
        // supaya bin kosong tidak menghasilkan LLR tak hingga.
        static LlrTable Fit(std::vector<float> const &liveProbs,
                            std::vector<float> const &spoofProbs,
                            int bins = 12,
                            float range = 6.0F)
        {
            /// Validasi dasar: Algoritma butuh referensi baseline dari kedua skenario
            if (liveProbs.empty() || spoofProbs.empty())
            {
                throw std::invalid_argument("LlrTable::Fit: butuh sampel live DAN spoof");
            }

            // [1] INITIALIZATION & LAPLACE SMOOTHING
            // Alokasi memori untuk histogram. Diisi nilai awal 1.0 (bukan 0.0).
            // Kenapa 1.0? Jika ada bin (kategori skor) yang kosong (0), perhitungan log(0)
            // akan meledak menjadi -Infinity dan menghancurkan SPRT. Laplace smoothing mencegah ini.
            std::vector<double> live(static_cast<std::size_t>(bins), 1.0);
            std::vector<double> spoof(static_cast<std::size_t>(bins), 1.0);

            // [2] HISTOGRAM BINNING (Distribusi Frekuensi)
            // Memetakan skor mentah AI ke dalam kategori bin.
            // Misal skor 0.9 masuk ke bin ke-11. Kita tambah jumlah (frekuensi) di bin tersebut.
            for (float p : liveProbs)
                live[BinOf(p, bins, range)] += 1.0;
            for (float p : spoofProbs)
                spoof[BinOf(p, bins, range)] += 1.0;

            // [3] PROBABILITY NORMALIZATION
            // Total observasi + jumlah bins (karena kita pakai Laplace smoothing +1 per bin di awal).
            // Ini digunakan sebagai pembagi (denominator) untuk mengubah 'Jumlah' menjadi 'Probabilitas (0.0 - 1.0)'.
            const double nLive = static_cast<double>(liveProbs.size()) + bins;
            const double nSpoof = static_cast<double>(spoofProbs.size()) + bins;

            std::vector<float> llr(static_cast<std::size_t>(bins));

            // [4] LOG-LIKELIHOOD RATIO CALCULATION
            for (std::size_t i = 0; i < llr.size(); ++i)
            {
                // Matematika LLR: ln( P(x|Live) / P(x|Spoof) )
                // Berdasarkan sifat logaritma: ln(A / B) = ln(A) - ln(B)
                // live[i] / nLive   -> Peluang skor ini muncul jika wajah ASLI
                // spoof[i] / nSpoof -> Peluang skor ini muncul jika wajah PALSU
                //
                // Jika Live > Spoof, hasilnya positif (Mendukung H1 / Asli).
                // Jika Spoof > Live, hasilnya negatif (Mendukung H0 / Palsu).
                llr[i] = static_cast<float>(
                    std::log(live[i] / nLive) -
                    std::log(spoof[i] / nSpoof));
            }

            return LlrTable(std::move(llr), range);
        }

        // Tabel PENGGANTI SEMENTARA sebelum ada data kalibrasi: LLR = logit(p) di tengah tiap bin.
        // Benar hanya jika p adalah probabilitas posterior yang terkalibrasi dengan prior seimbang,
        // jadi anggap ini PEMANAS, bukan hasil akhir. Hasil akhir: `sprt_tool fit`.
        static LlrTable FromLogit(
            int bins = 12,
            float range_ = 6.0F)
        {
            std::vector<float> llr(static_cast<std::size_t>(bins));
            for (std::size_t i = 0; i < llr.size(); ++i)
            {
                llr[i] = -range_ + (static_cast<float>(i) + 0.5F) * (2.0F * range_ / static_cast<float>(bins));
            }
            return LlrTable(std::move(llr), range_);
        }

        // Format teks: baris 1 = range,
        // baris berikutnya = LLR per bin (keluaran `sprt_tool fit`).
        static LlrTable Load(std::istream &in)
        {
            float range_ = 0.0F, v = 0.0F;
            std::vector<float> values;
            if (!(in >> range_))
                throw std::runtime_error("LlrTable::Load: range tidak terbaca");
            while (in >> v)
                values.push_back(v);
            return LlrTable(std::move(values), range_);
        }

        // debug
        // simpan ke teks: baris 1 = range,
        // baris berikutnya = LLR per bin (untuk `sprt_tool fit`).
        void save(std::ostream &out) const
        {
            out << range_ << '\n';
            for (float v : llr_)
                out << v << '\n';
        }

        [[nodiscard]] float llr(float liveProbability) const
        {
            return llr_[BinOf(
                liveProbability,
                static_cast<int>(llr_.size()),
                range_)];
        }

        [[nodiscard]] std::vector<float> const &values() const
        {
            return llr_;
        }

        [[nodiscard]] float range() const
        {
            return range_;
        }

    private:
        static std::size_t BinOf(
            float p,
            int bins,
            float range_)
        {
            constexpr float kEps = 1e-6F;
            p = std::clamp(p,
                           kEps,
                           1.0F - kEps);
            const float z = std::clamp(std::log(p / (1.0F - p)),
                                       -range_,
                                       range_);
            const int bin = static_cast<int>((z + range_) /
                                             (2.0F * range_) *
                                             static_cast<float>(bins));

            return static_cast<std::size_t>(std::clamp(
                bin,
                0,
                bins - 1));
        }

        std::vector<float> llr_;
        float range_;
    };

    // Satu akumulator per TRACK. Wald SPRT terpotong:
    //   S_t = sum_i  weight * clip(LLR(p_i))
    //   S_t >= ln((1-beta)/alpha)  -> Live
    //   S_t <= ln(beta/(1-alpha))  -> Spoof
    //   observasi ke-maxObservations tanpa keputusan -> Undecided
    class SprtAccumulator
    {
    public:
        SprtAccumulator(
            SprtConfig config,
            LlrTable const &table)
            : config(config),
              table(&table)
        {
            if (!(config.alpha > 0.0F &&
                  config.beta > 0.0F &&
                  config.alpha + config.beta < 1.0F))
            {
                throw std::invalid_argument("SprtConfig: butuh 0 < alpha, 0 < beta, alpha + beta < 1");
            }

            if (!(config.weight > 0.0F &&
                  config.weight <= 1.0F) ||
                config.llrClip <= 0.0F ||
                config.maxObservations < 1)
            {
                throw std::invalid_argument("SprtConfig: weight/llrClip/maxObservations tidak valid");
            }

            upper = std::log((1.0F - config.beta) /
                             config.alpha);
            lower = std::log(config.beta /
                             (1.0F - config.alpha));
        }

        // Hanya panggil dengan observasi yang LOLOS gerbang kualitas.
        Verdict observe(float liveProbability)
        {
            if (verdict_ != Verdict::Pending)
            {
                return verdict_;
            }

            const float llr = std::clamp(
                table->llr(liveProbability),
                -config.llrClip,
                config.llrClip);
            sum_ += config.weight * llr;
            ++count_;

            if (sum_ >= upper)
                verdict_ = Verdict::Live;
            else if (sum_ <= lower)
                verdict_ = Verdict::Spoof;
            else if (count_ >= config.maxObservations)
                verdict_ = Verdict::Undecided;
            return verdict_;
        }

        void reset()
        {
            sum_ = 0.0F;
            count_ = 0;
            verdict_ = Verdict::Pending;
        }

        [[nodiscard]] Verdict verdict() const
        {
            return verdict_;
        }
        [[nodiscard]] float sum() const
        {
            return sum_;
        }
        [[nodiscard]] int count() const
        {
            return count_;
        }
        [[nodiscard]] float upperBound() const
        {
            return upper;
        }
        [[nodiscard]] float lowerBound() const
        {
            return lower;
        }

    private:
        SprtConfig config;
        LlrTable const *table; // tabel milik pipeline, hidup lebih lama dari semua akumulator
        float upper = 0.0F;
        float lower = 0.0F;
        float sum_ = 0.0F;
        int count_ = 0;
        Verdict verdict_ = Verdict::Pending;
    };
}