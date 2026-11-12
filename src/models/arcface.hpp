#pragma once
// SEMUA yang dibutuhkan pengenalan wajah (ArcFace) ada di file ini dan arcface.cpp:
//   1. kSpec, kTemplate, kEmbeddingSize   resep tensor input (crop 112x112), template landmark, panjang vektor
//   2. L2Normalize, Dot                   matematika embedding (dot dua vektor satuan = cosine similarity)
//   3. Embedder                           crop wajah -> [WinML] -> vektor 512 angka yang sudah dinormalisasi
//   4. Gallery                            daftar orang dikenal (nama + embedding) dan pencocokan
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include "../gpu/gpu_context.hpp"
#include "../gpu/nv12_bridge.hpp"
#include "../inference/onnx_model.hpp"
#include "../preprocess/similarity_transform.hpp"
#include "../preprocess/warp.hpp"
#include "../common/types.hpp"

namespace FaceRuntime::ArcFace
{
    // 1. Spesifikasi
    // ======================================================================
    // nilai = (rgb - 127,5) / 127,5. Piksel di luar frame = replikasi tepi.
    inline constexpr Preprocess::TensorSpec kSpec{
        "arcface",
        112,
        112,
        1.0F / 127.5F,
        -1.0F,
        Preprocess::OobMode::Clamp};

    inline constexpr std::size_t kEmbeddingSize = 512; // output w600k_r50: [N, 512]

    inline constexpr std::array<Common::Point2f, 5> kTemplate{
        {{38.2946F, 51.6963F},
         {73.5318F, 51.5014F},
         {56.0252F, 71.7366F},
         {41.5493F, 92.3655F},
         {70.7299F, 92.2041F}}};

    // 2. Matematika embedding
    // ======================================================================

    // Mengubah vektor embedding (misal 512 dimensi) menjadi Vektor Satuan (Unit Vector).
    // Analogi Fisika: Mempertahankan "arah" identitas wajah, tapi menyeragamkan panjangnya menjadi tepat 1.0.
    // Caranya dengan Teorema Pythagoras (L2 Norm): mengakarkan total kuadrat lalu membagikan setiap elemennya.
    // Tujuan:
    // - Menghilangkan bias kualitas gambar atau pencahayaan (yang membuat panjang vektor acak).
    // - Syarat mutlak sebelum fungsi Dot() dijalankan agar hasilnya menjadi murni Cosine Similarity.
    inline void L2Normalize(std::vector<float> &values)
    {
        if (values.empty())
        {
            throw std::runtime_error("L2Normalize: vektor kosong.");
        }

        // pythagoras
        double sumSquares = 0.0;
        for (float v : values)
        {
            if (!std::isfinite(v))
            {
                throw std::runtime_error("L2Normalize: ada nilai bukan angka terhingga.");
            }
            sumSquares += static_cast<double>(v) * static_cast<double>(v);
        }
        const double norm = std::sqrt(sumSquares);
        if (norm < 1e-9)
        {
            throw std::runtime_error("L2Normalize: norma nol.");
        }
        for (float &v : values)
        {
            v = static_cast<float>(v / norm);
        }
    }

    // Menghitung kemiripan dua wajah menggunakan Dot Product (Perkalian Komponen).
    // Karena vektor 'a' dan 'b' sudah dinormalisasi (panjang = 1.0), maka rumus geometrisnya:
    // Dot = |A| * |B| * cos(θ) -> 1 * 1 * cos(θ) = cos(θ).
    // Output otomatis menjadi Cosine Similarity (rentang -1.0 sampai 1.0).
    //
    // Interpretasi Output:
    //  ~ 1.0 (Sudut 0°)   : Vektor sejajar, arah sama persis (Wajah sangat identik/orang yang sama).
    //    0.0 (Sudut 90°)  : Vektor tegak lurus (Wajah tidak memiliki kemiripan sama sekali).
    //  < 0.0 (Sudut >90°) : Vektor berlawanan (Wajah bertolak belakang secara matematis).
    inline float Dot(
        std::vector<float> const &a,
        std::vector<float> const &b)
    {
        if (a.size() != b.size())
        {
            throw std::runtime_error("Dot: panjang vektor berbeda.");
        }

        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            sum += static_cast<double>(a[i]) *
                   static_cast<double>(b[i]);
        }
        return static_cast<float>(sum);
    }

    // EMbed: Proses memetakan data kompleks mentah (gambar, teks, audio) menjadi bentuk koordinat
    class Embedder
    {
    public:
        Embedder(Gpu::GpuContext &gpu,
                 Preprocess::WarpContext &warp,
                 std::filesystem::path const &modelPath,
                 Preprocess::FrameSize source);

        // Crop 112x112 memakai `transform` (landmark -> kTemplate), jalankan ArcFace, lalu kembalikan
        // vektor 512 angka yang SUDAH L2-normalized. Gerbang OOB dan pose dikerjakan pemanggil.
        [[nodiscard]] std::vector<float> embed(Gpu::Nv12Frame const &frame,
                                               Preprocess::SimilarityTransform2D const &transform);

        void printModelSignature() const
        {
            model.printSignature("ArcFace");
        }

        // DEBUG: isi tensor crop 112x112 terakhir (lambat).
        [[nodiscard]] std::vector<float> debugReadback() const
        {
            return preprocessor.debugReadback();
        }

    private:
        Preprocess::WarpPreprocessor preprocessor;
        Inference::OnnxModel model;
    };

    // Gallery: daftar orang dikenal (nama + embedding) dan pencocokan
    struct Match
    {
        std::string name = "Unknown";
        float similarity = -1.0F;         // cosine terbaik; -1 kalau galeri kosong
        float runnerUpSimilarity = -1.0F; // cosine terbaik dari ORANG LAIN (nama berbeda); -1 kalau tidak ada
        [[nodiscard]] float margin() const { return similarity - runnerUpSimilarity; }
    };

    // Daftar orang dikenal: nama + embedding (SUDAH L2-normalized). Satu orang boleh punya beberapa
    // entri (beberapa pose). Tidak ada ambang di sini: keputusan ada di TrackDecision (pipeline.hpp).
    class Gallery
    {
    public:
        void add(
            std::string name,
            std::vector<float> embedding);
        [[nodiscard]] Match match(
            std::vector<float> const &embedding) const;
        [[nodiscard]] std::size_t size() const
        {
            return entries.size();
        }
        void save(std::filesystem::path const &path) const; // "nama v0 v1 ... v511" per baris
        void load(std::filesystem::path const &path);

    private:
        struct Entry
        {
            std::string name;
            std::vector<float> embedding;
        };
        std::vector<Entry> entries;
    };
}