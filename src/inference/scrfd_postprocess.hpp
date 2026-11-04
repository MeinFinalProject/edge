#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "../common/types.hpp"

// Murni CPU, tanpa WinRT/D3D. Aman di-compile dan diuji di mana saja.
namespace FaceRuntime::Inference
{
    struct TensorView
    {
        std::vector<std::uint64_t> shape; // mis. {12800, 4} atau {1, 12800, 4}
        const float *data = nullptr;
        std::size_t count = 0;
    };

    // Tiga kelompok output untuk satu stride (8, 16, atau 32).
    struct ScrfdLevel
    {
        int stride = 0;
        int anchorCount = 0;           // jumlah baris = (W/stride)*(H/stride)*2
        const float *scores = nullptr; // [anchorCount * 1]
        const float *bbox = nullptr;   // [anchorCount * 4]  jarak l,t,r,b (satuan stride)
        const float *kps = nullptr;    // [anchorCount * 10] offset 5 titik (satuan stride)
    };

    struct ScrfdDecodeConfig
    {
        std::uint32_t inputWidth = 640;
        std::uint32_t inputHeight = 640;
        float scoreThreshold = 0.10F; // rendah, supaya ByteTrack punya deteksi "low score"
        float nmsThreshold = 0.40F;
        // Skala letterbox: ukuran_resize / ukuran_sumber, PER SUMBU.
        // Padding ada di KANAN dan BAWAH, jadi tidak ada offset yang dikurangkan.
        float scaleX = 1.0F;
        float scaleY = 1.0F;
    };

    // Mengelompokkan output model (urutan bebas) menjadi 3 level berdasarkan
    // bentuknya: kolom terakhir 1/4/10 menentukan jenis, jumlah baris menentukan stride.
    // Return false dan isi `error` kalau model tidak cocok (mis. tanpa landmark).
    bool RouteScrfdOutputs(
        std::vector<TensorView> const &outputs,
        std::uint32_t inputWidth,
        std::uint32_t inputHeight,
        std::array<ScrfdLevel, 3> &levels,
        std::string &error);

    // Decode anchor -> kotak + landmark, balik ke koordinat frame kamera, lalu NMS.
    std::vector<Common::FaceDetection> DecodeScrfd(
        std::array<ScrfdLevel, 3> const &levels,
        ScrfdDecodeConfig const &config);

    std::vector<Common::FaceDetection> NonMaxSuppression(
        std::vector<Common::FaceDetection> detections,
        float iouThreshold);
}