#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <winrt/Windows.AI.MachineLearning.h>
#include "../gpu/command_recorder.hpp"
#include "../gpu/compute_kernel.hpp"
#include "../gpu/gpu_context.hpp"
#include "../gpu/nv12_bridge.hpp"
#include "../gpu/tensor_buffer.hpp"

// WARP = "ambil frame kamera NV12,
// geser/skalakan/putar lewat satu matriks, tulis sebagai tensor
// float NCHW siap makan model".
// SATU shader dan SATU kelas melayani SEMUA model; yang beda antar
// model cuma TensorSpec (ukuran, normalisasi, perlakuan tepi)
// dan matriks yang diberikan pemanggil.

namespace FaceRuntime::Preprocess
{
    struct FrameSize
    {
        std::uint32_t width;
        std::uint32_t height;
    };

    // Mode OOB (Out Of Bounds) menentukan apa yang dilakukan shader ketika
    // jika piksel tujuan jatuh di luar frame sumber
    enum class OobMode : std::uint32_t
    {
        Clamp = 0, // replikasi piksel tepi (crop wajah)
        Fill = 1   // isi dengan normBias = hitam ternormalisasi
    };

    // Resep tensor satu model. nilai_tensor = rgb255 * normScale + normBias
    struct TensorSpec
    {
        char const *name; // nama tensor di model ONNX (misal "input.1")
        std::uint32_t width;
        std::uint32_t height;

        // Skala pengali piksel [0.0 .. 255.0].
        // Contoh Pengaturan:
        // - Skala [0.0 .. 1.0]     : 1.0f / 255.0f          (~0.00392157f)
        // - Skala [-1.0 .. 1.0]    : 1.0f / 127.5f          (~0.00784314f)
        // - SCRFD (StdDev=128.0)   : 1.0f / 128.0f          ( 0.00781250f)
        float normScale;

        // Pergeseran offset (bias) nilai piksel setelah dikali normScale.
        // fungsinya untuk memastikan mean piksel berada di 0.0
        // Contoh Pengaturan:
        // - Tanpa Shift [0.0 .. 1.0]: 0.0f
        // - Center Shift [-1.0..1.0]: -1.0f                 (dari -127.5 / 127.5)
        // - SCRFD (Mean=127.5)     : -127.5f / 128.0f      (-0.99609375f)
        float normBias;
        OobMode oob;
    };

    // Satu program shader + satu perekam command list,
    // dipakai BERGANTIAN oleh semua WarpPreprocessor.
    // Aman karena kerja GPU berjalan serial (satu frame in-flight, Evaluate WinML sinkron);
    // CommandRecorder::begin() menunggu GPU idle bila perlu.
    struct WarpContext
    {
        explicit WarpContext(Gpu::GpuContext &gpu);

        // record command list untuk satu frame kamera NV12
        Gpu::CommandRecorder recorder;
        // menjalankan shader yang dibangun dengan root signature
        //(aturan antara resource dan shader CPP - HLSL untuk data GPU)
        //+ PSO (pipeline state object,
        // yang mengatur state GPU untuk menjalankan shader)
        Gpu::ComputeKernel kernel;
    };

    // Satu instance per model. Menyimpan TensorSpec, buffer tensor,
    // dan context warp.
    class WarpPreprocessor
    {
    public:
        WarpPreprocessor(
            Gpu::GpuContext &gpu,
            WarpContext &context,
            FrameSize source,
            TensorSpec const &spec);

        // Mengirim kerja GPU ke antrean (TIDAK menunggu) lalu mengembalikan tensor input model.
        // `dstToSrc` = matriks 2x3 [m00 m01 m02 m10 m11 m12]: "piksel tujuan (x, y) mengambil dari
        // koordinat (m00*x + m01*y + m02, m10*x + m11*y + m12) di frame kamera".
        winrt::Windows::AI::MachineLearning::TensorFloat run(
            Gpu::Nv12Frame const &frame,
            std::array<float, 6> const &dstToSrc);

        [[nodiscard]] TensorSpec const &spec() const
        {
            return spec_;
        }

        // DEBUG: salin isi tensor ke CPU (lambat, memblokir).
        [[nodiscard]] std::vector<float> debugReadback() const
        {
            return tensor.readbackToCpu(gpu);
        }

    private:
        Gpu::GpuContext &gpu;
        WarpContext &context;
        FrameSize source;
        TensorSpec spec_;
        Gpu::TensorBuffer tensor;
    };
}