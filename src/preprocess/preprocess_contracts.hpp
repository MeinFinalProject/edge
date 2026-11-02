#pragma once
#include <cstdint>
#include <winrt/Windows.AI.MachineLearning.h>
#include <d3d11.h>
#include <d3d12.h>
#include <d3d11_4.h>
#include <memory>

#include "../gpu/command_recorder.hpp"
#include "../gpu/compute_kernel.hpp"
#include "../gpu/d3d12_helpers.hpp"
#include "../gpu/gpu_context.hpp"
#include "../gpu/nv12_bridge.hpp"
#include "../gpu/tensor_buffer.hpp"
#include "../camera/camera.hpp"

namespace FaceRuntime::Preprocess
{
    struct ScrfdPreprocessConfig
    {
        std::uint32_t width = 640;
        std::uint32_t height = 640;
    };

    // Letterbox KANAN - BAWAH : gambar ditaruh di pojok kiri - atas(0, 0) kanvas 640x640,
    // sisanya (kanan dan bawah) hitam. Tidak ada pad_x / pad_y.
    struct ScrfdPreprocessTransform
    {
        std::uint32_t sourceWidth = 0;
        std::uint32_t sourceHeight = 0;
        std::uint32_t resizedWidth = 0;
        std::uint32_t resizedHeight = 0;
        float scaleX = 1.0F; // resizedWidth  / sourceWidth
        float scaleY = 1.0F; // resizedHeight / sourceHeight
    };

    class ScrfdPreprocessor
    {
    public:
        // Ukuran frame sumber diambil dari CameraConfig (satu-satunya sumber kebenaran).
        ScrfdPreprocessor(
            Gpu::GpuContext &gpu,
            CameraConfig const &camera,
            ScrfdPreprocessConfig config = {});

        // Mengisi tensor [1,3,640,640] dan mengembalikannya. Pekerjaan GPU sudah DIKIRIM
        // ke antrean tetapi belum tentu selesai; WinML memakai antrean yang sama sehingga aman.
        winrt::Windows::AI::MachineLearning::TensorFloat preprocess(Gpu::Nv12Frame const &frame);

        [[nodiscard]] ScrfdPreprocessTransform const &transform() const { return transform_; }

        // DEBUG: salin isi tensor ke CPU (lambat). Untuk dump ke file saat verifikasi.
        [[nodiscard]] std::vector<float> debugReadback() { return tensor.readbackToCpu(gpu); }

    private:
        Gpu::GpuContext &gpu;
        ScrfdPreprocessConfig config;
        ScrfdPreprocessTransform transform_;
        Gpu::TensorBuffer tensor;
        Gpu::ComputeKernel kernel;
        Gpu::CommandRecorder recorder;
    };

    struct SimilarityTransform2D
    {
        std::array<float, 6> srctoDst{};
        std::array<float, 6> dsttoSrc{};
        float rmse = 0.0F;
    };

    // -------------------------------------------------------------------- PAD
    struct PadPreprocessingConfig
    {
        std::uint32_t width = 80;
        std::uint32_t height = 80;
        float maxOutofBoundsRatio = 0.30F;
    };

    struct PreparedPadInput
    {
        winrt::Windows::AI::MachineLearning::TensorFloat tensor{nullptr};
        float outofBoundsRatio = 0.0F;
        [[nodiscard]] bool usable(float maxRatio) const { return tensor && outofBoundsRatio <= maxRatio; }
    };

    class PadPreprocessor
    {
    public:
        PadPreprocessor(Gpu::GpuContext &gpu, CameraConfig const &camera, PadPreprocessingConfig config = {});

        // BLOKIR: menunggu GPU selesai karena counter OOB harus dibaca CPU.
        PreparedPadInput preprocess(Gpu::Nv12Frame const &frame, SimilarityTransform2D const &transform);
        [[nodiscard]] std::vector<float> debugReadback() { return tensor.readbackToCpu(gpu); }

    private:
        Gpu::GpuContext &gpu;
        PadPreprocessingConfig config;
        std::uint32_t sourceWidth;
        std::uint32_t sourceHeight;
        Gpu::TensorBuffer tensor;
        Gpu::ComputeKernel kernel;
        Gpu::CommandRecorder recorder;

        winrt::com_ptr<ID3D12Resource> oobCounter;    // 4 byte di VRAM, ditulis shader (atomic)
        winrt::com_ptr<ID3D12Resource> oobZeroSource; // 4 byte UPLOAD berisi 0, untuk mereset counter
        winrt::com_ptr<ID3D12Resource> oobReadback;   // 4 byte READBACK, dibaca CPU
    };

    // ---------------------------------------------------------------- ArcFace
    struct ArcFacePreprocessConfig
    {
        std::uint32_t width = 112;
        std::uint32_t height = 112;
    };

    class ArcFacePreprocessor
    {
    public:
        ArcFacePreprocessor(Gpu::GpuContext &gpu, CameraConfig const &camera, ArcFacePreprocessConfig config = {});

        winrt::Windows::AI::MachineLearning::TensorFloat preprocess(
            Gpu::Nv12Frame const &frame, SimilarityTransform2D const &transform);
        [[nodiscard]] std::vector<float> debugReadback() { return tensor.readbackToCpu(gpu); }

    private:
        Gpu::GpuContext &gpu;
        ArcFacePreprocessConfig config;
        std::uint32_t sourceWidth;
        std::uint32_t sourceHeight;
        Gpu::TensorBuffer tensor;
        Gpu::ComputeKernel kernel;
        Gpu::CommandRecorder recorder;
    };
}