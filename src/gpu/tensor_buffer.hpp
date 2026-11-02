#pragma once
#include <d3d12.h>
#include <winrt/base.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <array>
#include <cstdint>
#include <vector>
#include "gpu_context.hpp"

namespace FaceRuntime::Gpu
{
    // Satu tensor float32 berlayout NCHW yang tinggal di VRAM.
    // Shader menulis ke buffer ini; WinML membacanya sebagai input model.
    class TensorBuffer
    {
    public:
        TensorBuffer(GpuContext &gpu,
                     std::array<std::int64_t, 4> shapeNchw);
        [[nodiscard]] ID3D12Resource *resource() const
        {
            return buffer.get();
        }
        [[nodiscard]] winrt::Windows::AI::MachineLearning::TensorFloat tensor() const
        {
            return winmlTensor;
        }
        [[nodiscard]] std::uint64_t sizeInBytes() const
        {
            return bytes;
        }

        // DEBUG SAJA: salin isi tensor ke CPU (lambat, memblokir). Untuk dump & verifikasi.
        [[nodiscard]] std::vector<float> readbackToCpu(
            GpuContext &gpu) const;

    private:
        std::array<std::int64_t, 4> shape;
        std::uint64_t bytes = 0;
        winrt::com_ptr<ID3D12Resource> buffer;
        winrt::Windows::AI::MachineLearning::TensorFloat winmlTensor{nullptr};
    };
}
