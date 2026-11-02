#include "tensor_buffer.hpp"
#include <windows.ai.machinelearning.native.h>
#include <cstring>
#include "command_recorder.hpp"
#include "d3d12_helpers.hpp"

namespace FaceRuntime::Gpu
{
    TensorBuffer::TensorBuffer(
        GpuContext &gpu,
        std::array<std::int64_t, 4> shapeNchw) : shape(shapeNchw)
    {
        std::uint64_t elements = 1;
        for (std::int64_t dim : shape)
        {
            elements *= dim;
        }
        bytes = elements * sizeof(float);

        // DEFAULT heap (VRAM) + izin UAV (shader boleh menulis).
        // Dibuat langsung dalam state UNORDERED_ACCESS dan SELALU dikembalikan ke state ini.
        buffer = CreateBuffer(
            gpu.device(),
            bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // Bungkus buffer D3D12 itu menjadi TensorFloat milik WinML (tanpa salinan).
        auto factory = winrt::get_activation_factory<winrt::Windows::AI::MachineLearning::TensorFloat,
                                                     ITensorStaticsNative>();
        winrt::com_ptr<::IUnknown> unknownTensor;
        winrt::check_hresult(
            factory->CreateFromD3D12Resource(
                buffer.get(),
                shape.data(),                             // {N, C, H, W}
                static_cast<std::uint32_t>(shape.size()), // 4 dimensi
                unknownTensor.put()));
        winmlTensor = unknownTensor.as<winrt::Windows::AI::MachineLearning::TensorFloat>();
    }

    // debug: salin isi tensor ke CPU (lambat, memblokir).
    // Untuk dump & verifikasi.
    std::vector<float> TensorBuffer::readbackToCpu(GpuContext &gpu) const
    {
        auto readback = CreateBuffer(
            gpu.device(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

        CommandRecorder recorder(gpu);
        auto *commandList = recorder.begin();
        D3D12_RESOURCE_BARRIER toCopy = TransitionBarrier(
            buffer.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        commandList->ResourceBarrier(1, &toCopy);
        commandList->CopyBufferRegion(readback.get(), 0, buffer.get(), 0, bytes);
        D3D12_RESOURCE_BARRIER toUav = TransitionBarrier(
            buffer.get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        commandList->ResourceBarrier(1, &toUav);
        recorder.submitAndWait();

        std::vector<float> result(bytes / sizeof(float));
        D3D12_RANGE readRange{0, static_cast<SIZE_T>(bytes)};
        void *mapped = nullptr;
        winrt::check_hresult(readback->Map(0, &readRange, &mapped));
        std::memcpy(result.data(), mapped, bytes);
        D3D12_RANGE writtenRange{0, 0}; // CPU tidak menulis apa pun
        readback->Unmap(0, &writtenRange);
        return result;
    }
}