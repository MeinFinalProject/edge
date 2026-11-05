#include "preprocess_contracts.hpp"
#include <cstring>
#include "../gpu/d3d12_helpers.hpp"
#include "kernel_constants.hpp"
#include "pad_tensorize.h" // array byte `g_pad_tensorize`

namespace FaceRuntime::Preprocess
{
    PadPreprocessor::PadPreprocessor(Gpu::GpuContext &gpu, CameraConfig const &camera, PadPreprocessingConfig config)
        : gpu(gpu),
          config(config),
          sourceWidth(camera.width),
          sourceHeight(camera.height),
          tensor(gpu,
                 {1, 3, static_cast<std::int64_t>(config.height), static_cast<std::int64_t>(config.width)}),
          kernel(gpu.device(),
                 g_pad_tensorize,
                 sizeof(g_pad_tensorize),
                 sizeof(WarpConstants) / 4,
                 /*numUavs=*/2),
          recorder(gpu)
    {
        oobCounter = Gpu::CreateBuffer(
            gpu.device(),
            4,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        oobZeroSource = Gpu::CreateBuffer(
            gpu.device(),
            4,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ);
        {
            void *mapped = nullptr;
            D3D12_RANGE noRead{0, 0}; // CPU hanya menulis
            winrt::check_hresult(oobZeroSource->Map(0, &noRead, &mapped));
            const std::uint32_t zero = 0;
            std::memcpy(mapped, &zero, sizeof(zero));
            oobZeroSource->Unmap(0, nullptr);
        }

        oobReadback = Gpu::CreateBuffer(
            gpu.device(), 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    }

    PreparedPadInput PadPreprocessor::preprocess(
        Gpu::Nv12Frame const &frame, SimilarityTransform2D const &transform)
    {
        WarpConstants constants{};
        constants.srcWidth = sourceWidth;
        constants.srcHeight = sourceHeight;
        constants.dstWidth = config.width;
        constants.dstHeight = config.height;
        constants.m00 = transform.dsttoSrc[0];
        constants.m01 = transform.dsttoSrc[1];
        constants.m02 = transform.dsttoSrc[2];
        constants.m10 = transform.dsttoSrc[3];
        constants.m11 = transform.dsttoSrc[4];
        constants.m12 = transform.dsttoSrc[5];

        auto *commandList = recorder.begin();

        // (1) Reset counter ke 0: salin 4 byte nol dari buffer UPLOAD.
        D3D12_RESOURCE_BARRIER toCopyDest = Gpu::TransitionBarrier(
            oobCounter.get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST);
        commandList->ResourceBarrier(1, &toCopyDest);
        commandList->CopyBufferRegion(oobCounter.get(), 0, oobZeroSource.get(), 0, 4);
        D3D12_RESOURCE_BARRIER toUav = Gpu::TransitionBarrier(
            oobCounter.get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        commandList->ResourceBarrier(1, &toUav);

        // (2) Jalankan shader. u0 = tensor, u1 = counter OOB.
        kernel.bind(commandList, frame, &constants, tensor.resource(), oobCounter.get());
        commandList->Dispatch((config.width + 15) / 16, (config.height + 15) / 16, 1);

        // (3) Tunggu penulisan selesai, lalu salin counter ke buffer READBACK.
        D3D12_RESOURCE_BARRIER uavBarriers[2] = {
            Gpu::UavBarrier(tensor.resource()), Gpu::UavBarrier(oobCounter.get())};
        commandList->ResourceBarrier(2, uavBarriers);

        D3D12_RESOURCE_BARRIER toCopySource = Gpu::TransitionBarrier(
            oobCounter.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        commandList->ResourceBarrier(1, &toCopySource);
        commandList->CopyBufferRegion(oobReadback.get(), 0, oobCounter.get(), 0, 4);
        D3D12_RESOURCE_BARRIER backToUav = Gpu::TransitionBarrier(
            oobCounter.get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        commandList->ResourceBarrier(1, &backToUav);

        recorder.submitAndWait(); // CPU harus menunggu: hasil counter dibaca di bawah

        // (4) Baca counter.
        std::uint32_t outOfBoundsPixels = 0;
        {
            void *mapped = nullptr;
            D3D12_RANGE readRange{0, 4};
            winrt::check_hresult(oobReadback->Map(0, &readRange, &mapped));
            std::memcpy(&outOfBoundsPixels, mapped, sizeof(outOfBoundsPixels));
            D3D12_RANGE noWrite{0, 0};
            oobReadback->Unmap(0, &noWrite);
        }

        PreparedPadInput result;
        result.tensor = tensor.tensor();
        result.outofBoundsRatio =
            static_cast<float>(outOfBoundsPixels) /
            static_cast<float>(config.width * config.height);
        return result;
    }
}
