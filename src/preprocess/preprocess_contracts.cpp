#include "preprocess_contracts.hpp"
#include "../src/preprocess/kernel_constants.hpp"
#include "scrfd_tensorize.h"
#include <utility>
#include <iostream>
#include <stdexcept>

namespace FaceRuntime::Preprocess
{
    namespace
    {
        ScrfdPreprocessTransform ComputeLetterbox(
            std::uint32_t sourceWidth,
            std::uint32_t sourceHeight,
            ScrfdPreprocessConfig const &config)
        {
            if (sourceWidth == 0 || sourceHeight == 0)
            {
                throw std::invalid_argument("ScrfdPreprocessor: ukuran sumber 0.");
            }

            ScrfdPreprocessTransform t;
            t.sourceWidth = sourceWidth;
            t.sourceHeight = sourceHeight;

            // Sama persis dengan scrfd.py InsightFace:
            const float imageRatio = static_cast<float>(sourceHeight) / static_cast<float>(sourceWidth);
            const float modelRatio = static_cast<float>(config.height) / static_cast<float>(config.width);
            if (imageRatio > modelRatio)
            {
                t.resizedHeight = config.height;
                t.resizedWidth = static_cast<std::uint32_t>(
                    static_cast<float>(t.resizedHeight) / imageRatio);
            }
            else
            {
                t.resizedWidth = config.width;
                t.resizedHeight = static_cast<std::uint32_t>(
                    static_cast<float>(t.resizedWidth) * imageRatio);
            }
            t.scaleX = static_cast<float>(t.resizedWidth) / static_cast<float>(sourceWidth);
            t.scaleY = static_cast<float>(t.resizedHeight) / static_cast<float>(sourceHeight);
            return t;
        }
    }

    ScrfdPreprocessor::ScrfdPreprocessor(
        Gpu::GpuContext &gpu,
        CameraConfig const &camera,
        ScrfdPreprocessConfig config)
        : gpu(gpu),
          config(config),
          transform_(ComputeLetterbox(camera.width,
                                      camera.height,
                                      config)),
          tensor(gpu,
                 {1, 3, static_cast<std::int64_t>(config.height),
                  static_cast<std::int64_t>(config.width)}),
          kernel(gpu.device(),
                 g_scrfd_tensorize,
                 sizeof(g_scrfd_tensorize),
                 sizeof(ScrfdConstants) / 4,
                 1),
          recorder(gpu)
    {
    }

    winrt::Windows::AI::MachineLearning::TensorFloat
    ScrfdPreprocessor::preprocess(Gpu::Nv12Frame const &frame)
    {
        ScrfdConstants constants{};
        constants.srcWidth = transform_.sourceWidth;
        constants.srcHeight = transform_.sourceHeight;
        constants.dstWidth = config.width;
        constants.dstHeight = config.height;
        constants.resizedWidth = transform_.resizedWidth;
        constants.resizedHeight = transform_.resizedHeight;

        auto *commandList = recorder.begin();

        // Tekstur NV12 berada di state COMMON dan hanya DIBACA: D3D12 mempromosikannya
        // otomatis ke state baca saat dipakai, dan mengembalikannya ke COMMON setelah selesai.
        kernel.bind(commandList, frame, &constants, tensor.resource());

        // 640/16 = 40 grup per sisi. Harus sama dengan [numthreads(16,16,1)] di HLSL.
        commandList->Dispatch((config.width + 15) / 16, (config.height + 15) / 16, 1);

        // Pastikan semua penulisan shader tuntas sebelum WinML membaca tensor.
        D3D12_RESOURCE_BARRIER uav = Gpu::UavBarrier(tensor.resource());
        commandList->ResourceBarrier(1, &uav);

        recorder.submit();
        return tensor.tensor();
    }
}