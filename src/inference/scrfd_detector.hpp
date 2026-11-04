#pragma once
#include <filesystem>
#include <vector>
#include "../common/types.hpp"
#include "../preprocess/preprocess_contracts.hpp"
#include "onnx_model.hpp"
#include "scrfd_postprocess.hpp"

namespace FaceRuntime::Inference
{
    class ScrfdDetector
    {
    public:
        ScrfdDetector(
            Gpu::GpuContext &gpu,
            std::filesystem::path const &modelPath,
            CameraConfig const &camera,
            ScrfdDecodeConfig config = {});

        // frame NV12 -> [GPU tensorize] -> [WinML SCRFD] -> [CPU decode+NMS] -> wajah (piksel kamera)
        [[nodiscard]] std::vector<Common::FaceDetection> detect(Gpu::Nv12Frame const &frame);

        void printModelSignature() const { model.printSignature("SCRFD"); }

    private:
        ScrfdDecodeConfig config;
        Preprocess::ScrfdPreprocessor preprocessor;
        OnnxModel model;
    };
}
