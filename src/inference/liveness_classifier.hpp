#pragma once
#include <filesystem>
#include "../gpu/gpu_context.hpp"
#include "onnx_model.hpp"
#include "../preprocess/preprocess_contracts.hpp"
#include "../common/types.hpp"

namespace FaceRuntime::Inference
{
    using FaceDetection = Common::FaceDetection;
    class LivenessClassifier
    {
    public:
        LivenessClassifier(
            Gpu::GpuContext &gpu,
            std::filesystem::path const &modelPath,
            CameraConfig const &camera);

        float LiveProbabilityFromOutput(std::vector<float> values);

        // Probabilitas 0..1 bahwa wajah di tensor adalah wajah asli.
        [[nodiscard]] float realProbability(
            Gpu::Nv12Frame const &frame,
            FaceDetection &face);
        void printModelSignature() const
        {
            model.printSignature("Liveness");
        }

    private:
        OnnxModel model;
        FaceRuntime::Preprocess::PadPreprocessor preprocessor;
    };
}
