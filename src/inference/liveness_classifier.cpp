#include "liveness_classifier.hpp"
#include <stdexcept>

namespace FaceRuntime::Inference
{
    LivenessClassifier::LivenessClassifier(
        Gpu::GpuContext &gpu,
        std::filesystem::path const &modelPath,
        CameraConfig const &camera)
        : model(modelPath, gpu.learningDevice()),
          preprocessor(gpu, camera)
    {
    }

    float LivenessClassifier::LiveProbabilityFromOutput(std::vector<float> values)
    {
        if (values.empty())
        {
            throw std::runtime_error("Liveness: output kosong.");
        }
        for (float v : values)
        {
            if (!std::isfinite(v))
            {
                throw std::runtime_error("Liveness: output model bukan angka terhingga.");
            }
        }

        if (values.size() == 1)
        {
            const float v = values[0];
            if (v >= 0.0F && v <= 1.0F)
            {
                return v; // sudah probabilitas final
            }
            return 1.0F / (1.0F + std::exp(-v)); // logit mentah
        }
        throw std::runtime_error("Liveness: output model tidak sesuai.");
    }

    float LivenessClassifier::realProbability(
        Gpu::Nv12Frame const &frame,
        FaceDetection &face)
    {

        auto transform = Preprocess::EstimateSimilarityTransform(face.landmarks, Preprocess::kLivenessTemplate80);
        auto padInput = preprocessor.preprocess(frame, transform);
        auto outputs = model.run(padInput.tensor);
        if (outputs.empty())
        {
            throw std::runtime_error("Liveness: model tidak mengeluarkan output.");
        }
        return LiveProbabilityFromOutput(outputs[0].data);
    }
}
