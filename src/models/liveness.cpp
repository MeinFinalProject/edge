#include "liveness.hpp"

namespace FaceRuntime::Liveness
{
    Classifier::Classifier(Gpu::GpuContext &gpu,
                           Preprocess::WarpContext &warp,
                           std::filesystem::path const &modelPath,
                           Preprocess::FrameSize source)
        : preprocessor(gpu,
                       warp,
                       source,
                       kSpec),
          model(modelPath,
                gpu.learningDevice())
    {
    }

    float Classifier::realProbability(Gpu::Nv12Frame const &frame,
                                      Preprocess::SimilarityTransform2D const &transform)
    {
        auto outputs = model.run(
            preprocessor.run(frame,
                             transform.dsttoSrc));
        if (outputs.empty())
        {
            throw std::runtime_error("Liveness: model tidak mengeluarkan output.");
        }
        return ProbabilityFromOutput(outputs[0].data);
    }
}
