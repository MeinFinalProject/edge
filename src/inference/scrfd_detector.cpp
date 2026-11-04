#include "scrfd_detector.hpp"
#include <stdexcept>

namespace FaceRuntime::Inference
{
    ScrfdDetector::ScrfdDetector(
        Gpu::GpuContext &gpu,
        std::filesystem::path const &modelPath,
        CameraConfig const &camera,
        ScrfdDecodeConfig config)
        : config(config),
          preprocessor(gpu, camera),
          model(modelPath, gpu.learningDevice())
    {
    }

    std::vector<Common::FaceDetection> ScrfdDetector::detect(Gpu::Nv12Frame const &frame)
    {
        auto input = preprocessor.preprocess(frame);
        auto outputs = model.run(input);

        std::vector<TensorView> views;
        views.reserve(outputs.size());
        for (auto const &out : outputs)
        {
            views.push_back(TensorView{
                out.shape,
                out.data.data(),
                out.data.size()});
        }

        auto const &letterbox = preprocessor.transform();

        std::array<ScrfdLevel, 3> levels;
        std::string error;
        if (!RouteScrfdOutputs(views, config.inputWidth, config.inputHeight, levels, error))
        {
            throw std::runtime_error("SCRFD: " + error);
        }

        ScrfdDecodeConfig decode;
        decode.inputWidth = config.inputWidth;
        decode.inputHeight = config.inputHeight;
        decode.scoreThreshold = config.scoreThreshold;
        decode.nmsThreshold = config.nmsThreshold;
        decode.scaleX = letterbox.scaleX; // padding di kanan-bawah: cukup bagi dengan skala
        decode.scaleY = letterbox.scaleY;
        return DecodeScrfd(levels, decode);
    }
}
