#pragma once
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace FaceRuntime::Inference
{
    // Salinan CPU dari satu tensor output model.
    struct OutputTensor
    {
        std::vector<std::uint64_t> shape;
        std::vector<float> data;
    };

    class OnnxModel
    {
    public:
        OnnxModel(
            std::filesystem::path const &modelPath,
            winrt::Windows::AI::MachineLearning::LearningModelDevice const &device);

        // Sinkron: kembali setelah GPU selesai menjalankan model.
        [[nodiscard]] std::vector<OutputTensor> run(
            winrt::Windows::AI::MachineLearning::TensorFloat const &input) const;

        // Debug: Cetak nama + shape semua input/output model (untuk verifikasi dengan Netron).
        void printSignature(char const *label) const;

    private:
        winrt::Windows::AI::MachineLearning::LearningModel model{nullptr};
        winrt::Windows::AI::MachineLearning::LearningModelSession session{nullptr};
        // Debug: Nama input model (untuk verifikasi dengan Netron).
        winrt::hstring inputName;
    };
}