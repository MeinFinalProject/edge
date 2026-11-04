#include "onnx_model.hpp"
#include <iostream>
#include <stdexcept>
#include <utility>

namespace FaceRuntime::Inference
{
    using namespace winrt::Windows::AI::MachineLearning;

    OnnxModel::OnnxModel(
        std::filesystem::path const &modelPath,
        LearningModelDevice const &device)
    {
        if (!std::filesystem::exists(modelPath))
        {
            throw std::runtime_error("Model file not found: " + modelPath.string());
        }

        // Baca file .onnx -> graf komputasi di memori.
        model = LearningModel::LoadFromFilePath(
            winrt::hstring{modelPath.c_str()});

        // Sesi = model + device. Di sinilah DirectML menyiapkan operator di GPU.
        session = LearningModelSession(model, device);

        if (model.InputFeatures().Size() != 1)
        {
            throw std::runtime_error("Model diharapkan punya tepat 1 input: " + modelPath.string());
        }
        inputName = model.InputFeatures().GetAt(0).Name();
    }

    std::vector<OutputTensor> OnnxModel::run(
        TensorFloat const &input) const
    {
        // Binding = "kertas pasangan": nama input -> tensor. Buat baru tiap panggilan (murah).
        LearningModelBinding binding(session);
        binding.Bind(inputName, input);

        // Antrean D3D12 yang sama dengan shader kita -> urutan terjamin.
        auto result = session.Evaluate(binding, L"face-runtime");

        std::vector<OutputTensor> outputs;
        for (auto const &pair : result.Outputs())
        {
            auto tensor = pair.Value().try_as<TensorFloat>();
            if (!tensor)
            {
                continue; // output non-float diabaikan
            }
            OutputTensor out;
            for (auto dim : tensor.Shape())
            {
                out.shape.push_back(dim);
            }
            auto view = tensor.GetAsVectorView();
            out.data.resize(view.Size());
            view.GetMany(0, winrt::array_view<float>(out.data));
            outputs.push_back(std::move(out));
        }
        return outputs;
    }

    void OnnxModel::printSignature(char const *label) const
    {
        auto dump = [](auto const &features, char const *kind)
        {
            for (auto const &feature : features)
            {
                std::cout << "  " << kind << " '" << winrt::to_string(feature.Name()) << "'";
                if (auto tensor = feature.template try_as<TensorFeatureDescriptor>())
                {
                    std::cout << " shape=[";
                    bool first = true;
                    for (auto dim : tensor.Shape())
                    {
                        std::cout << (first ? "" : ", ") << dim; // -1 = dimensi dinamis
                        first = false;
                    }
                    std::cout << "]";
                }
                std::cout << "\n";
            }
        };
        std::cout << "[Model " << label << "]\n";
        dump(model.InputFeatures(), "input ");
        dump(model.OutputFeatures(), "output");
    }
}