#include "preprocess_contracts.hpp"
#include "scrfd_tensorize.h"
#include <utility>
#include <iostream>
#include <stdexcept>

namespace FaceRuntime::Preprocess
{
    ScrfdPreprocessor::ScrfdPreprocessor(
        winrt::com_ptr<ID3D11Texture2D> firstTexture,
        ScrfdGpuServices services,
        ScrfdPreprocessConfig config)
        : config(config), services(std::move(services))
    {
        if (!firstTexture)
        {
            throw std::invalid_argument("firstTexture pointer is null.");
        }
        if (!this->services.importer)
        {
            throw std::invalid_argument(
                "ScrfdPreprocessor: importer is null.");
        }

        if (!this->services.d3d12Device)
        {
            throw std::invalid_argument(
                "ScrfdPreprocessor: d3d12Device is null.");
        }

        D3D11_TEXTURE2D_DESC desc{};
        firstTexture->GetDesc(&desc);

        transform.sourceWidth = desc.Width;
        transform.sourceHeight = desc.Height;

        const float sourceAspect =
            static_cast<float>(transform.sourceHeight) /
            static_cast<float>(transform.sourceWidth);

        const float targetAspect =
            static_cast<float>(config.height) /
            static_cast<float>(config.width);

        if (sourceAspect > targetAspect)
        {
            transform.resizedHeight = config.height;
            transform.resizedWidth =
                static_cast<std::uint32_t>(
                    transform.resizedHeight / sourceAspect);
        }
        else
        {
            transform.resizedWidth = config.width;
            transform.resizedHeight =
                static_cast<std::uint32_t>(
                    transform.resizedWidth * sourceAspect);
        }

        transform.scale =
            static_cast<float>(transform.resizedHeight) /
            static_cast<float>(transform.sourceHeight);
    };

    winrt::Windows::AI::MachineLearning::TensorFloat
    ScrfdPreprocessor::preprocess(
        winrt::com_ptr<ID3D11Texture2D> texture)
    {
        std::cout << "[SCRFD] preprocess mulai\n";

        if (!texture)
        {
            throw std::invalid_argument(
                "ScrfdPreprocessor::preprocess: texture is null.");
        }

        auto views = services.importer->importFrame(texture);

        auto access = services.importer->access(views);

        std::cout << "[SCRFD] access wrapped resource berhasil\n";
        std::cout << "[SCRFD] import Y/UV berhasil\n";

        return nullptr;
    }
}