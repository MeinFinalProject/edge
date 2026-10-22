#pragma once
#include <cstdint>
#include <winrt/Windows.AI.MachineLearning.h>
#include <d3d11.h>
#include <d3d12.h>
#include <d3d11_4.h>
#include <memory>
#include "../gpu/camera_texture_import.hpp"

namespace FaceRuntime::Preprocess
{
    struct ScrfdPreprocessConfig
    {
        std::uint32_t width = 640;
        std::uint32_t height = 640;
    };

    struct ScrfdPreprocessTransform
    {
        float scale = 1.0F;
        std::uint32_t sourceWidth = 0;
        std::uint32_t sourceHeight = 0;
        std::uint32_t resizedWidth = 0;
        std::uint32_t resizedHeight = 0;
    };

    struct PadPreprocessingConfig
    {
        std::uint32_t width = 80;
        std::uint32_t height = 80;
        float maxOutofBoundsRatio = 0.30F;
    };

    struct PreparedPadInput
    {
        winrt::Windows::AI::MachineLearning::TensorFloat tensor{nullptr};
        float outofBoundsRatio = 0.0F;
    };

    struct ArcFacePreprocessConfig
    {
        std::uint32_t width = 112;
        std::uint32_t height = 112;
    };

    struct SimilarityTransform2D
    {
        std::array<float, 6> srctoDst{};
        std::array<float, 6> dsttoSrc{};
        float rmse = 0.0F;
    };

    struct ScrfdGpuServices
    {
        std::shared_ptr<FaceRuntime::Gpu::CameraTextureImporter> importer;

        winrt::com_ptr<ID3D12Device> d3d12Device;
        winrt::com_ptr<ID3D12CommandQueue> commandQueue;

        winrt::Windows::AI::MachineLearning::LearningModelDevice
            learningDevice{nullptr};
    };

    class ScrfdPreprocessor
    {
    public:
        ScrfdPreprocessor(winrt::com_ptr<ID3D11Texture2D> firstTexture, ScrfdGpuServices services, ScrfdPreprocessConfig config = {});
        winrt::Windows::AI::MachineLearning::TensorFloat preprocess(winrt::com_ptr<ID3D11Texture2D> texture);

    private:
        ScrfdPreprocessConfig config;
        ScrfdPreprocessTransform transform;
        ScrfdGpuServices services;

        winrt::com_ptr<ID3D12Resource> tensorBuffer;
        winrt::com_ptr<ID3D12Resource> constantBuffer;

        winrt::Windows::AI::MachineLearning::TensorFloat winmlTensor{nullptr};
        winrt::com_ptr<ID3D12RootSignature> rootSignature;
        winrt::com_ptr<ID3D12PipelineState> pipelineState;
    };

    class PadPreprocessor
    {
    public:
        explicit PadPreprocessor(winrt::com_ptr<ID3D11Texture2D> firstTexture, PadPreprocessingConfig config = {});
        PreparedPadInput preprocess(winrt::com_ptr<ID3D11Texture2D> texture, SimilarityTransform2D const &transform);

    private:
        PadPreprocessingConfig config;

        winrt::com_ptr<ID3D12Resource> tensorBuffer;
        winrt::com_ptr<ID3D12Resource> constantBuffer;

        winrt::com_ptr<ID3D12Resource> oobCounterBuffer;
        winrt::com_ptr<ID3D12Resource> oobReadbackBuffer;

        winrt::Windows::AI::MachineLearning::TensorFloat winmlTensor{nullptr};
        winrt::com_ptr<ID3D12RootSignature> rootSignature;
        winrt::com_ptr<ID3D12PipelineState> pipelineState;
    };

    class ArcFacePreprocessor
    {
    public:
        explicit ArcFacePreprocessor(winrt::com_ptr<ID3D11Texture2D> firstTexture, ArcFacePreprocessConfig config = {});
        winrt::Windows::AI::MachineLearning::TensorFloat preprocess(winrt::com_ptr<ID3D11Texture2D> texture, SimilarityTransform2D const &transform);

    private:
        ArcFacePreprocessConfig config;

        winrt::com_ptr<ID3D12Resource> tensorBuffer;
        winrt::com_ptr<ID3D12Resource> constantBuffer;

        winrt::Windows::AI::MachineLearning::TensorFloat winmlTensor{nullptr};
        winrt::com_ptr<ID3D12RootSignature> rootSignature;
        winrt::com_ptr<ID3D12PipelineState> pipelineState;
    };
}