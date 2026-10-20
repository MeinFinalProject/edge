#pragma once

#include <d3d11_4.h>
#include <d3d12.h>
#include <d3d11on12.h>
#include <dxgi1_6.h>
#include <winrt/base.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <memory>

namespace FaceRuntime::Gpu
{
    class CameraTextureImporter;
};

namespace FaceRuntime::Gpu
{
    class GpuContext
    {
    public:
        GpuContext();
        ~GpuContext() = default;

        GpuContext(GpuContext const &) = delete;
        GpuContext &operator=(GpuContext const &) = delete;

        winrt::Windows::AI::MachineLearning::LearningModelDevice getLearningDevice() const;
        std::shared_ptr<CameraTextureImporter> getCameraTextureImporter() const;

    private:
        void initializeHardware();
        void createAiContexts();

        winrt::com_ptr<ID3D12Device> d3d12Device;
        winrt::com_ptr<ID3D12CommandQueue> commandQueue;

        winrt::com_ptr<ID3D11Device> d3d11Device;
        winrt::com_ptr<ID3D11On12Device> d3d11On12Device;

        winrt::Windows::AI::MachineLearning::LearningModelDevice learningModelDevice{nullptr};
        std::shared_ptr<CameraTextureImporter> textureImporter;
    };
}