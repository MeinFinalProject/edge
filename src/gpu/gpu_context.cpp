#include "gpu_context.hpp"
#include "camera_texture_import.hpp"
#include <memory>
#include <winrt/base.h>
#include <windows.ai.machinelearning.native.h>

namespace FaceRuntime::Gpu
{
    GpuContext::GpuContext()
    {
        initializeHardware();
        createAiContexts();
    }

    winrt::com_ptr<ID3D11DeviceContext> GpuContext::getD3D11Context() const
    {
        return d3d11Context;
    }

    winrt::com_ptr<ID3D12Device> GpuContext::getD3D12Device() const
    {
        return d3d12Device;
    }

    winrt::com_ptr<ID3D12CommandQueue> GpuContext::getCommandQueue() const
    {
        return commandQueue;
    }

    winrt::com_ptr<ID3D11Device> GpuContext::getD3D11Device() const
    {
        return d3d11Device;
    }

    winrt::Windows::AI::MachineLearning::LearningModelDevice GpuContext::getLearningDevice() const
    {
        return learningModelDevice;
    }

    std::shared_ptr<CameraTextureImporter> GpuContext::getCameraTextureImporter() const
    {
        return textureImporter;
    }

    void GpuContext::initializeHardware()
    {
        winrt::check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d12Device)));

        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

        winrt::check_hresult(d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

        IUnknown *commandQueues[] = {commandQueue.get()};
        winrt::check_hresult(D3D11On12CreateDevice(
            d3d12Device.get(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            nullptr,
            0,
            commandQueues,
            1,
            0,
            d3d11Device.put(),
            d3d11Context.put(),
            nullptr));

        winrt::check_hresult(d3d11Device.as(IID_PPV_ARGS(&d3d11On12Device)));
    }

    void GpuContext::createAiContexts()
    {
        winrt::com_ptr<ILearningModelDeviceFactoryNative> factory =
            winrt::get_activation_factory<
                winrt::Windows::AI::MachineLearning::LearningModelDevice,
                ILearningModelDeviceFactoryNative>();

        winrt::com_ptr<IUnknown> unknownDevice;
        winrt::check_hresult(factory->CreateFromD3D12CommandQueue(commandQueue.get(), unknownDevice.put()));

        learningModelDevice = unknownDevice.as<winrt::Windows::AI::MachineLearning::LearningModelDevice>();

        textureImporter = std::make_shared<CameraTextureImporter>(d3d11Device, d3d12Device, d3d11On12Device);
    }
}