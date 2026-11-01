#include "gpu_context.hpp"
#include <memory>
#include <winrt/base.h>
#include <windows.ai.machinelearning.native.h>

namespace FaceRuntime::Gpu
{
    GpuContext::GpuContext()
    {
        winrt::check_hresult(
            D3D12CreateDevice(
                nullptr,
                D3D_FEATURE_LEVEL_12_0,
                IID_PPV_ARGS(d3d12Device.put())));

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        queueDesc.NodeMask = 0;
        winrt::check_hresult(
            d3d12Device->CreateCommandQueue(
                &queueDesc,
                IID_PPV_ARGS(commandQueue.put())));

        winrt::check_hresult(
            d3d12Device->CreateFence(
                0,
                D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(fence.put())));
        fenceEvent.attach(
            ::CreateEventW(
                nullptr,
                FALSE,
                FALSE,
                nullptr));
        if (!fenceEvent)
        {
            winrt::throw_last_error();
        }

        winrt::com_ptr<ILearningModelDeviceFactoryNative> factory =
            winrt::get_activation_factory<winrt::Windows::AI::MachineLearning::LearningModelDevice,
                                          ILearningModelDeviceFactoryNative>();

        winrt::com_ptr<::IUnknown> unknownDevice;
        winrt::check_hresult(
            factory->CreateFromD3D12CommandQueue(
                commandQueue.get(),
                unknownDevice.put()));
        unknownDevice.as<winrt::Windows::AI::MachineLearning::LearningModelDevice>();
    }

    GpuContext::~GpuContext()
    {
        try
        {
            waitIdle();
        }
        catch (...)
        {
        }
    }

    void GpuContext::waitIdle()
    {
        const std::uint64_t targetFenceValue = ++lastFenceValue;
        winrt::check_hresult(
            commandQueue->Signal(
                fence.get(),
                targetFenceValue));

        if (fence->GetCompletedValue() < targetFenceValue)
        {
            winrt::check_hresult(
                fence->SetEventOnCompletion(
                    targetFenceValue,
                    fenceEvent.get()));
            if (::WaitForSingleObject(
                    fenceEvent.get(),
                    INFINITE) != WAIT_OBJECT_0)
            {
                winrt::throw_last_error();
            }
        }

        if (fence->GetCompletedValue() == UINT64_MAX)
        {
            const HRESULT reason = d3d12Device->GetDeviceRemovedReason();
            winrt::throw_hresult(FAILED(reason) ? reason : DXGI_ERROR_DEVICE_REMOVED);
        }
    }
}