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
    class GpuContext
    {
    public:
        GpuContext();
        ~GpuContext();

        GpuContext(GpuContext const &) = delete;
        GpuContext &operator=(GpuContext const &) = delete;

        [[nodiscard]] ID3D12Device *device() const
        {
            return d3d12Device.get();
        }
        [[nodiscard]] ID3D12CommandQueue *queue() const
        {
            return commandQueue.get();
        }
        [[nodiscard]] winrt::com_ptr<ID3D12Device> deviceRef() const
        {
            return d3d12Device;
        }
        [[nodiscard]] winrt::Windows::AI::MachineLearning::LearningModelDevice learningDevice() const
        {
            return learningModelDevice;
        }

        void waitIdle();

    private:
        winrt::com_ptr<ID3D12Device> d3d12Device;
        winrt::com_ptr<ID3D12CommandQueue> commandQueue;
        winrt::com_ptr<ID3D12Fence> fence;
        winrt::handle fenceEvent;
        std::uint64_t lastFenceValue = 0;

        winrt::Windows::AI::MachineLearning::LearningModelDevice learningModelDevice{nullptr};
    };
}