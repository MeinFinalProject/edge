#pragma once
#include <winrt/base.h>
#include <d3d11.h>
#include <d3d11on12.h>
#include <mutex>

namespace FaceRuntime::Gpu
{
    struct ImportedViews
    {
        winrt::com_ptr<ID3D11Resource> wrappedResource;
        winrt::com_ptr<ID3D11ShaderResourceView> lumaView;
        winrt::com_ptr<ID3D11ShaderResourceView> chromaView;
    };

    class WrappedAccess
    {
    public:
        WrappedAccess(winrt::com_ptr<ID3D11On12Device> device, ID3D11Resource *resource) noexcept : device(std::move(device)), resource(resource), active(device != nullptr && resource != nullptr)
        {
            if (active)
            {
                device->AcquireWrappedResources(&resource, 1);
            }
        }

        ~WrappedAccess()
        {
            if (active)
            {
                device->ReleaseWrappedResources(&resource, 1);
            }
        }

        WrappedAccess(WrappedAccess const &) = delete;
        WrappedAccess &operator=(WrappedAccess const &) = delete;

        WrappedAccess(WrappedAccess &&other) noexcept
            : device(std::move(other.device)),
              resource(std::exchange(other.resource, nullptr)),
              active(std::exchange(other.active, false)) {}

        WrappedAccess &operator=(WrappedAccess &&other) noexcept
        {
            if (this != &other)
            {
                if (active)
                {
                    device->ReleaseWrappedResources(&resource, 1);
                }
                device = std::move(other.device);
                resource = std::exchange(other.resource, nullptr);
                active = std::exchange(other.active, false);
            }
            return *this;
        }

    private:
        winrt::com_ptr<ID3D11On12Device> device;
        ID3D11Resource *resource = nullptr;
        bool active = false;
    };

    class CameraTextureImporter
    {
    public:
        explicit CameraTextureImporter(
            winrt::com_ptr<ID3D11Device> d3d11Device,
            winrt::com_ptr<ID3D12Device> d3d12Device,
            winrt::com_ptr<ID3D11On12Device> d3d11On12Device);

        ImportedViews importFrame(winrt::com_ptr<ID3D11Texture2D> const &texture);
        WrappedAccess access(ImportedViews const &views);

    private:
        winrt::com_ptr<ID3D11Device> d3d11Device;
        winrt::com_ptr<ID3D12Device> d3d12Device;
        winrt::com_ptr<ID3D11On12Device> d3d11On12Device;

        std::mutex cacheMutex;
        std::unordered_map<ID3D11Texture2D *, ImportedViews> viewCache;
    };
}