#pragma once
#include <d3d11_4.h>
#include <d3d12.h>
#include <winrt/base.h>
#include <cstddef>
#include <vector>
#include <unordered_map>

namespace FaceRuntime::Gpu
{
    struct Nv12Frame
    {
        ID3D12Resource *resource = nullptr;
        ID3D12DescriptorHeap *srvHeap = nullptr;
    };

    class Nv12Bridge
    {
    public:
        explicit Nv12Bridge(
            winrt::com_ptr<ID3D12Device> device);
        [[nodiscard]] Nv12Frame acquire(
            winrt::com_ptr<ID3D11Texture2D> const &cameraTexture);
        [[nodiscard]] std::size_t cachedTextureCount() const
        {
            return slots.size();
        };
        void clear()
        {
            slots.clear();
        };

    private:
        struct Slot
        {
            winrt::com_ptr<ID3D11Texture2D> source;
            winrt::com_ptr<ID3D12Resource> resource;
            winrt::com_ptr<ID3D12DescriptorHeap> srvHeap;
        };
        winrt::com_ptr<ID3D12Device> d3d12Device;
        std::unordered_map<ID3D11Texture2D *, Slot> slots;

        Slot const &importTexture(winrt::com_ptr<ID3D11Texture2D> const &cameraTexture);
        void createSrvHeap(ID3D12Resource *nv12,
                           winrt::com_ptr<ID3D12DescriptorHeap> &heap) const;
    };
}