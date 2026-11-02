#pragma once
#include <d3d12.h>
#include <winrt/base.h>

namespace FaceRuntime::Gpu
{
    // "Pindahkan resource ini dari peran A ke peran B."
    inline D3D12_RESOURCE_BARRIER TransitionBarrier(
        ID3D12Resource *resource,
        D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        return barrier;
    }

    // "Semua penulisan UAV ke resource ini harus tuntas sebelum perintah berikutnya membacanya."
    inline D3D12_RESOURCE_BARRIER UavBarrier(
        ID3D12Resource *resource)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.UAV.pResource = resource;
        return barrier;
    }

    // Membuat buffer (pita memori datar) di heap pilihan.
    //   DEFAULT  = VRAM, tercepat, CPU tidak boleh menyentuh(tensor, counter)
    //   UPLOAD   = CPU tulis -> GPU baca(sumber nol untuk reset counter)
    //   READBACK = GPU tulis -> CPU baca(mengambil hasil)
    inline winrt::com_ptr<ID3D12Resource> CreateBuffer(
        ID3D12Device *device,
        UINT64 sizeInBytes,
        D3D12_HEAP_TYPE heapType,
        D3D12_RESOURCE_STATES initialState,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
    {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = heapType;
        heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Alignment = 0;
        desc.Width = sizeInBytes;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;

        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;

        winrt::com_ptr<ID3D12Resource> resource;
        auto hr = device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initialState,
            nullptr,
            IID_PPV_ARGS(resource.put()));
        if (FAILED(hr))
        {
            throw std::runtime_error("Failed to create D3D12 resource");
        }
        return resource;
    }
}