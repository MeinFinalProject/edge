#include "nv12_bridge.hpp"
#include <dxgi1_2.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace FaceRuntime::Gpu
{
    namespace
    {
        [[noreturn]] void FailWithHresult(char const *what, HRESULT hr)
        {
            std::ostringstream message;
            message << "Nv12Bridge: " << what << " gagal (HRESULT 0x" << std::hex
                    << static_cast<unsigned long>(hr) << ").";
            throw std::runtime_error(message.str());
        }

        // Kamera dan D3D12 harus di GPU FISIK yang sama (berbagi memori lintas GPU tidak mungkin).
        // Ini bukan sinkronisasi; hanya memberi pesan error yang jelas untuk laptop dua-GPU,
        // alih-alih HRESULT misterius dari OpenSharedHandle.
        void EnsureSameAdapter(ID3D11Texture2D *texture, ID3D12Device *d3d12Device)
        {
            winrt::com_ptr<ID3D11Device> device;
            texture->GetDevice(device.put());

            auto dxgiDevice = device.as<IDXGIDevice>();
            winrt::com_ptr<IDXGIAdapter> adapter;
            winrt::check_hresult(dxgiDevice->GetAdapter(adapter.put()));
            DXGI_ADAPTER_DESC adapterDesc{};
            winrt::check_hresult(adapter->GetDesc(&adapterDesc));

            const LUID d3d12Luid = d3d12Device->GetAdapterLuid();
            if (adapterDesc.AdapterLuid.LowPart != d3d12Luid.LowPart ||
                adapterDesc.AdapterLuid.HighPart != d3d12Luid.HighPart)
            {
                throw std::runtime_error(
                    "Kamera dan D3D12 berada di GPU berbeda (laptop dengan 2 GPU?). "
                    "Lihat bagian Troubleshooting panduan.");
            }
        }
    }

    Nv12Bridge::Nv12Bridge(
        winrt::com_ptr<ID3D12Device> device)
        : d3d12Device(std::move(device))
    {
        if (!d3d12Device)
        {
            throw std::invalid_argument("Nv12Bridge: d3d12Device null.");
        }
    }

    Nv12Frame Nv12Bridge::acquire(
        winrt::com_ptr<ID3D11Texture2D> const &cameraTexture)
    {
        if (!cameraTexture)
        {
            throw std::invalid_argument("Nv12Bridge::acquire: cameraTexture null.");
        }

        Slot const *slot = nullptr;
        if (auto it = slots.find(cameraTexture.get());
            it != slots.end())
        {
            slot = &it->second;
        }
        else
        {
            slot = &importTexture(cameraTexture);
        }

        return Nv12Frame{slot->resource.get(), slot->srvHeap.get()};
    }

    Nv12Bridge::Slot const &Nv12Bridge::importTexture(winrt::com_ptr<ID3D11Texture2D> const &cameraTexture)
    {
        D3D11_TEXTURE2D_DESC desc{};
        cameraTexture->GetDesc(&desc);

        std::ostringstream why;
        why << std::hex;
        if (desc.Format != DXGI_FORMAT_NV12)
        {
            why << "Nv12Bridge: tekstur kamera bukan NV12 (format DXGI = " << std::dec
                << static_cast<int>(desc.Format) << ").";
            throw std::runtime_error(why.str());
        }
        if (desc.ArraySize != 1)
        {
            why << "Nv12Bridge: tekstur kamera adalah texture array (ArraySize = " << std::dec
                << desc.ArraySize << "). Butuh index subresource yang benar";
            throw std::runtime_error(why.str());
        }
        if ((desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0)
        {
            why << "Nv12Bridge: tekstur kamera tidak punya bind SHADER_RESOURCE (bind=0x"
                << desc.BindFlags << "), sehingga D3D12 tidak bisa membuat SRV.";
            throw std::runtime_error(why.str());
        }
        if ((desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE) == 0)
        {
            why << "Nv12Bridge: tekstur kamera tidak ditandai SHARED_NTHANDLE (misc=0x"
                << desc.MiscFlags << "), jadi tidak bisa dibagikan ke D3D12. "
                << "Zero-copy tidak mungkin pada kamera/driver ini.";
            throw std::runtime_error(why.str());
        }

        EnsureSameAdapter(cameraTexture.get(), d3d12Device.get());

        // 2) Jadikan tekstur kamera sebuah NT handle (nomor tiket antar-API) lalu buka di D3D12:
        //    memori fisik yang SAMA, nol salinan.
        auto dxgiResource = cameraTexture.as<IDXGIResource1>();
        winrt::handle sharedHandle; // otomatis CloseHandle di akhir fungsi
        HRESULT hr = dxgiResource->CreateSharedHandle(
            nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, sharedHandle.put());
        if (FAILED(hr))
        {
            FailWithHresult("CreateSharedHandle pada tekstur kamera", hr);
        }

        Slot slot;
        slot.source = cameraTexture;
        hr = d3d12Device->OpenSharedHandle(sharedHandle.get(), IID_PPV_ARGS(slot.resource.put()));
        if (FAILED(hr))
        {
            FailWithHresult("OpenSharedHandle di D3D12", hr);
        }

        createSrvHeap(slot.resource.get(), slot.srvHeap);

        return slots.emplace(cameraTexture.get(), std::move(slot)).first->second;
    }

    void Nv12Bridge::createSrvHeap(ID3D12Resource *nv12, winrt::com_ptr<ID3D12DescriptorHeap> &heap) const
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = 2;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heapDesc.NodeMask = 0;
        winrt::check_hresult(d3d12Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(heap.put())));

        const UINT increment = d3d12Device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE slot = heap->GetCPUDescriptorHandleForHeapStart();

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Texture2D.ResourceMinLODClamp = 0.0F;

        // Slot 0: plane Y (luma) -> Texture2D<float> di t0
        srvDesc.Format = DXGI_FORMAT_R8_UNORM;
        srvDesc.Texture2D.PlaneSlice = 0;
        d3d12Device->CreateShaderResourceView(nv12, &srvDesc, slot);

        // Slot 1: plane UV (chroma) -> Texture2D<float2> di t1 (x = U, y = V)
        slot.ptr += increment;
        srvDesc.Format = DXGI_FORMAT_R8G8_UNORM;
        srvDesc.Texture2D.PlaneSlice = 1;
        d3d12Device->CreateShaderResourceView(nv12, &srvDesc, slot);
    }
}
