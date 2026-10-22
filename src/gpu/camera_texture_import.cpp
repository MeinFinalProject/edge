#include "camera_texture_import.hpp"
#include <stdexcept>
#include <dxgi1_4.h>
#include <iostream>

namespace FaceRuntime::Gpu
{
    CameraTextureImporter::CameraTextureImporter(
        winrt::com_ptr<ID3D11Device> d3d11Device,
        winrt::com_ptr<ID3D12Device> d3d12Device,
        winrt::com_ptr<ID3D11On12Device> d3d11On12Device)
        : d3d11Device(std::move(d3d11Device)),
          d3d12Device(std::move(d3d12Device)),
          d3d11On12Device(std::move(d3d11On12Device))
    {
        if (!this->d3d11Device ||
            !this->d3d12Device ||
            !this->d3d11On12Device)
        {
            throw std::invalid_argument("One or more device pointers are null.");
        }
    }

    WrappedAccess CameraTextureImporter::access(
        ImportedViews const &views)
    {
        if (!d3d11On12Device)
        {
            throw std::runtime_error(
                "CameraTextureImporter: D3D11On12 device null.");
        }

        if (!views.wrappedResource)
        {
            throw std::runtime_error(
                "CameraTextureImporter: wrapped resource null.");
        }

        WrappedAccess access(
            d3d11On12Device,
            views.wrappedResource.get());

        return access;
    }

    ImportedViews CameraTextureImporter::importFrame(winrt::com_ptr<ID3D11Texture2D> const &texture)
    {
        if (!texture)
        {
            throw std::invalid_argument("Texture pointer is null.");
        }

        std::cout << "Import tekstur berhasil\n";

        std::scoped_lock lock(cacheMutex);
        auto it = viewCache.find(texture.get());
        if (it != viewCache.end())
        {
            std::cout << "Menggunakan cache\n";
            return it->second;
        }
        std::cout << "Membuat cache baru\n";

        winrt::com_ptr<IDXGIResource1> dxgiResource;
        winrt::check_hresult(texture.as(IID_PPV_ARGS(&dxgiResource)));

        std::cout << "Membuat IDXGIResource1 berhasil\n";

        HANDLE sharedHandle = nullptr;
        winrt::check_hresult(dxgiResource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, &sharedHandle));

        std::cout << "Membuat shared handle berhasil\n";

        winrt::com_ptr<ID3D12Resource> d3d12Resource;
        winrt::check_hresult(d3d12Device->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(&d3d12Resource)));

        CloseHandle(sharedHandle);

        std::cout << "Membuka shared handle berhasil di D3D12\n";

        D3D11_RESOURCE_FLAGS d3d11Flags = {};
        d3d11Flags.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        winrt::com_ptr<ID3D11Resource> wrappedResource;
        winrt::check_hresult(d3d11On12Device->CreateWrappedResource(
            d3d12Resource.get(),
            &d3d11Flags,
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
            IID_PPV_ARGS(&wrappedResource)));

        std::cout << "Membuat wrapped resource berhasil\n";

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDescY = {};
        srvDescY.Format = DXGI_FORMAT_R8_UNORM;
        srvDescY.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDescY.Texture2D.MipLevels = 1;
        srvDescY.Texture2D.MostDetailedMip = 0;

        winrt::com_ptr<ID3D11ShaderResourceView> lumaView;
        winrt::check_hresult(d3d11Device->CreateShaderResourceView(wrappedResource.get(), &srvDescY, lumaView.put()));

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDescUV = {};
        srvDescUV.Format = DXGI_FORMAT_R8G8_UNORM;
        srvDescUV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDescUV.Texture2D.MipLevels = 1;
        srvDescUV.Texture2D.MostDetailedMip = 0;

        winrt::com_ptr<ID3D11ShaderResourceView> chromaView;
        winrt::check_hresult(d3d11Device->CreateShaderResourceView(wrappedResource.get(), &srvDescUV, chromaView.put()));

        std::cout << "Membuat shader resource view untuk Y dan UV berhasil\n";

        ImportedViews views{std::move(wrappedResource), std::move(lumaView), std::move(chromaView)};

        viewCache[texture.get()] = views;
        return views;
    }
}