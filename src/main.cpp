#include "camera/camera.hpp"
#include "gpu/gpu_context.hpp"
#include "preprocess/preprocess_contracts.hpp"

#include <chrono>
#include <iostream>

int main()
{
    FaceRuntime::Gpu::GpuContext gpuContext;
    FaceRuntime::Camera camera;
    camera.start();

    auto firstTexture = camera.waitForFrame(
        std::chrono::seconds(2));
    if (!firstTexture)
    {
        std::cerr << "timeout menunggu frame\n";
        return 1;
    }

    std::cout << " diterima, texture: " << firstTexture.get() << "\n";

    D3D11_TEXTURE2D_DESC desc{};

    firstTexture->GetDesc(&desc);

    std::cout << "height dan width texture: "
              << desc.Height << "x" << desc.Width << " format: " << (desc.Format == DXGI_FORMAT_NV12 ? "NV12" : "Bukan NV12") << "\n";

    FaceRuntime::Preprocess::ScrfdGpuServices services{
        gpuContext.getCameraTextureImporter(),
        gpuContext.getD3D12Device(),
        gpuContext.getCommandQueue(),
        gpuContext.getLearningDevice()};

    FaceRuntime::Preprocess::ScrfdPreprocessor scrfd(
        firstTexture,
        std::move(services));

    std::cout << "SCRFD preprocessor berhasil dibuat\n";

    for (int i = 0; i < 30; ++i)
    {
        auto texture = camera.waitForFrame(
            std::chrono::seconds(2));

        if (!texture)
        {
            std::cerr << "timeout menunggu frame\n";
            break;
        }

        try
        {
            auto tensor = scrfd.preprocess(
                std::move(texture));

            if (!tensor)
            {
                std::cerr
                    << "SCRFD preprocess selesai, "
                       "tetapi tensor masih kosong\n";
                break;
            }

            std::cout << "SCRFD tensor berhasil dibuat\n";
        }
        catch (winrt::hresult_error const &error)
        {
            std::cerr
                << "SCRFD preprocess gagal dengan HRESULT: 0x"
                << std::hex
                << static_cast<unsigned long>(
                       error.code().value)
                << std::dec
                << "\n";
            break;
        }
        catch (std::exception const &error)
        {
            std::cerr
                << "SCRFD preprocess gagal: "
                << error.what()
                << "\n";
            break;
        }
    }
    std::cout << "menghentikan kamera\n";

    camera.stop();
}
