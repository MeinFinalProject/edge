#include <chrono>
#include <iostream>
#include <winrt/base.h>
#include <winrt/Windows.Media.Capture.Frames.h>
#include <winrt/Windows.Foundation.h>
#include "camera/camera.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/nv12_bridge.hpp"
using namespace std::chrono_literals;

int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        using namespace FaceRuntime;
        CameraConfig cameraConfig;
        Gpu::GpuContext gpu;
        Gpu::Nv12Bridge bridge(gpu.deviceRef());
        Camera camera(cameraConfig);
        camera.start();

        const auto startTime = std::chrono::steady_clock::now();
        int processed = 0;
        int timeouts = 0;
        while (std::chrono::steady_clock::now() - startTime < 10s)
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
            {
                ++timeouts;
                continue;
            }
            auto texture = TextureOf(frame);
            auto const nv12 = bridge.acquire(texture);
            (void)nv12;
            std::this_thread::sleep_for(100ms); // tiket dipegang 100 ms, seperti process() yang berat
            frame.Close();
            ++processed;
        }
        std::cout << "diproses " << processed << " frame dalam 10 s (harapan: sekitar 91), "
                  << "timeout " << timeouts << ", tekstur di cache "
                  << bridge.cachedTextureCount() << "\n";

        camera.stop();
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "HRESULT 0x" << std::hex << static_cast<unsigned long>(e.code().value)
                  << ": " << winrt::to_string(e.message()) << "\n";
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << "\n";
    }
}
