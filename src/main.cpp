// #include <chrono>
// #include <iostream>
// #include <winrt/base.h>
// #include <winrt/Windows.AI.MachineLearning.h>
// #include <winrt/Windows.Foundation.h>
// #include "camera/camera.hpp"
// #include "common/debug_dump.hpp"
// #include "gpu/gpu_context.hpp"
// #include "gpu/nv12_bridge.hpp"
// #include "preprocess/preprocess_contracts.hpp"
// using namespace std::chrono_literals;
//
// int main()
// {
//     winrt::init_apartment(winrt::apartment_type::multi_threaded);
//     try
//     {
//         using namespace FaceRuntime;
//         Gpu::GpuContext gpu;
//         CameraConfig cameraConfig; // 1280x720 @30
//         Gpu::Nv12Bridge bridge(gpu.deviceRef());
//         Camera camera(cameraConfig);
//         camera.start();

//         // Beri kamera waktu menyesuaikan exposure; buang 29 frame, pakai frame ke-30.
//         for (int i = 0; i < 29; ++i)
//         {
//             auto warmup = camera.waitForFrame(2000ms);
//             if (!warmup)
//             {
//                 std::cerr << "timeout\n";
//                 return 1;
//             }
//             warmup.Close();
//         }
//         auto frame = camera.waitForFrame(2000ms);
//         if (!frame)
//         {
//             std::cerr << "timeout\n";
//             return 1;
//         }
//         auto texture = TextureOf(frame);
//         auto const nv12 = bridge.acquire(texture);

//         Preprocess::ScrfdPreprocessor scrfd(gpu, cameraConfig); // ukuran frame dari CameraConfig
//         scrfd.preprocess(nv12);
//         auto floats = scrfd.debugReadback(); // menunggu GPU selesai
//         frame.Close();                       // baru sekarang tiket dilepas
//         Common::WriteFloatsToFile("scrfd.bin", floats);
//         std::cout << "scrfd.bin ditulis (" << floats.size() << " float). "
//                   << "scaleX=" << scrfd.transform().scaleX
//                   << " resized=" << scrfd.transform().resizedWidth << "x"
//                   << scrfd.transform().resizedHeight << "\n";
//         camera.stop();
//     }
//     catch (winrt::hresult_error const &e)
//     {
//         std::cerr << "HRESULT 0x" << std::hex << static_cast<unsigned long>(e.code().value)
//                   << ": " << winrt::to_string(e.message()) << "\n";
//     }
//     catch (std::exception const &e)
//     {
//         std::cerr << e.what() << "\n";
//     }
// }
#include <chrono>
#include <iostream>
#include <thread>
#include <winrt/base.h>
#include "camera/camera.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/nv12_bridge.hpp"
#include "preprocess/preprocess_contracts.hpp"
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.h>
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
        Preprocess::ScrfdPreprocessor scrfd(gpu, cameraConfig);

        constexpr int kFrames = 1000;
        int changed = 0;
        for (int i = 0; i < kFrames; ++i)
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
            {
                std::cerr << "timeout\n";
                return 1;
            }
            auto texture = TextureOf(frame);
            auto const nv12 = bridge.acquire(texture);

            scrfd.preprocess(nv12);
            const auto first = scrfd.debugReadback(); // baca #1: segera setelah acquire

            std::this_thread::sleep_for(15ms); // tiket masih dipegang

            scrfd.preprocess(nv12);
            const auto second = scrfd.debugReadback(); // baca #2: isinya harus IDENTIK

            if (first != second)
                ++changed; // perbandingan float demi float
            frame.Close();
        }
        std::cout << "frame yang isinya berubah saat dipegang: " << changed
                  << " dari " << kFrames << "\n";
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
