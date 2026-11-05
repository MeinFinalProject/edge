#include <chrono>
#include <iostream>
#include <winrt/base.h>
#include "camera/camera.hpp"
#include "common/paths.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/nv12_bridge.hpp"
#include "inference/scrfd_detector.hpp"
#include "inference/liveness_classifier.hpp"
#include "tracking/byte_tracker.hpp"
using namespace std::chrono_literals;

int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        using namespace FaceRuntime;
        Gpu::GpuContext gpu;
        Gpu::Nv12Bridge bridge(gpu.deviceRef());
        CameraConfig cameraConfig; // 1280x720 @30: satu-satunya sumber ukuran frame
        Camera camera(cameraConfig);
        camera.start();

        Inference::ScrfdDetector detector(
            gpu, Common::ExecutableDirectory() / "models" / "insightface" / "buffalo_l" / "det_10g.onnx", cameraConfig);
        Inference::LivenessClassifier liveness(
            gpu, Common::ExecutableDirectory() / "models" / "insightface" / "addons" / "liveness.onnx", cameraConfig);

        detector.printModelSignature();
        Tracking::ByteTracker tracker;
        for (int i = 0; i < 90; ++i)
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
                break;
            auto texture = TextureOf(frame);
            auto const nv12 = bridge.acquire(texture);

            auto faces = detector.detect(nv12);
            auto tracked = tracker.update(faces);
            if (i % 10 == 0)
                for (auto const &t : tracked.active)
                    std::cout << "  track " << t.trackId << " skor=" << t.score
                              << " det#" << t.detectionIndex << "\n";
            for (int id : tracked.removedIds)
                std::cout << "  track " << id << " dibuang\n";
            gpu.waitIdle(); // pastikan GPU selesai membaca tekstur kamera
            frame.Close();  // baru sekarang tiket dilepas
            if (i % 10 == 0)
            {
                std::cout << "frame " << i << ": " << faces.size() << " wajah\n";
                for (auto &f : faces)
                {
                    auto const &lm = f.landmarks;
                    // Sanity landmark: mata kiri.x < mata kanan.x ; hidung di antara mata ;
                    // mulut di bawah hidung.
                    const bool ok = lm[0].x < lm[1].x && lm[2].y > lm[0].y && lm[3].y > lm[2].y;
                    std::cout << "  skor=" << f.score
                              << " box=(" << f.box.x1 << "," << f.box.y1 << ")-("
                              << f.box.x2 << "," << f.box.y2 << ")"
                              << " mataKiri=(" << lm[0].x << "," << lm[0].y << ")"
                              << " urutanLandmark=" << (ok ? "OK" : "SALAH") << "\n";

                    const float liveProb = liveness.realProbability(nv12, f);
                    if (i % 10 == 0)
                        std::cout << "  probabilitas live=" << liveProb << "\n";
                }
            }
        }
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
