#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <winrt/base.h>
#include "camera/camera.hpp"
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Media.Capture.Frames.h>
#include "gpu/gpu_context.hpp"
#include "gpu/nv12_bridge.hpp"
#include "preprocess/preprocess_contracts.hpp"

using namespace std::chrono_literals;
namespace chr = std::chrono;
using namespace FaceRuntime;

namespace
{
    using CameraFrame = winrt::Windows::Media::Capture::Frames::MediaFrameReference;

    constexpr int kWarmupSeconds = 3;      // beri kamera waktu menstabilkan eksposur
    constexpr int kPhaseSeconds = 5;       // lama tiap fase fps
    constexpr int kInvarianceFrames = 300; // jumlah frame tes invarian
    constexpr double kTolerance = 0.85;    // lulus bila >= 85% dari yang diharapkan

    // Menahan tiket tepat `ms` milidetik (spin; tanpa granularitas timer Windows 15,6 ms).
    void HoldFor(chr::milliseconds ms)
    {
        const auto end = chr::steady_clock::now() + ms;
        while (chr::steady_clock::now() < end)
        {
            std::this_thread::yield();
        }
    }

    // Timestamp penangkapan frame (ms), bila disediakan kamera (nullable).
    std::optional<double> CaptureTimeMs(CameraFrame const &frame)
    {
        if (auto t = frame.SystemRelativeTime())
        {
            return static_cast<double>(t.Value().count()) / 10000.0; // TimeSpan = 100 ns
        }
        return std::nullopt;
    }

    double MeanAbsDiff(std::vector<float> const &a, std::vector<float> const &b)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        }
        return a.empty() ? 0.0 : sum / static_cast<double>(a.size());
    }

    struct PhaseStats
    {
        int frames = 0;
        int timeouts = 0;
        double fps = 0.0;
        long long avgWaitUs = 0;
        double camIntervalAvgMs = 0.0; // dari timestamp frame
        double camIntervalMaxMs = 0.0;
    };

    // Menahan tiket `hold` per frame selama `seconds` detik.
    PhaseStats RunPhase(Camera &camera, Gpu::Nv12Bridge &bridge, chr::milliseconds hold, int seconds)
    {
        PhaseStats s;
        long long waitUs = 0;
        std::optional<double> prevT;
        double sumInterval = 0.0;
        int intervals = 0;

        const auto start = chr::steady_clock::now();
        while (chr::steady_clock::now() - start < chr::seconds(seconds))
        {
            const auto w0 = chr::steady_clock::now();
            auto frame = camera.waitForFrame(2000ms);
            waitUs += chr::duration_cast<chr::microseconds>(chr::steady_clock::now() - w0).count();
            if (!frame)
            {
                ++s.timeouts;
                continue;
            }
            auto texture = TextureOf(frame);
            auto const nv12 = bridge.acquire(texture);
            (void)nv12;

            if (auto t = CaptureTimeMs(frame))
            {
                if (prevT)
                {
                    const double d = *t - *prevT;
                    sumInterval += d;
                    ++intervals;
                    s.camIntervalMaxMs = std::max(s.camIntervalMaxMs, d);
                }
                prevT = t;
            }

            HoldFor(hold);
            frame.Close();
            ++s.frames;
        }
        s.fps = s.frames / static_cast<double>(seconds);
        s.avgWaitUs = s.frames ? waitUs / s.frames : 0;
        s.camIntervalAvgMs = intervals ? sumInterval / intervals : 0.0;
        return s;
    }

    char const *Verdict(bool ok) { return ok ? "PASS    " : "PERIKSA "; }

    int Run()
    {
        CameraConfig cameraConfig; // 1280x720 @30
        Gpu::GpuContext gpu;
        Gpu::Nv12Bridge bridge(gpu.deviceRef());
        Camera camera(cameraConfig);
        camera.start(); // mencetak diagnostik kontrol kamera ("[Camera] ...")

        std::cout << std::fixed << std::setprecision(1);

        // ---- Pemanasan -------------------------------------------------------------------
        {
            const auto start = chr::steady_clock::now();
            while (chr::steady_clock::now() - start < chr::seconds(kWarmupSeconds))
            {
                auto frame = camera.waitForFrame(2000ms);
                if (frame)
                {
                    frame.Close();
                }
            }
        }

        // ---- A/B/C. fps: baseline lalu tahan tiket 33 ms dan 100 ms ----------------------
        std::cout << "\n[fps] tiap fase " << kPhaseSeconds << " detik...\n";
        const PhaseStats base = RunPhase(camera, bridge, 0ms, kPhaseSeconds);
        const double camFps = base.camIntervalAvgMs > 0.0 ? 1000.0 / base.camIntervalAvgMs : base.fps;
        std::cout << "  tahan   0 ms : " << base.fps << " frame/s | selang kamera rata-rata "
                  << base.camIntervalAvgMs << " ms, maks " << base.camIntervalMaxMs
                  << " ms | timeout " << base.timeouts << "\n";

        const PhaseStats h33 = RunPhase(camera, bridge, 33ms, kPhaseSeconds);
        std::cout << "  tahan  33 ms : " << h33.fps << " frame/s | menunggu "
                  << h33.avgWaitUs / 1000.0 << " ms | timeout " << h33.timeouts << "\n";

        const PhaseStats h100 = RunPhase(camera, bridge, 100ms, kPhaseSeconds);
        std::cout << "  tahan 100 ms : " << h100.fps << " frame/s | menunggu "
                  << h100.avgWaitUs / 1000.0 << " ms | timeout " << h100.timeouts << "\n";

        // ---- D. kesegaran frame setelah tahan lama ---------------------------------------
        std::cout << "\n[kesegaran] menahan satu tiket 10 detik...\n";
        constexpr auto kLongHold = 10s;
        std::optional<double> freshnessDeltaMs;
        {
            auto a = camera.waitForFrame(2000ms);
            const auto ta = a ? CaptureTimeMs(a) : std::nullopt;
            if (a && ta)
            {
                auto texture = TextureOf(a);
                auto const nv12 = bridge.acquire(texture);
                (void)nv12;
                HoldFor(kLongHold);
                a.Close();
                auto b = camera.waitForFrame(2000ms);
                if (b)
                {
                    if (auto tb = CaptureTimeMs(b))
                    {
                        freshnessDeltaMs = *tb - *ta;
                    }
                    b.Close();
                }
            }
            else if (a)
            {
                a.Close();
            }
        }
        if (freshnessDeltaMs)
        {
            std::cout << "  selisih timestamp = " << *freshnessDeltaMs << " ms\n";
        }
        else
        {
            std::cout << "  timestamp tidak tersedia\n";
        }

        // ---- E. tes invarian: isi frame yang dipegang tidak boleh berubah -----------------
        std::cout << "\n[invarian] " << kInvarianceFrames << " frame (GPU relatif menganggur)...\n";
        Preprocess::ScrfdPreprocessor scrfd(gpu, cameraConfig);
        int changed = 0;
        int checked = 0;
        double motionSum = 0.0;
        std::vector<float> prevFirst;
        while (checked < kInvarianceFrames)
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
            {
                std::cout << "  timeout menunggu frame; tes dihentikan\n";
                break;
            }
            auto texture = TextureOf(frame);
            auto const nv12 = bridge.acquire(texture);

            scrfd.preprocess(nv12);
            const auto first = scrfd.debugReadback(); // baca #1: segera setelah acquire
            std::this_thread::sleep_for(15ms);        // tiket masih dipegang
            scrfd.preprocess(nv12);
            const auto second = scrfd.debugReadback(); // baca #2: harus IDENTIK

            if (first != second)
            {
                ++changed;
            }
            if (!prevFirst.empty())
            {
                motionSum += MeanAbsDiff(first, prevFirst);
            }
            prevFirst = first;
            frame.Close();
            ++checked;
        }
        const double motion = checked > 1 ? motionSum / (checked - 1) : 0.0;

        camera.stop();

        // ---- Ringkasan ----------------------------------------------------------------------
        std::cout << "\n================== RINGKASAN ==================\n";
        std::cout << std::setprecision(3);
        std::cout << "[INFO]     kamera sebenarnya: " << camFps << " fps (target "
                  << cameraConfig.fps << ")\n";

        std::cout << std::setprecision(1);
        std::cout << "[" << Verdict(camFps >= kTolerance * cameraConfig.fps) << "] fps kamera >= "
                  << kTolerance * cameraConfig.fps << " (terukur " << camFps << ")\n";

        auto reportHold = [&](char const *label, PhaseStats const &p, double holdMs)
        {
            const double expected = std::min(camFps, 1000.0 / holdMs);
            const bool ok = p.timeouts == 0 && p.fps >= kTolerance * expected;
            std::cout << "[" << Verdict(ok) << "] " << label << ": " << p.fps
                      << " frame/s (harapan >= " << kTolerance * expected << ")\n";
        };
        reportHold("tahan  33 ms tidak menahan kamera", h33, 33.0);
        reportHold("tahan 100 ms tidak menahan kamera", h100, 100.0);

        if (freshnessDeltaMs)
        {
            const double heldMs = chr::duration_cast<chr::milliseconds>(kLongHold).count();
            const char *note = *freshnessDeltaMs >= 0.9 * heldMs  ? "kamera terus mengirim saat tiket ditahan"
                               : *freshnessDeltaMs < 0.1 * heldMs ? "kamera BERHENTI saat tiket ditahan"
                                                                  : "tidak konklusif";
            std::cout << "[" << Verdict(*freshnessDeltaMs >= 0.9 * heldMs) << "] kesegaran setelah tahan "
                      << heldMs / 1000.0 << " s: selisih " << *freshnessDeltaMs << " ms (" << note << ")\n";
        }
        else
        {
            std::cout << "[N/A     ] kesegaran: kamera tidak menyediakan timestamp\n";
        }

        std::cout << "[" << (changed == 0 ? "PASS    " : "GAGAL   ") << "] invarian: " << changed
                  << " dari " << checked << " frame berubah saat dipegang\n";
        std::cout << std::setprecision(4);
        std::cout << "[INFO]     kekuatan tes invarian (rata-rata perubahan antar-frame): " << motion
                  << "  (mendekati 0 = adegan statis/gelap, tes lemah; gerakkan tangan atau tambah cahaya)\n";
        std::cout << "===============================================\n";
        return 0;
    }
}

int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        return Run();
    }
    catch (winrt::hresult_error const &error)
    {
        std::cerr << "HRESULT 0x" << std::hex << static_cast<unsigned long>(error.code().value)
                  << std::dec << ": " << winrt::to_string(error.message()) << "\n";
    }
    catch (std::exception const &error)
    {
        std::cerr << "Error: " << error.what() << "\n";
    }
    return 1;
}