#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <winrt/base.h>
#include "camera/camera.hpp"
#include "common/paths.hpp"
#include "gpu/gpu_context.hpp"
#include "pipeline/pipeline.hpp"

using namespace std::chrono_literals;
using namespace FaceRuntime;

namespace
{
    void PrintUsage()
    {
        std::cout << "Pemakaian:\n"
                     "  FaceRuntime.exe selftest [folder=.]         UJI AKHIR sekali jalan (+ tulis scrfd.bin, pad.bin, arc.bin)\n"
                     "  FaceRuntime.exe enroll <nama> [sampel=5]    daftarkan wajah ke galeri (nama tanpa spasi)\n"
                     "  FaceRuntime.exe run [detik=60] [trace]      jalankan pipeline penuh\n"
                     "  FaceRuntime.exe record <0|1> <file.csv> [frame=150]   rekam skor liveness (1 = asli, 0 = spoof)\n";
    }

    std::filesystem::path GalleryPath()
    {
        return Common::ExecutableDirectory() / "gallery.txt";
    }

    // UJI AKHIR: satu frame berisi wajahmu, seluruh rantai diperiksa sekaligus.
    int SelfTest(Camera &camera, Pipeline::FacePipeline &pipeline, std::filesystem::path const &folder)
    {
        pipeline.printModelSignatures();
        std::cout << "\nHadapkan wajahmu ke kamera (tegak, di tengah, jarak lengan)...\n";
        for (int i = 0; i < 300; ++i)
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
            {
                std::cout << "kamera tidak mengirim frame.\n";
                return 2;
            }
            auto report = pipeline.selfTest(std::move(frame), folder);
            if (!report.faceFound)
            {
                if (i % 30 == 0)
                    std::cout << "  belum ada wajah yang layak ...\n";
                continue;
            }

            int failed = 0;
            std::cout << "\n";
            for (auto const &item : report.items)
            {
                std::cout << (item.ok ? "[PASS] " : "[GAGAL] ") << item.name << "\n         " << item.detail << "\n";
                failed += item.ok ? 0 : 1;
            }
            std::cout << "\n"
                      << (failed == 0 ? "HASIL: SEMUA PASS" : "HASIL: ADA YANG GAGAL (lihat tabel 'Kalau gagal')") << "\n";
            std::cout << "Tensor ditulis ke " << std::filesystem::absolute(folder).string()
                      << ". Untuk melihatnya sebagai gambar:\n"
                         "  python tools\\tensor_to_png.py scrfd.bin 640 640 scrfd.png scrfd\n"
                         "  python tools\\tensor_to_png.py pad.bin 80 80 pad.png unit\n"
                         "  python tools\\tensor_to_png.py arc.bin 112 112 arc.png arcface\n";
            return failed == 0 ? 0 : 1;
        }
        std::cout << "tidak ada wajah yang layak dalam 300 frame.\n";
        return 2;
    }

    void Enroll(Camera &camera, Pipeline::FacePipeline &pipeline, std::string const &name, int samples)
    {
        std::cout << "Mendaftarkan '" << name << "': hadap kamera, variasikan pose sedikit. Target " << samples << " sampel.\n";
        int added = 0;
        int attempts = 0;
        while (added < samples && attempts < 600)
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
                break;
            if (++attempts % 6 != 0)
            {
                frame.Close(); // frame dilewati (belum dipakai GPU), kembalikan tiketnya ke kamera
                continue;
            }
            auto result = pipeline.enroll(std::move(frame), name);
            if (result.added)
                std::cout << "  sampel " << ++added << "/" << samples << " ok\n";
            else if (attempts % 30 == 0)
                std::cout << "  belum bisa: " << result.message << "\n";
        }
        if (added == 0)
        {
            std::cout << "tidak ada sampel yang berhasil; galeri tidak diubah.\n";
            return;
        }
        pipeline.gallery().save(GalleryPath());
        std::cout << "galeri disimpan ke " << GalleryPath().string() << " (" << pipeline.gallery().size() << " entri)\n";
    }

    // Jalankan pipeline penuh dan cetak hanya PERUBAHAN status tiap track.
    void Run(Camera &camera, Pipeline::FacePipeline &pipeline, int seconds)
    {
        std::map<int, Pipeline::TrackStatus> lastStatus;
        const auto start = std::chrono::steady_clock::now();
        int frames = 0;
        while (std::chrono::steady_clock::now() - start < std::chrono::seconds(seconds))
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
            {
                std::cout << "kamera tidak mengirim frame selama 2 detik, berhenti.\n";
                break;
            }
            ++frames;
            for (auto const &r : pipeline.process(std::move(frame)))
            {
                auto it = lastStatus.find(r.trackId);
                if (it != lastStatus.end() && it->second == r.status)
                    continue;
                lastStatus[r.trackId] = r.status;
                std::cout << "track " << r.trackId << " -> " << Pipeline::ToString(r.status);
                if (r.status == Pipeline::TrackStatus::Identified)
                    std::cout << " " << r.name << " (sim=" << r.similarity << " margin=" << r.margin << ")";
                if (r.livenessObservations > 0)
                    std::cout << "  [S=" << r.sprtSum << " n=" << r.livenessObservations << "]";
                std::cout << "\n";
            }
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << frames << " frame dalam " << elapsed << " detik (" << frames / elapsed << " fps)\n";
    }

    void Record(Camera &camera, Pipeline::FacePipeline &pipeline, int label, std::string const &csv, int frames)
    {
        pipeline.startLivenessRecording(csv, label);
        std::cout << "Merekam " << frames << " frame ke " << csv << " (label " << label << "). Hadapkan "
                  << (label == 1 ? "wajah asli" : "layar HP / foto cetak") << " ke kamera.\n";
        for (int i = 0; i < frames; ++i)
        {
            auto frame = camera.waitForFrame(2000ms);
            if (!frame)
                break;
            (void)pipeline.process(std::move(frame));
        }
        pipeline.stopLivenessRecording();
        std::cout << "selesai.\n";
    }
}

int main(int argc, char **argv)
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        const std::string mode = argc > 1 ? argv[1] : "run";
        if ((mode != "selftest" && mode != "enroll" && mode != "run" && mode != "record") ||
            (mode == "enroll" && argc < 3) || (mode == "record" && argc < 4))
        {
            PrintUsage();
            return 1;
        }

        Gpu::GpuContext gpu;
        CameraConfig cameraConfig; // 1280x720 @30: satu-satunya sumber ukuran frame
        Camera camera(cameraConfig);
        camera.start();

        Pipeline::FacePipelineConfig config;
        // ----- Hasil kalibrasi dan tuning (Tahap 8) diisi di sini -----
        // config.decision.sprt.weight = 0.5F;
        // config.decision.acceptSimilarity = 0.40F;
        // config.decision.minMargin = 0.05F;
        if (mode == "run" && argc > 3 && std::string(argv[3]) == "trace")
            config.trace = &std::cout;
        if (mode == "record")
        {
            config.trace = &std::cout;      // lihat tiap observasi yang direkam
            config.observeEveryNFrames = 1; // rekam TIAP frame; stride disapu nanti lewat `sprt_tool replay`
        }

        Pipeline::FacePipeline pipeline(
            gpu,
            Preprocess::FrameSize{cameraConfig.width, cameraConfig.height},
            Common::ExecutableDirectory() / "models",
            config);
        pipeline.gallery().load(GalleryPath());

        int exitCode = 0;
        if (mode == "selftest")
            exitCode = SelfTest(camera, pipeline, argc > 2 ? argv[2] : ".");
        else if (mode == "enroll")
            Enroll(camera, pipeline, argv[2], argc > 3 ? std::stoi(argv[3]) : 5);
        else if (mode == "run")
            Run(camera, pipeline, argc > 2 ? std::stoi(argv[2]) : 60);
        else
            Record(camera, pipeline, std::stoi(argv[2]), argv[3], argc > 4 ? std::stoi(argv[4]) : 150);

        camera.stop();
        return exitCode;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "HRESULT 0x" << std::hex << static_cast<unsigned long>(e.code().value)
                  << ": " << winrt::to_string(e.message()) << "\n";
        return 2;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << "\n";
        return 2;
    }
}
