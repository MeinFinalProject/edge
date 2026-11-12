#pragma once
// SEMUA yang dibutuhkan SCRFD (detektor wajah) ada di file ini dan scrfd.cpp:
//   1. kSpec        resep tensor input (640x640, normalisasi, tepi hitam)
//   2. Letterbox    cara memuat frame kamera ke 640x640 tanpa mengubah rasio
//   3. Detector     frame NV12 -> [GPU warp] -> [WinML SCRFD] -> [CPU decode + NMS] -> wajah
// Rincian decode anchor dan NMS ada di scrfd.cpp.
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>
#include "../common/types.hpp"
#include "../gpu/gpu_context.hpp"
#include "../gpu/nv12_bridge.hpp"
#include "../inference/onnx_model.hpp"
#include "../preprocess/warp.hpp"

namespace FaceRuntime::Scrfd
{
    // nilai = (rgb - 127,5) / 128.
    // Area di luar gambar (padding letterbox) berisi hitam,
    // yaitu (0 - 127,5) / 128 = -0,99609375 = normBias.
    inline constexpr Preprocess::TensorSpec kSpec{
        "scrfd",
        640,
        640,
        1.0F / 128.0F,
        -127.5F / 128.0F,
        Preprocess::OobMode::Fill};

    struct Options
    {
        float scoreThreshold = 0.10F; // threshold rendah untuk bytetrack
        float nmsThreshold = 0.40F;   // threshold NMS (non-max suppression), untuk menghapus kotak wajah yang tumpang tindih
    };

    // Letterbox kanan-bawah: gambar menempel di kiri-atas, sisanya hitam. Dinyatakan sebagai peta
    // afin tanpa rotasi dari piksel tensor ke piksel frame kamera, jadi memakai shader warp yang
    // sama dengan crop wajah.
    struct Letterbox
    {
        std::uint32_t resizedWidth = 0;
        std::uint32_t resizedHeight = 0;
        float scaleX = 1.0F;             // resizedWidth  / lebar frame kamera
        float scaleY = 1.0F;             // resizedHeight / tinggi frame kamera
        std::array<float, 6> dstToSrc{}; // [m00 m01 m02 m10 m11 m12] = [1/scaleX 0 0 0 1/scaleY 0]
    };

    // Satu wajah hasil SCRFD. SEMUA koordinat sudah dalam piksel FRAME KAMERA
    // (bukan ruang 640x640), jadi bisa langsung dipakai alignment dan tracker.
    struct FaceDetection
    {
        Common::BoxXyxy box;
        float score = 0.0f;
        // Urutan wajib: mata kiri, mata kanan, hidung, mulut kiri, mulut kanan
        // ("kiri/kanan" menurut posisi di gambar, bukan menurut orangnya).
        std::array<Common::Point2f, 5> landmarks;
    };

    [[nodiscard]] Letterbox ComputeLetterbox(
        Preprocess::FrameSize source);

    class Detector
    {
    public:
        Detector(Gpu::GpuContext &gpu,
                 Preprocess::WarpContext &warp,
                 std::filesystem::path const &modelPath,
                 Preprocess::FrameSize source,
                 Options options = {});

        // Wajah dalam koordinat piksel frame kamera, terurut menurut skor, setelah NMS.
        [[nodiscard]] std::vector<Common::FaceDetection> detect(
            Gpu::Nv12Frame const &frame);

        void printModelSignature() const
        {
            model.printSignature("SCRFD");
        }

        // Ukuran bagian tensor yang berisi gambar (sisanya padding hitam).
        [[nodiscard]] Preprocess::FrameSize contentSize() const
        {
            return {
                letterbox.resizedWidth,
                letterbox.resizedHeight};
        }

        // DEBUG: isi tensor input terakhir (640x640x3, lambat).
        [[nodiscard]] std::vector<float> debugReadback() const
        {
            return preprocessor.debugReadback();
        }

    private:
        Options options;
        Letterbox letterbox;
        Preprocess::WarpPreprocessor preprocessor;
        Inference::OnnxModel model;
    };
}