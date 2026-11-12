#pragma once
#include <algorithm>
#include <array>

namespace FaceRuntime::Common
{
    struct Point2f
    {
        float x = 0.0F;
        float y = 0.0F;
    };

    // Kotak dengan format sudut kiri-atas (x1,y1)
    // dan kanan-bawah (x2,y2), satuan piksel.
    struct BoxXyxy
    {
        Point2f topLeft{};
        Point2f bottomRight{};

        [[nodiscard]] float width() const
        {
            return bottomRight.x - topLeft.x;
        }
        [[nodiscard]] float height() const
        {
            return bottomRight.y - topLeft.y;
        }
        [[nodiscard]] float area() const
        {
            return std::max(0.0f, width()) * std::max(0.0f, height());
        }
    };
    // Satu wajah hasil SCRFD. SEMUA koordinat sudah dalam piksel FRAME KAMERA
    // (bukan ruang 640x640), jadi bisa langsung dipakai alignment dan tracker.
    struct FaceDetection
    {
        BoxXyxy box;
        float score = 0.0f;
        // Urutan wajib: mata kiri, mata kanan, hidung, mulut kiri, mulut kanan
        // ("kiri/kanan" menurut posisi di gambar, bukan menurut orangnya).
        std::array<Point2f, 5> landmarks;
    };
}