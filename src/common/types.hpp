#pragma once
#include <algorithm>
#include <array>
#include "point2f.hpp"

namespace FaceRuntime::Common
{
    // Kotak dengan format sudut kiri-atas (x1,y1)
    // dan kanan-bawah (x2,y2), satuan piksel.
    struct BoxXyxy
    {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;

        [[nodiscard]] float width() const
        {
            return x2 - x1;
        }
        [[nodiscard]] float height() const
        {
            return y2 - y1;
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