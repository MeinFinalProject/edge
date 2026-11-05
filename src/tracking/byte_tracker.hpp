#pragma once
#include <memory>
#include <vector>
#include "../common/types.hpp"

namespace FaceRuntime::Tracking
{
    struct ByteTrackerConfig
    {
        float highThreshold = 0.5F;             // deteksi >= ini = high score(asosiasi tahap 1)
        float lowThreshold = 0.1F;              // deteksi di (low, high) = low score(asosiasi tahap 2)
        float newTrackThreshold = 0.6F;         // minimal skor untuk membuat track baru
        float matchThreshold = 0.8F;            // IOU minimal untuk mengasosiasikan deteksi dengan track tahap 1
        float lowMatchThreshold = 0.5F;         // IOU minimal untuk mengasosiasikan deteksi dengan track tahap 2
        float unconfirmedMatchThreshold = 0.7F; // IOU minimal untuk mengasosiasikan deteksi dengan track yang belum dikonfirmasi
        int maxFrameAge = 30;                   // jumlah frame tanpa deteksi sebelum track dihapus
    };

    struct TrackedFace
    {
        int trackId = 0;
        Common::BoxXyxy box; // posisi setelah dihaluskan Kalman (piksel frame kamera)
        float score = 0.0F;
        int detectionIndex = -1; // index di vector deteksi frame ini (untuk ambil landmark)
    };

    struct TrackerResult
    {
        std::vector<TrackedFace> active;
        std::vector<int> removedIds; // trackId yang dihapus karena terlalu lama tanpa deteksi
    };

    class ByteTracker
    {
    public:
        explicit ByteTracker(ByteTrackerConfig config = {});
        ~ByteTracker();
        ByteTracker(ByteTracker const &) = delete;
        ByteTracker &operator=(ByteTracker const &) = delete;

        // Panggil sekali per frame untuk tracking.
        [[nodiscard]] TrackerResult update(
            std::vector<Common::FaceDetection> const &detections);
        void reset(); // hapus semua track, mulai dari awal

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
}
