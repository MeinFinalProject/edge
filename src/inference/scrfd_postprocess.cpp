#include "scrfd_postprocess.hpp"
#include <algorithm>
#include <utility>

namespace FaceRuntime::Inference
{
    namespace
    {
        constexpr std::array<int, 3> kStrides{8, 16, 32};
        constexpr int kAnchorsPerCell = 2; // SCRFD InsightFace: 2 anchor per sel

        float Iou(Common::BoxXyxy const &a,
                  Common::BoxXyxy const &b)
        {
            const float ix1 = std::max(a.x1, b.x1);
            const float iy1 = std::max(a.y1, b.y1);
            const float ix2 = std::min(a.x2, b.x2);
            const float iy2 = std::min(a.y2, b.y2);

            // Area irisan
            const float iw = std::max(0.0F, ix2 - ix1);
            const float ih = std::max(0.0F, iy2 - iy1);

            // Area gabungan
            const float inter = iw * ih;
            const float uni = a.area() + b.area() - inter;
            return uni > 0.0F ? inter / uni : 0.0F;
        }
    }

    bool RouteScrfdOutputs(
        std::vector<TensorView> const &outputs,
        std::uint32_t inputWidth,
        std::uint32_t inputHeight,
        std::array<ScrfdLevel, 3> &levels,
        std::string &error)
    {
        // Inisialisasi level dengan anchorCount yang benar
        for (std::size_t i = 0; i < 3; ++i)
        {
            levels[i] = ScrfdLevel{};
            levels[i].stride = kStrides[i];
            levels[i].anchorCount = static_cast<int>(
                (inputWidth / kStrides[i]) *
                (inputHeight / kStrides[i]) *
                kAnchorsPerCell);
        }

        // Iterasi semua output model,
        // cocokkan dengan level berdasarkan jumlah baris dan dimensi terakhir
        for (auto const &out : outputs)
        {
            // Abaikan output kosong
            if (out.shape.empty() || out.data == nullptr)
            {
                continue;
            }

            // Dimensi terakhir menentukan jenis output: 1=score, 4=bbox, 10=kps / landmark
            const std::int64_t lastDim = out.shape.back();
            if (lastDim != 1 && lastDim != 4 && lastDim != 10)
            {
                continue; // bukan output yang kita kenal
            }

            // Jumlah baris = total elemen / dimensi terakhir
            const std::int64_t rows = static_cast<std::int64_t>(out.count) / lastDim;

            // Cocokkan dengan level berdasarkan jumlah baris
            for (auto &level : levels)
            {
                if (rows != level.anchorCount)
                {
                    continue;
                }
                if (lastDim == 1)
                {
                    level.scores = out.data;
                }
                else if (lastDim == 4)
                {
                    level.bbox = out.data;
                }
                else
                {
                    level.kps = out.data;
                }
            }
        }
        for (auto const &level : levels)
        {
            if (!level.scores || !level.bbox)
            {
                error = "Output SCRFD tidak lengkap untuk stride " + std::to_string(level.stride) +
                        ": skor/bbox tidak ditemukan. Cek ukuran input model (harus " +
                        std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + ").";
                return false;
            }
            if (!level.kps)
            {
                error = "Model SCRFD ini tidak punya output landmark (kps) untuk stride " +
                        std::to_string(level.stride) +
                        ". Pakai varian dengan 5 landmark (mis. det_10g.onnx dari buffalo_l).";
                return false;
            }
        }
        return true;
    }

    // Decode anchor -> kotak + landmark, balik ke koordinat frame kamera, lalu NMS.
    std::vector<Common::FaceDetection> DecodeScrfd(
        std::array<ScrfdLevel, 3> const &levels,
        ScrfdDecodeConfig const &config)
    {
        std::vector<Common::FaceDetection> result;
        // Iterasi semua level (stride) dan semua anchor di tiap level
        for (auto const &level : levels)
        {
            const int stride = level.stride;
            const int featureWidth = static_cast<int>(config.inputWidth) / stride;
            const float s = static_cast<float>(stride);

            for (int i = 0; i < level.anchorCount; ++i)
            {
                // Ambil skor anchor, lewati kalau di bawah threshold
                const float score = level.scores[i];
                if (score < config.scoreThreshold)
                {
                    continue;
                }

                // Urutan anchor: sel demi sel (baris demi baris), tiap sel 2 anchor berurutan.
                const int cell = i / kAnchorsPerCell;
                const float cx = static_cast<float>(cell % featureWidth) * s;
                const float cy = static_cast<float>(cell / featureWidth) * s;

                // Ambil jarak left,top,right,bottom dan landmark, balik ke koordinat frame kamera
                const float *d = level.bbox + static_cast<std::size_t>(i) * 4;
                Common::FaceDetection det;
                det.score = score;
                det.box.x1 = (cx - d[0] * s) / config.scaleX;
                det.box.y1 = (cy - d[1] * s) / config.scaleY;
                det.box.x2 = (cx + d[2] * s) / config.scaleX;
                det.box.y2 = (cy + d[3] * s) / config.scaleY;

                const float *k = level.kps + static_cast<std::size_t>(i) * 10;
                for (std::size_t j = 0; j < 5; ++j)
                {
                    det.landmarks[j].x = (cx + k[2 * j] * s) / config.scaleX;
                    det.landmarks[j].y = (cy + k[2 * j + 1] * s) / config.scaleY;
                }
                result.push_back(det);
            }
        }

        return NonMaxSuppression(std::move(result), config.nmsThreshold);
    }

    std::vector<Common::FaceDetection> NonMaxSuppression(
        std::vector<Common::FaceDetection> detections,
        float iouThreshold)
    {
        // Daftar kandidat wajah langsung diurutkan dari skor terbesar ke terkecil.
        // Kotak dengan score tertinggi (misal 99%) otomatis berada di detections[0].
        std::sort(detections.begin(), detections.end(),
                  [](auto const &a, auto const &b)
                  { return a.score > b.score; });

        std::vector<Common::FaceDetection> kept;
        // Iterasi semua kandidat, simpan yang tidak tumpang tindih dengan yang sudah disimpan.
        for (auto const &candidate : detections)
        {
            bool suppressed = false;
            for (auto const &winner : kept)
            {
                if (Iou(candidate.box, winner.box) > iouThreshold)
                {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed)
            {
                kept.push_back(candidate);
            }
        }
        return kept;
    }
}