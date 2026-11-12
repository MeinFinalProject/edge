#include "scrfd.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace FaceRuntime::Scrfd
{
    Letterbox ComputeLetterbox(Preprocess::FrameSize source)
    {
        if (source.width == 0 || source.height == 0)
        {
            throw std::invalid_argument("Letterbox: ukuran frame sumber 0.");
        }

        Letterbox letterbox;
        const float imageRatio = static_cast<float>(source.height) /
                                 static_cast<float>(source.width);
        const float modelRatio = static_cast<float>(kSpec.height) /
                                 static_cast<float>(kSpec.width);
        if (imageRatio > modelRatio)
        {
            letterbox.resizedHeight = kSpec.height;
            letterbox.resizedWidth = static_cast<std::uint32_t>(static_cast<float>(letterbox.resizedHeight) /
                                                                imageRatio);
        }
        else
        {
            letterbox.resizedWidth = kSpec.width;
            letterbox.resizedHeight = static_cast<std::uint32_t>(static_cast<float>(letterbox.resizedWidth) *
                                                                 imageRatio);
        }
        letterbox.scaleX = static_cast<float>(letterbox.resizedWidth) /
                           static_cast<float>(source.width);
        letterbox.scaleY = static_cast<float>(letterbox.resizedHeight) /
                           static_cast<float>(source.height);
        letterbox.dstToSrc = {
            1.0F / letterbox.scaleX,
            0.0F,
            0.0F,
            0.0F,
            1.0F / letterbox.scaleY,
            0.0F};
        return letterbox;
    };

    namespace
    {
        constexpr std::array<int, 3> kStrides{
            8,
            16,
            32};
        constexpr int kAnchorsPerCell = 2; // SCRFD InsightFace: 2 anchor per sel

        // Tiga kelompok output untuk satu stride (8, 16, atau 32).
        // semua gambar kaya dikotakin -> stride
        // dari stride, ntar di anchorin kira kira kotak ini ada gambar ga?
        // kalau ada gambarnya:
        // confidencenya berapa(1)?
        // prediksi kotaknya gimana (makanya geser panjang lebarnya berapa dari pusat, terus lebar tingginya(jadi deh bounding box) (4)?
        // terus 5 landmark koordinatnya gimana? (10)
        // 1 Baris (Row): Mewakili 1 titik Paku (Anchor).
        // Kolom (Column): Mewakili isi informasi/tebakan dari paku tersebut.
        struct Level
        {
            int stride = 0;
            int anchorCount = 0;           // jumlah baris = (W/stride)*(H/stride)*2
            const float *scores = nullptr; // [anchorCount * 1]
            const float *bbox = nullptr;   // [anchorCount * 4]  jarak l,t,r,b (satuan stride)
            const float *kps = nullptr;    // [anchorCount * 10] offset 5 titik (satuan stride)
        };

        // Menghitung Intersection over Union (IoU) antara dua kotak.
        // IoU = area irisan / area gabungan
        // Kotak yang tidak tumpang tindih menghasilkan IoU=0.0.
        float Iou(Common::BoxXyxy const &a, Common::BoxXyxy const &b)
        {
            const float ix1 = std::max(a.topLeft.x, b.topLeft.x);
            const float iy1 = std::max(a.topLeft.y, b.topLeft.y);
            const float ix2 = std::min(a.bottomRight.x, b.bottomRight.x);
            const float iy2 = std::min(a.bottomRight.y, b.bottomRight.y);

            const float iw = std::max(0.0F, ix2 - ix1); // area irisan
            const float ih = std::max(0.0F, iy2 - iy1);
            const float inter = iw * ih;
            const float uni = a.area() + b.area() - inter; // area gabungan
            return uni > 0.0F ? inter / uni : 0.0F;
        }

        // Mengelompokkan output model (urutan bebas) menjadi 3 level berdasarkan bentuknya:
        // kolom terakhir 1/4/10 menentukan jenis, jumlah baris menentukan stride.
        // Return false dan isi `error` kalau model tidak cocok (mis. tanpa landmark).
        bool RouteOutputs(
            std::vector<Inference::OutputTensor> const &outputs,
            std::array<Level, 3> &levels,
            std::string &error)
        {
            // Inisialisasi level kosong
            for (std::size_t i = 0; i < 3; ++i)
            {
                levels[i] = Level{};
                levels[i].stride = kStrides[i];
                levels[i].anchorCount = static_cast<int>(
                    (kSpec.width / kStrides[i]) *
                    (kSpec.height / kStrides[i]) *
                    kAnchorsPerCell);
            }

            for (auto const &out : outputs)
            {
                if (out.shape.empty() || out.data.empty())
                {
                    continue; // abaikan output kosong
                }

                // Dimensi terakhir menentukan jenis output:
                // 1=score, 4=bbox, 10=kps / landmark
                const std::int64_t lastDim = static_cast<std::int64_t>(out.shape.back());
                if (lastDim != 1 && lastDim != 4 && lastDim != 10)
                {
                    continue; // bukan output yang kita kenal
                }

                // Menghitung jumlah 'baris' (jumlah Anchor) pada tensor ini.
                //
                // Karena data tensor dari ONNX Runtime ditarik menjadi satu array 1D panjang (flattened),
                // kita harus mencari tahu berapa banyak "titik paku (Anchor)" yang ada di dalamnya.
                // Caranya: membagi total elemen array dengan isi data per pakunya (1, 4, atau 10).
                //
                // Contoh: Jika array landmark punya 128.000 elemen, dan 1 paku butuh 10 angka (lastDim),
                // maka tensor ini berisi data untuk 12.800 paku (berarti ini Stride 8).
                const std::int64_t rows = static_cast<std::int64_t>(out.data.size()) / lastDim;
                for (auto &level : levels)
                {
                    // Pastikan jumlah baris sesuai dengan jumlah Anchor(paku) yang diharapkan untuk level ini.
                    if (rows != level.anchorCount)
                    {
                        continue;
                    }
                    if (lastDim == 1)
                        level.scores = out.data.data();
                    else if (lastDim == 4)
                        level.bbox = out.data.data();
                    else
                        level.kps = out.data.data();
                }
            }

            for (auto const &level : levels)
            {
                if (!level.scores || !level.bbox)
                {
                    error = "Output SCRFD tidak lengkap untuk stride " + std::to_string(level.stride) +
                            ": skor/bbox tidak ditemukan. Cek ukuran input model (harus " +
                            std::to_string(kSpec.width) + "x" + std::to_string(kSpec.height) + ").";
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

        // Menghapus kotak yang tumpang tindih (Non-Maximum Suppression, NMS).
        // Kotak dengan skor lebih rendah yang tumpang tindih dengan kotak lain di atas ambang batas IoU akan dihapus.
        std::vector<Common::FaceDetection> NonMaxSuppression(std::vector<Common::FaceDetection> detections, float iouThreshold)
        {
            // Skor terbesar ke terkecil: kotak dengan skor tertinggi otomatis di detections[0].
            std::sort(
                detections.begin(),
                detections.end(),
                [](auto const &a, auto const &b)
                { return a.score > b.score; });

            std::vector<Common::FaceDetection> kept;
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

        // Decode anchor -> kotak + landmark, balik ke koordinat frame kamera, lalu NMS.
        // Padding letterbox ada di KANAN dan BAWAH, jadi cukup dibagi skala (tidak ada offset).
        std::vector<Common::FaceDetection> Decode(
            std::array<Level, 3> const &levels,
            Letterbox const &letterbox,
            Options const &options)
        {
            std::vector<Common::FaceDetection> result;

            // Mengubah prediksi model (anchor) menjadi kotak wajah dan landmark, lalu menyesuaikan ke koordinat frame kamera.
            for (auto const &level : levels)
            {
                const int stride = level.stride;
                const int featureWidth = static_cast<int>(kSpec.width) / stride;
                const float s = static_cast<float>(stride);

                // Urutan anchor: sel demi sel (baris demi baris), tiap sel 2 anchor berurutan.
                for (int i = 0; i < level.anchorCount; ++i)
                {
                    const float score = level.scores[i];
                    if (score < options.scoreThreshold)
                    {
                        continue;
                    }

                    // Urutan anchor: sel demi sel (baris demi baris), tiap sel 2 anchor berurutan.
                    const int cell = i / kAnchorsPerCell;
                    const float cx = static_cast<float>(cell % featureWidth) * s;
                    const float cy = static_cast<float>(cell / featureWidth) * s;

                    const float *d = level.bbox + static_cast<std::size_t>(i) * 4;
                    Common::FaceDetection det;
                    det.score = score;
                    det.box.topLeft.x = (cx - d[0] * s) / letterbox.scaleX;
                    det.box.topLeft.y = (cy - d[1] * s) / letterbox.scaleY;
                    det.box.bottomRight.x = (cx + d[2] * s) / letterbox.scaleX;
                    det.box.bottomRight.y = (cy + d[3] * s) / letterbox.scaleY;

                    // Landmark: 5 titik (x,y) = 10 angka
                    const float *k = level.kps + static_cast<std::size_t>(i) * 10;
                    for (std::size_t j = 0; j < 5; ++j)
                    {
                        det.landmarks[j].x = (cx + k[2 * j] * s) / letterbox.scaleX;
                        det.landmarks[j].y = (cy + k[2 * j + 1] * s) / letterbox.scaleY;
                    }
                    result.push_back(det);
                }
            }
            return NonMaxSuppression(std::move(result), options.nmsThreshold);
        }
    }

    Detector::Detector(Gpu::GpuContext &gpu,
                       Preprocess::WarpContext &warp,
                       std::filesystem::path const &modelPath,
                       Preprocess::FrameSize source,
                       Options options)
        : options(options),
          letterbox(ComputeLetterbox(source)),
          preprocessor(gpu,
                       warp,
                       source,
                       kSpec),
          model(modelPath,
                gpu.learningDevice())
    {
    }

    std::vector<Common::FaceDetection> Detector::detect(
        Gpu::Nv12Frame const &frame)
    {
        auto outputs = model.run(
            preprocessor.run(frame,
                             letterbox.dstToSrc));

        std::array<Level, 3> levels;
        std::string error;
        if (!RouteOutputs(outputs,
                          levels,
                          error))
        {
            throw std::runtime_error("SCRFD: " + error);
        }
        return Decode(levels,
                      letterbox,
                      options);
    }
}