#include "pipeline.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "../camera/camera.hpp"
#include "../common/debug_dump.hpp"

namespace FaceRuntime::Pipeline
{
    namespace
    {
        // Lokasi file model relatif terhadap modelDir (folder `models` di samping .exe).
        constexpr char const *kScrfdModel = "insightface/buffalo_l/det_10g.onnx";
        constexpr char const *kArcFaceModel = "insightface/buffalo_l/w600k_r50.onnx";
        constexpr char const *kLivenessModel = "insightface/addons/liveness.onnx";
        constexpr char const *kLlrFile = "llr.txt";

        Liveness::LlrTable LoadLlrTable(std::filesystem::path const &path, bool &fromFile)
        {
            std::ifstream in(path);
            if (in)
            {
                fromFile = true;
                return Liveness::LlrTable::Load(in);
            }
            fromFile = false;
            std::cerr << "[peringatan] " << path.string()
                      << " tidak ada: memakai tabel LLR pengganti logit(p), BELUM dikalibrasi.\n";
            return Liveness::LlrTable::FromLogit();
        }

        // EstimateSimilarityTransform melempar runtime_error untuk landmark degenerate. Hanya itu
        // yang ditangkap di sini; kesalahan model (ONNX salah bentuk) tetap naik ke pemanggil.
        bool TryEstimate(std::array<Common::Point2f, 5> const &landmarks,
                         std::array<Common::Point2f, 5> const &target,
                         Preprocess::SimilarityTransform2D &out)
        {
            try
            {
                out = Preprocess::EstimateSimilarityTransform(landmarks, target);
                return true;
            }
            catch (std::runtime_error const &)
            {
                return false;
            }
        }

        std::string Fixed(double value, int digits = 2)
        {
            std::ostringstream text;
            text << std::fixed << std::setprecision(digits) << value;
            return text.str();
        }

        struct Stats
        {
            double minimum = 1e30;
            double maximum = -1e30;
            double stddev = 0.0;
        };

        // Ringkasan statistik satu rentang nilai tensor.
        Stats Summarize(std::vector<float> const &values)
        {
            Stats s;
            if (values.empty())
            {
                return s;
            }
            double sum = 0.0;
            double sumSquares = 0.0;
            for (float v : values)
            {
                s.minimum = std::min(s.minimum, static_cast<double>(v));
                s.maximum = std::max(s.maximum, static_cast<double>(v));
                sum += v;
                sumSquares += static_cast<double>(v) * v;
            }
            const double mean = sum / static_cast<double>(values.size());
            s.stddev = std::sqrt(std::max(0.0, sumSquares / static_cast<double>(values.size()) - mean * mean));
            return s;
        }
    }

    FacePipeline::FacePipeline(Gpu::GpuContext &gpu,
                               Preprocess::FrameSize source,
                               std::filesystem::path const &modelDir,
                               FacePipelineConfig config)
        : gpu_(gpu),
          source_(source),
          config_(config),
          bridge_(gpu.deviceRef()),
          warp_(gpu),
          detector_(gpu, warp_, modelDir / kScrfdModel, source),
          liveness_(gpu, warp_, modelDir / kLivenessModel, source),
          arcFace_(gpu, warp_, modelDir / kArcFaceModel, source),
          llr_(LoadLlrTable(modelDir / kLlrFile, llrFromFile_))
    {
    }

    void FacePipeline::printModelSignatures() const
    {
        detector_.printModelSignature();
        liveness_.printModelSignature();
        arcFace_.printModelSignature();
    }

    FacePipeline::TrackSlot &FacePipeline::slotFor(int trackId)
    {
        return slots_.try_emplace(trackId, config_.decision, llr_).first->second;
    }

    float FacePipeline::outOfBoundsRatio(Preprocess::SimilarityTransform2D const &transform,
                                         Preprocess::TensorSpec const &spec) const
    {
        return Preprocess::ComputeOutOfBoundsRatio(transform, spec.width, spec.height, source_.width, source_.height);
    }

    // ======================================================================
    // process(): SATU FRAME DARI AWAL SAMPAI AKHIR
    // ======================================================================
    std::vector<FaceResult> FacePipeline::process(MediaFrameReference frame)
    {
        FrameLease lease(gpu_, std::move(frame)); // ~FrameLease: waitIdle() lalu frame.Close()
        ++frameIndex_;
        auto texture = TextureOf(lease.frame());
        Gpu::Nv12Frame const nv12 = bridge_.acquire(texture);

        auto const detections = detector_.detect(nv12);
        auto const tracked = tracker_.update(detections);
        for (int id : tracked.removedIds)
        {
            slots_.erase(id);
        }

        // Maksimal SATU observasi per frame: track yang masih butuh bukti dan sudah waktunya,
        // dengan pose paling frontal (RMSE terkecil).
        Candidate best;
        for (auto const &track : tracked.active)
        {
            TrackSlot &slot = slotFor(track.trackId);
            slot.decision.tick(frameIndex_);

            if (track.detectionIndex < 0 || track.detectionIndex >= static_cast<int>(detections.size()))
            {
                continue; // track hanya diprediksi Kalman di frame ini, tidak ada deteksi/landmark
            }
            auto const &face = detections[static_cast<std::size_t>(track.detectionIndex)];

            const bool needsEvidence = recording_.is_open() ||
                                       slot.decision.wantsLiveness() ||
                                       slot.decision.wantsEmbedding();
            if (!needsEvidence || frameIndex_ < slot.nextObservationFrame ||
                face.score < config_.minDetectionScore)
            {
                continue;
            }

            Preprocess::SimilarityTransform2D pose;
            if (!TryEstimate(face.landmarks, ArcFace::kTemplate, pose) || pose.rmse > config_.maxPoseRmse)
            {
                continue;
            }
            if (best.slot == nullptr || pose.rmse < best.rmse)
            {
                best.slot = &slot;
                best.trackId = track.trackId;
                best.face = &face;
                best.rmse = pose.rmse;
            }
        }
        if (best.slot != nullptr)
        {
            observe(nv12, *best.face, best.trackId, *best.slot);
        }

        std::vector<FaceResult> results;
        results.reserve(tracked.active.size());
        for (auto const &track : tracked.active)
        {
            TrackDecision const &decision = slotFor(track.trackId).decision;
            FaceResult result;
            result.trackId = track.trackId;
            result.box = track.box;
            result.status = decision.status();
            result.name = decision.name();
            result.similarity = decision.similarity();
            result.margin = decision.margin();
            result.liveProbability = decision.lastLiveProbability();
            result.sprtSum = decision.sprtSum();
            result.livenessObservations = decision.livenessObservations();
            results.push_back(std::move(result));
        }
        return results;
    }

    // Satu observasi untuk satu track: liveness (kalau masih Pending) atau ArcFace (kalau sudah Live).
    void FacePipeline::observe(Gpu::Nv12Frame const &nv12,
                               Common::FaceDetection const &face,
                               int trackId,
                               TrackSlot &slot)
    {
        slot.nextObservationFrame = frameIndex_ + config_.observeEveryNFrames;

        std::ostringstream line;
        line << std::fixed << std::setprecision(3) << "f=" << frameIndex_ << " t=" << trackId << ' ';

        if (recording_.is_open() || slot.decision.wantsLiveness())
        {
            Preprocess::SimilarityTransform2D transform;
            if (!TryEstimate(face.landmarks, Liveness::kTemplate, transform))
            {
                return;
            }
            const float oob = outOfBoundsRatio(transform, Liveness::kSpec);
            if (oob > config_.maxOobRatio)
            {
                if (config_.trace != nullptr)
                {
                    *config_.trace << line.str() << "liveness DILEWATI: crop keluar frame " << Fixed(oob * 100, 0) << "%\n";
                }
                return; // observasi tidak valid: tidak dihitung sebagai bukti
            }

            const float p = liveness_.realProbability(nv12, transform);
            if (recording_.is_open())
            {
                recording_ << "r_" << recordingStem_ << "_t" << trackId << ',' << recordingLabel_ << ','
                           << frameIndex_ << ',' << p << '\n';
                if (config_.trace != nullptr)
                {
                    *config_.trace << line.str() << "REKAM p=" << p << '\n';
                }
                return;
            }

            slot.decision.addLiveness(p, frameIndex_);
            if (config_.trace != nullptr)
            {
                *config_.trace << line.str() << "liveness p=" << p << " S=" << slot.decision.sprtSum()
                               << " n=" << slot.decision.livenessObservations() << " -> "
                               << ToString(slot.decision.status()) << '\n';
            }
            return;
        }

        Preprocess::SimilarityTransform2D transform;
        if (!TryEstimate(face.landmarks, ArcFace::kTemplate, transform))
        {
            return;
        }
        const float oob = outOfBoundsRatio(transform, ArcFace::kSpec);
        if (oob > config_.maxOobRatio)
        {
            if (config_.trace != nullptr)
            {
                *config_.trace << line.str() << "arcface DILEWATI: crop keluar frame " << Fixed(oob * 100, 0) << "%\n";
            }
            return;
        }

        auto const embedding = arcFace_.embed(nv12, transform);
        slot.decision.addEmbedding(embedding, gallery_, frameIndex_);
        if (config_.trace != nullptr)
        {
            *config_.trace << line.str() << "arcface sim=" << slot.decision.similarity()
                           << " margin=" << slot.decision.margin() << " -> "
                           << ToString(slot.decision.status());
            if (slot.decision.status() == TrackStatus::Identified)
            {
                *config_.trace << ' ' << slot.decision.name();
            }
            *config_.trace << '\n';
        }
    }

    // ======================================================================
    // enroll(): daftarkan wajah terbesar di frame ini
    // ======================================================================
    EnrollResult FacePipeline::enroll(MediaFrameReference frame, std::string const &name)
    {
        FrameLease lease(gpu_, std::move(frame));
        auto texture = TextureOf(lease.frame());
        Gpu::Nv12Frame const nv12 = bridge_.acquire(texture);
        auto const detections = detector_.detect(nv12);

        Common::FaceDetection const *largest = nullptr;
        for (auto const &d : detections)
        {
            if (largest == nullptr || d.box.area() > largest->box.area())
            {
                largest = &d;
            }
        }
        if (largest == nullptr)
        {
            return {false, "tidak ada wajah di frame"};
        }
        if (largest->score < config_.minDetectionScore)
        {
            return {false, "skor deteksi terlalu rendah (" + Fixed(largest->score) + ")"};
        }

        Preprocess::SimilarityTransform2D transform;
        if (!TryEstimate(largest->landmarks, ArcFace::kTemplate, transform))
        {
            return {false, "landmark tidak valid"};
        }
        if (transform.rmse > config_.maxPoseRmse)
        {
            return {false, "wajah kurang frontal (rmse=" + Fixed(transform.rmse) + ", maks " + Fixed(config_.maxPoseRmse) + ")"};
        }
        const float oob = outOfBoundsRatio(transform, ArcFace::kSpec);
        if (oob > config_.maxOobRatio)
        {
            return {false, "wajah terpotong di tepi frame (" + Fixed(oob * 100, 0) + "% crop di luar)"};
        }

        gallery_.add(name, arcFace_.embed(nv12, transform));
        return {true, "ok"};
    }

    // ======================================================================
    // Rekam skor liveness untuk kalibrasi
    // ======================================================================
    void FacePipeline::startLivenessRecording(std::filesystem::path const &csvPath, int label)
    {
        if (label != 0 && label != 1)
        {
            throw std::invalid_argument("label rekam harus 0 (spoof) atau 1 (asli).");
        }
        recording_.close();
        recording_.clear();
        recording_.open(csvPath, std::ios::trunc);
        if (!recording_)
        {
            throw std::runtime_error("tidak bisa menulis " + csvPath.string());
        }
        recording_ << "seq,label,frame,p\n";
        recordingLabel_ = label;
        recordingStem_ = csvPath.stem().string();
    }

    void FacePipeline::stopLivenessRecording()
    {
        recording_.close();
    }

    // ======================================================================
    // selfTest(): UJI AKHIR sekali jalan, satu frame, seluruh rantai
    // ======================================================================
    SelfTestReport FacePipeline::selfTest(MediaFrameReference frame, std::filesystem::path const &dumpDirectory)
    {
        SelfTestReport report;
        auto add = [&report](std::string name, bool ok, std::string detail)
        {
            report.items.push_back({std::move(name), ok, std::move(detail)});
        };

        FrameLease lease(gpu_, std::move(frame));
        auto texture = TextureOf(lease.frame());
        Gpu::Nv12Frame const nv12 = bridge_.acquire(texture);
        auto const detections = detector_.detect(nv12);
        const bool dump = !dumpDirectory.empty();
        if (dump)
        {
            std::filesystem::create_directories(dumpDirectory);
        }

        // 1) Tensor SCRFD: gambar menempel di kiri-atas, SELURUH sisanya hitam (nilai normBias).
        {
            auto tensor = detector_.debugReadback();
            if (dump)
            {
                Common::WriteFloatsToFile(dumpDirectory / "scrfd.bin", tensor);
            }
            const auto content = detector_.contentSize();
            const std::uint32_t w = Scrfd::kSpec.width;
            const std::uint32_t h = Scrfd::kSpec.height;
            const std::size_t plane = static_cast<std::size_t>(w) * h;
            std::size_t padCount = 0;
            std::size_t padWrong = 0;
            std::vector<float> filled;
            if (tensor.size() == 3 * plane)
            {
                for (std::size_t c = 0; c < 3; ++c)
                {
                    for (std::uint32_t y = 0; y < h; ++y)
                    {
                        for (std::uint32_t x = 0; x < w; ++x)
                        {
                            const float v = tensor[c * plane + static_cast<std::size_t>(y) * w + x];
                            if (x >= content.width || y >= content.height)
                            {
                                ++padCount;
                                padWrong += std::fabs(v - Scrfd::kSpec.normBias) > 1e-4F ? 1 : 0;
                            }
                            else
                            {
                                filled.push_back(v);
                            }
                        }
                    }
                }
            }
            const Stats stats = Summarize(filled);
            const double percent = 100.0 * content.width * content.height / (static_cast<double>(w) * h);
            add("SCRFD: tensor 640x640, gambar kiri-atas, sisa hitam",
                tensor.size() == 3 * plane && padWrong == 0 && stats.stddev > 0.05,
                "isi gambar " + std::to_string(content.width) + "x" + std::to_string(content.height) + " = " +
                    Fixed(percent, 1) + "% dari tensor; piksel hitam yang salah " + std::to_string(padWrong) +
                    " dari " + std::to_string(padCount) + "; variasi isi gambar " + Fixed(stats.stddev, 3));
        }

        // 2) Deteksi wajah
        Common::FaceDetection const *best = nullptr;
        for (auto const &d : detections)
        {
            if (d.score >= config_.minDetectionScore && (best == nullptr || d.score > best->score))
            {
                best = &d;
            }
        }
        add("SCRFD: ada wajah dengan skor >= " + Fixed(config_.minDetectionScore),
            best != nullptr,
            best != nullptr ? std::to_string(detections.size()) + " wajah, skor terbaik " + Fixed(best->score, 3)
                            : "tidak ada; hadapkan wajah ke kamera");
        if (best == nullptr)
        {
            return report; // faceFound tetap false: pemanggil mencoba frame berikutnya
        }
        report.faceFound = true;

        const auto &lm = best->landmarks;
        const bool orderOk = lm[0].x < lm[1].x && lm[2].y > lm[0].y && lm[3].y > lm[2].y;
        add("SCRFD: urutan 5 landmark (mata kiri < mata kanan, hidung di bawah mata, mulut di bawah hidung)",
            orderOk, orderOk ? "benar" : "urutan landmark salah: decode output SCRFD bermasalah");

        // 3) Transform dan gerbang
        Preprocess::SimilarityTransform2D t80;
        Preprocess::SimilarityTransform2D t112;
        if (!TryEstimate(lm, Liveness::kTemplate, t80) || !TryEstimate(lm, ArcFace::kTemplate, t112))
        {
            add("Transform landmark -> template", false, "landmark degenerate");
            return report;
        }
        const float oob80 = outOfBoundsRatio(t80, Liveness::kSpec);
        const float oob112 = outOfBoundsRatio(t112, ArcFace::kSpec);
        add("Gerbang observasi (pose dan crop di dalam frame)",
            t112.rmse <= config_.maxPoseRmse && oob80 <= config_.maxOobRatio && oob112 <= config_.maxOobRatio,
            "rmse pose " + Fixed(t112.rmse) + " (maks " + Fixed(config_.maxPoseRmse) + "), crop keluar frame " +
                Fixed(oob80 * 100, 0) + "% / " + Fixed(oob112 * 100, 0) + "% (maks " + Fixed(config_.maxOobRatio * 100, 0) + "%)");

        // 4) Liveness
        const float p = liveness_.realProbability(nv12, t80);
        {
            auto tensor = liveness_.debugReadback();
            if (dump)
            {
                Common::WriteFloatsToFile(dumpDirectory / "pad.bin", tensor);
            }
            const Stats stats = Summarize(tensor);
            add("Liveness: tensor 80x80 bernilai 0..1 dan probabilitas terbaca",
                stats.minimum >= -1e-3 && stats.maximum <= 1.001 && stats.stddev > 0.02 && p >= 0.0F && p <= 1.0F,
                "nilai tensor " + Fixed(stats.minimum, 3) + ".." + Fixed(stats.maximum, 3) + ", variasi " +
                    Fixed(stats.stddev, 3) + "; probabilitas asli p=" + Fixed(p, 3));
        }

        // 5) ArcFace
        const auto e1 = arcFace_.embed(nv12, t112);
        {
            auto tensor = arcFace_.debugReadback();
            if (dump)
            {
                Common::WriteFloatsToFile(dumpDirectory / "arc.bin", tensor);
            }
            const Stats stats = Summarize(tensor);
            add("ArcFace: tensor 112x112 bernilai -1..1",
                stats.minimum >= -1.001 && stats.maximum <= 1.001 && stats.stddev > 0.05,
                "nilai tensor " + Fixed(stats.minimum, 3) + ".." + Fixed(stats.maximum, 3) + ", variasi " + Fixed(stats.stddev, 3));

            const auto e2 = arcFace_.embed(nv12, t112); // frame yang sama dua kali harus menghasilkan vektor yang sama
            const float norm = ArcFace::Dot(e1, e1);
            const float again = ArcFace::Dot(e1, e2);
            add("ArcFace: embedding 512 angka, panjang 1, deterministik",
                e1.size() == ArcFace::kEmbeddingSize && std::fabs(norm - 1.0F) < 1e-3F && again > 0.999F,
                std::to_string(e1.size()) + " angka, dot(e,e)=" + Fixed(norm, 4) + ", dua kali jalan: cosine " + Fixed(again, 4));
        }

        // 6) Galeri (informasi saja)
        if (gallery_.size() == 0)
        {
            add("Galeri", true, "kosong (belum ada yang didaftarkan)");
        }
        else
        {
            const auto m = gallery_.match(e1);
            add("Galeri", true,
                std::to_string(gallery_.size()) + " entri; terdekat '" + m.name + "' similarity " + Fixed(m.similarity, 3) +
                    ", margin " + Fixed(m.margin(), 3));
        }
        return report;
    }
}
