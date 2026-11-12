#pragma once
// SEMUA alur satu frame ada di file ini dan pipeline.cpp:
//   1. FrameLease      tiket frame kamera, ditutup otomatis SETELAH kerja GPU selesai
//   2. TrackDecision   keputusan per track (liveness SPRT -> identitas); murni angka, tanpa GPU
//   3. FacePipeline    frame -> deteksi -> tracking -> liveness -> ArcFace -> hasil, dalam satu fungsi process()
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <winrt/Windows.Media.Capture.Frames.h>
#include "../common/types.hpp"
#include "../gpu/gpu_context.hpp"
#include "../gpu/nv12_bridge.hpp"
#include "../models/arcface.hpp"
#include "../models/liveness.hpp"
#include "../models/scrfd.hpp"
#include "../preprocess/similarity_transform.hpp"
#include "../preprocess/warp.hpp"
#include "../tracking/byte_tracker.hpp"

namespace FaceRuntime::Pipeline
{
    using MediaFrameReference = winrt::Windows::Media::Capture::Frames::MediaFrameReference;

    // ======================================================================
    // 1. FrameLease
    // ======================================================================
    // Tiket frame kamera. Tekstur kamera baru boleh dikembalikan ke kamera (Close) SETELAH semua kerja
    // GPU yang membaca teksturnya selesai. RAII membuat urutan itu otomatis benar di semua jalur,
    // termasuk saat terjadi exception: destruktor menunggu GPU idle, lalu menutup tiket.
    class FrameLease
    {
    public:
        FrameLease(Gpu::GpuContext &gpu,
                   MediaFrameReference frame)
            : gpu(gpu),
          frame_(std::move(frame))
        {
        }

        ~FrameLease()
        {
            try
            {
                gpu.waitIdle();
            }
            catch (...)
            {
            }
            try
            {
                if (frame_)
                {
                    frame_.Close();
                }
            }
            catch (...)
            {
            }
        }

        FrameLease(FrameLease const &) = delete;
        FrameLease &operator=(FrameLease const &) = delete;

        [[nodiscard]] MediaFrameReference const &frame() const
        {
            return frame_;
        }

    private:
        Gpu::GpuContext &gpu;
        MediaFrameReference frame_;
    };

    // ======================================================================
    // 2. TrackDecision
    // ======================================================================
    enum class TrackStatus
    {
        Pending,    // liveness belum diputuskan
        Live,       // lolos liveness, identitas belum diputuskan
        Identified, // identitas terkunci
        Unknown,    // asli, tapi tidak ada yang cocok (coba lagi setelah cooldown)
        Spoof,      // PERMANEN untuk track ini
        Undecided   // bukti liveness tak cukup sampai batas (coba lagi setelah cooldown)
    };

    inline char const *ToString(TrackStatus status)
    {
        switch (status)
        {
        case TrackStatus::Pending:
            return "Pending";
        case TrackStatus::Live:
            return "Live";
        case TrackStatus::Identified:
            return "Identified";
        case TrackStatus::Unknown:
            return "Unknown";
        case TrackStatus::Spoof:
            return "Spoof";
        case TrackStatus::Undecided:
            return "Undecided";
        }
        return "?";
    }

    struct DecisionConfig
    {
        Liveness::SprtConfig sprt;
        float acceptSimilarity = 0.40F; // cosine minimal -> TUNING dari data
        float minMargin = 0.05F;        // selisih ke orang lain terdekat -> TUNING
        int maxEmbeddings = 3;          // embedding yang dirata-rata sebelum menyerah (Unknown)
        int retryCooldownFrames = 45;   // jeda sebelum Undecided/Unknown dicoba lagi
    };

    // keputusan per track (liveness SPRT -> identitas); murni angka, tanpa GPU
    class TrackDecision
    {
    public:
        TrackDecision(DecisionConfig const &config,
                      Liveness::LlrTable const &table)
            : config_(config),
              sprt_(config.sprt,
                    table)
        {
        }

        [[nodiscard]] TrackStatus status() const
        {
            return status_;
        }
        [[nodiscard]] bool wantsLiveness() const
        {
            return status_ == TrackStatus::Pending;
        }
        [[nodiscard]] bool wantsEmbedding() const
        {
            return status_ == TrackStatus::Live;
        }

        void addLiveness(float liveProbability, std::int64_t frame)
        {
            if (!wantsLiveness())
                return;
            lastLiveProbability_ = liveProbability;
            switch (sprt_.observe(liveProbability))
            {
            case Liveness::Verdict::Live:
                status_ = TrackStatus::Live;
                break;
            case Liveness::Verdict::Spoof:
                status_ = TrackStatus::Spoof;
                settledAt_ = frame;
                break;
            case Liveness::Verdict::Undecided:
                status_ = TrackStatus::Undecided;
                settledAt_ = frame;
                break;
            case Liveness::Verdict::Pending:
                break;
            }
        }

        // `embedding` harus SUDAH L2-normalized.
        void addEmbedding(std::vector<float> const &embedding,
                          ArcFace::Gallery const &gallery,
                          std::int64_t frame)
        {
            if (!wantsEmbedding())
                return;
            if (sum_.size() != embedding.size())
                sum_.assign(embedding.size(), 0.0F);
            for (std::size_t i = 0; i < embedding.size(); ++i)
                sum_[i] += embedding[i];
            ++embeddings_;

            std::vector<float> mean = sum_; // rata-rata embedding, dinormalisasi ulang
            double norm = 0.0;
            for (float v : mean)
                norm += static_cast<double>(v) * v;
            norm = std::sqrt(norm);
            if (norm > 1e-9)
                for (float &v : mean)
                    v = static_cast<float>(v / norm);

            auto const match = gallery.match(mean);
            similarity_ = match.similarity;
            margin_ = match.margin();
            if (match.similarity >= config_.acceptSimilarity && match.margin() >= config_.minMargin)
            {
                status_ = TrackStatus::Identified;
                name_ = match.name;
            }
            else if (embeddings_ >= config_.maxEmbeddings)
            {
                status_ = TrackStatus::Unknown;
                settledAt_ = frame;
            }
        }

        // Panggil sekali per frame untuk tiap track aktif.
        // Jika Undecided/Unknown sudah lewat cooldown, reset SPRT atau rata-rata embedding.
        void tick(std::int64_t frame)
        {
            // Reset Undecided/Unknown setelah cooldown, supaya bisa dicoba lagi.
            if ((status_ == TrackStatus::Undecided ||
                 status_ == TrackStatus::Unknown) &&
                frame - settledAt_ >= config_.retryCooldownFrames)
            {
                if (status_ == TrackStatus::Undecided)
                {
                    sprt_.reset();
                    status_ = TrackStatus::Pending;
                }
                else
                {
                    sum_.clear();
                    embeddings_ = 0;
                    status_ = TrackStatus::Live;
                }
            }
        }

        [[nodiscard]] std::string const &name() const
        {
            return name_;
        }
        [[nodiscard]] float similarity() const
        {
            return similarity_;
        }
        [[nodiscard]] float margin() const
        {
            return margin_;
        }
        [[nodiscard]] float lastLiveProbability() const
        {
            return lastLiveProbability_;
        }
        [[nodiscard]] float sprtSum() const
        {
            return sprt_.sum();
        }
        [[nodiscard]] int livenessObservations() const
        {
            return sprt_.count();
        }

    private:
        DecisionConfig config_;
        Liveness::SprtAccumulator sprt_;
        TrackStatus status_ = TrackStatus::Pending;
        std::string name_ = "Unknown";
        std::vector<float> sum_;
        int embeddings_ = 0;
        float similarity_ = -1.0F;
        float margin_ = -1.0F;
        float lastLiveProbability_ = -1.0F;
        std::int64_t settledAt_ = 0;
    };

    // 3. FacePipeline
    // ======================================================================
    struct FacePipelineConfig
    {
        float minDetectionScore = 0.60F; // deteksi di bawah ini tidak dipakai sebagai observasi
        float maxPoseRmse = 8.0F;        // "cukup frontal": RMSE landmark terhadap template ArcFace
        float maxOobRatio = 0.30F;       // maks. bagian crop yang keluar frame (observasi valid)
        int observeEveryNFrames = 3;     // jarak antar-observasi per track (menurunkan korelasi)
        DecisionConfig decision;         // SPRT + ambang identitas + cooldown
        std::ostream *trace = nullptr;   // bila diisi, satu baris jejak per observasi
    };

    struct FaceResult
    {
        int trackId = 0;
        Common::BoxXyxy box;
        TrackStatus status = TrackStatus::Pending;
        std::string name = "Unknown";
        float similarity = -1.0F;      // cosine terbaik ke galeri (-1 bila belum ada)
        float margin = -1.0F;          // selisih ke orang lain terdekat
        float liveProbability = -1.0F; // observasi liveness terakhir (-1 bila belum ada)
        float sprtSum = 0.0F;          // akumulasi LLR saat ini
        int livenessObservations = 0;
    };

    struct EnrollResult
    {
        bool added = false;
        std::string message;
    };

    struct CheckItem
    {
        std::string name;
        bool ok = false;
        std::string detail;
    };

    struct SelfTestReport
    {
        bool faceFound = false; // false: belum ada wajah yang layak, coba frame berikutnya
        std::vector<CheckItem> items;
    };

    class FacePipeline
    {
    public:
        FacePipeline(Gpu::GpuContext &gpu,
                     Preprocess::FrameSize source,
                     std::filesystem::path const &modelDir, // berisi insightface/...
                     FacePipelineConfig config = {});
        FacePipeline(FacePipeline const &) = delete;
        FacePipeline &operator=(FacePipeline const &) = delete;

        // SATU fungsi yang merangkum semuanya. Tiket frame ditutup di dalamnya, setelah seluruh kerja
        // GPU selesai (juga saat exception), jadi pemanggil tidak perlu memikirkan urutan Close().
        [[nodiscard]] std::vector<FaceResult> process(
            MediaFrameReference frame);

        // Daftarkan wajah terbesar di frame ini ke galeri (panggil beberapa kali, beda pose).
        EnrollResult enroll(
            MediaFrameReference frame,
            std::string const &name);

        // Mode rekam untuk kalibrasi liveness: tiap observasi valid ditulis "seq,label,frame,p";
        // keputusan SPRT dimatikan selama merekam. label: 1 = wajah asli, 0 = spoof.
        void startLivenessRecording(
            std::filesystem::path const &csvPath,
            int label);

        void stopLivenessRecording();

        // UJI AKHIR sekali jalan: memeriksa seluruh rantai pada SATU frame (tensor tiap model,
        // landmark, gerbang, liveness, ArcFace, galeri). Bila `dumpDirectory` diisi, tensor ditulis
        // sebagai scrfd.bin, pad.bin, arc.bin ke folder itu (untuk tools/tensor_to_png.py).
        [[nodiscard]] SelfTestReport selfTest(
            MediaFrameReference frame,
            std::filesystem::path const &dumpDirectory = {});

        void printModelSignatures() const;

        [[nodiscard]] ArcFace::Gallery &gallery()
        {
            return gallery_;
        }
        [[nodiscard]] bool llrTableFromFile() const
        {
            return llrFromFile_;
        }

    private:
        struct TrackSlot
        {
            TrackSlot(DecisionConfig const &config,
                      Liveness::LlrTable const &table)
                : decision(config, table) {}
            TrackDecision decision;
            std::int64_t nextObservationFrame = 0;
        };

        struct Candidate
        {
            TrackSlot *slot = nullptr;
            int trackId = 0;
            Common::FaceDetection const *face = nullptr;
            float rmse = 0.0F;
        };

        TrackSlot &slotFor(int trackId);
        void observe(Gpu::Nv12Frame const &nv12, Common::FaceDetection const &face, int trackId, TrackSlot &slot);
        [[nodiscard]] float outOfBoundsRatio(Preprocess::SimilarityTransform2D const &transform,
                                             Preprocess::TensorSpec const &spec) const;

        Gpu::GpuContext &gpu_;
        Preprocess::FrameSize source_;
        FacePipelineConfig config_;

        Gpu::Nv12Bridge bridge_;
        Preprocess::WarpContext warp_;
        Scrfd::Detector detector_;
        Liveness::Classifier liveness_;
        ArcFace::Embedder arcFace_;
        Tracking::ByteTracker tracker_;

        bool llrFromFile_ = false; // harus dideklarasikan SEBELUM llr_ (dipakai saat llr_ dibuat)
        Liveness::LlrTable llr_;
        ArcFace::Gallery gallery_;

        std::unordered_map<int, TrackSlot> slots_;
        std::int64_t frameIndex_ = 0;

        std::ofstream recording_;
        int recordingLabel_ = 0;
        std::string recordingStem_;
    };
}