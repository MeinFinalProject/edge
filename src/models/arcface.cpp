#include "arcface.hpp"
#include <fstream>
#include <sstream>
#include <utility>

namespace FaceRuntime::ArcFace
{
    // ======================================================================
    // 3. Embedder
    // ======================================================================
    Embedder::Embedder(Gpu::GpuContext &gpu,
                       Preprocess::WarpContext &warp,
                       std::filesystem::path const &modelPath,
                       Preprocess::FrameSize source)
        : preprocessor(gpu,
                       warp,
                       source,
                       kSpec),
          model(modelPath,
                gpu.learningDevice())
    {
    }

    std::vector<float> Embedder::embed(Gpu::Nv12Frame const &frame,
                                       Preprocess::SimilarityTransform2D const &transform)
    {
        auto outputs = model.run(preprocessor.run(frame,
                                                  transform.dsttoSrc));
        if (outputs.empty())
        {
            throw std::runtime_error("ArcFace: model tidak mengeluarkan output.");
        }
        std::vector<float> embedding = std::move(outputs[0].data);
        if (embedding.size() != kEmbeddingSize)
        {
            throw std::runtime_error("ArcFace: panjang embedding " + std::to_string(embedding.size()) +
                                     ", diharapkan " + std::to_string(kEmbeddingSize) +
                                     ". Periksa file ONNX di Netron.");
        }
        L2Normalize(embedding); // setelah ini dot(a, b) = cosine similarity
        return embedding;
    }

    // ======================================================================
    // 4. Gallery
    // ======================================================================
    void Gallery::add(std::string name, std::vector<float> embedding)
    {
        if (name.find(' ') != std::string::npos)
        {
            throw std::invalid_argument("Nama tidak boleh mengandung spasi (format file berbasis spasi).");
        }
        entries.push_back({std::move(name),
                           std::move(embedding)});
    }

    Match Gallery::match(std::vector<float> const &embedding) const
    {
        Match result;
        std::vector<float> scores(entries.size(), -1.0F);
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            if (entries[i].embedding.size() != embedding.size())
            {
                continue;
            }
            scores[i] = Dot(entries[i].embedding, embedding); // keduanya satuan -> dot = cosine
            if (scores[i] > result.similarity)
            {
                result.similarity = scores[i];
                result.name = entries[i].name;
            }
        }
        // Runner-up hanya dari nama BERBEDA: dua entri milik orang yang sama bukan pesaing.
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            if (entries[i].name != result.name && scores[i] > result.runnerUpSimilarity)
            {
                result.runnerUpSimilarity = scores[i];
            }
        }
        return result;
    }

    void Gallery::save(std::filesystem::path const &path) const
    {
        std::ofstream out(path);
        if (!out)
        {
            throw std::runtime_error("Tidak bisa menulis galeri: " + path.string());
        }
        for (auto const &entry : entries)
        {
            out << entry.name;
            for (float v : entry.embedding)
            {
                out << ' ' << v;
            }
            out << '\n';
        }
    }

    void Gallery::load(std::filesystem::path const &path)
    {
        entries.clear();
        std::ifstream in(path);
        if (!in)
        {
            return; // belum ada galeri = kosong, bukan error
        }
        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream stream(line);
            Entry entry;
            stream >> entry.name;
            float v = 0.0F;
            while (stream >> v)
            {
                entry.embedding.push_back(v);
            }
            if (!entry.name.empty() && !entry.embedding.empty())
            {
                entries.push_back(std::move(entry));
            }
        }
    }
}
