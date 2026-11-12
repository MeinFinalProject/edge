// Alat offline untuk liveness SPRT. Tidak butuh kamera, GPU, atau Windows SDK.
//
//   sprt_tool fit    <file|folder> [bins=12]                  -> tabel LLR ke stdout
//   sprt_tool replay <file|folder> <llr.txt> [alpha beta weight clip maxObs stride]
//   sprt_tool vote   <file|folder> [threshold=0.8] [stride=1] -> pembanding: voting 2-dari-3
//
// <folder> = semua *.csv di dalamnya dibaca. Format baris CSV (dari mode `record`):
//   seq,label,frame,p        label 1 = wajah asli, 0 = spoof; satu seq = satu sesi (satu track)
// `fit` juga menerima format ringkas  label,p
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "../src/models/liveness.hpp"

using namespace FaceRuntime::Liveness;
namespace fs = std::filesystem;

namespace
{
    struct Session
    {
        int label = 0;
        std::vector<float> p; // urut frame
    };

    // Membaca satu file atau semua *.csv dalam satu folder. Baris rusak dilewati (dihitung).
    std::map<std::string, Session> LoadSessions(std::string const &pathText)
    {
        std::vector<fs::path> files;
        const fs::path path(pathText);
        if (fs::is_directory(path))
        {
            for (auto const &entry : fs::directory_iterator(path))
                if (entry.path().extension() == ".csv")
                    files.push_back(entry.path());
        }
        else
        {
            files.push_back(path);
        }
        if (files.empty())
        {
            std::cerr << "tidak ada file .csv di " << pathText << "\n";
            std::exit(1);
        }

        std::map<std::string, Session> sessions;
        int skipped = 0;
        int anonymous = 0;
        for (auto const &file : files)
        {
            std::ifstream in(file);
            if (!in)
            {
                std::cerr << "tidak bisa membuka " << file.string() << "\n";
                std::exit(1);
            }
            std::string line;
            while (std::getline(in, line))
            {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (line.empty() || line.rfind("seq,label", 0) == 0 || line.rfind("label,p", 0) == 0)
                    continue; // header
                std::vector<std::string> cells;
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ','))
                    cells.push_back(cell);
                try
                {
                    if (cells.size() >= 4)
                    {
                        auto &s = sessions[file.stem().string() + "|" + cells[0]];
                        s.label = std::stoi(cells[1]);
                        s.p.push_back(std::stof(cells[3]));
                        if (s.label != 0 && s.label != 1)
                            throw std::invalid_argument("label");
                    }
                    else if (cells.size() == 2)
                    {
                        auto &s = sessions["anon" + std::to_string(anonymous++)]; // format ringkas
                        s.label = std::stoi(cells[0]);
                        s.p.push_back(std::stof(cells[1]));
                        if (s.label != 0 && s.label != 1)
                            throw std::invalid_argument("label");
                    }
                    else
                    {
                        throw std::invalid_argument("kolom");
                    }
                }
                catch (std::exception const &)
                {
                    ++skipped;
                }
            }
        }
        if (skipped > 0)
            std::cerr << "peringatan: " << skipped << " baris rusak dilewati\n";
        return sessions;
    }

    double Percent(int part, int whole) { return whole > 0 ? 100.0 * part / whole : 0.0; }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "pemakaian: lihat komentar di bagian atas tools/sprt_tool.cpp\n";
        return 1;
    }
    const std::string command = argv[1];
    auto sessions = LoadSessions(argv[2]);

    if (command == "fit")
    {
        std::vector<float> live, spoof;
        for (auto const &[id, s] : sessions)
            for (float p : s.p)
                (s.label == 1 ? live : spoof).push_back(p);
        const int bins = argc > 3 ? std::stoi(argv[3]) : 12;
        auto table = LlrTable::Fit(live, spoof, bins, 6.0F);
        table.save(std::cout);
        std::cerr << "observasi asli=" << live.size() << " spoof=" << spoof.size() << " bin=" << bins << "\n";
        return 0;
    }

    int n[2] = {0, 0}, asLive[2] = {0, 0}, asSpoof[2] = {0, 0}, undecided[2] = {0, 0};
    double observations[2] = {0, 0};

    if (command == "replay" && argc >= 4)
    {
        std::ifstream tableFile(argv[3]);
        if (!tableFile)
        {
            std::cerr << "tidak bisa membuka tabel LLR " << argv[3] << "\n";
            return 1;
        }
        LlrTable table = LlrTable::Load(tableFile);
        SprtConfig config;
        if (argc > 4)
            config.alpha = std::stof(argv[4]);
        if (argc > 5)
            config.beta = std::stof(argv[5]);
        if (argc > 6)
            config.weight = std::stof(argv[6]);
        if (argc > 7)
            config.llrClip = std::stof(argv[7]);
        if (argc > 8)
            config.maxObservations = std::stoi(argv[8]);
        const std::size_t stride = argc > 9 ? static_cast<std::size_t>(std::stoi(argv[9])) : 1;

        for (auto const &[id, s] : sessions)
        {
            SprtAccumulator acc(config, table);
            for (std::size_t i = 0; i < s.p.size() && acc.verdict() == Verdict::Pending; i += stride)
                acc.observe(s.p[i]);
            ++n[s.label];
            observations[s.label] += acc.count();
            switch (acc.verdict())
            {
            case Verdict::Live:
                ++asLive[s.label];
                break;
            case Verdict::Spoof:
                ++asSpoof[s.label];
                break;
            default:
                ++undecided[s.label];
                break; // Pending (data habis) dan Undecided = belum diputuskan
            }
        }
        std::cout << "SPRT alpha=" << config.alpha << " beta=" << config.beta << " weight=" << config.weight
                  << " clip=" << config.llrClip << " maxObs=" << config.maxObservations << " stride=" << stride << "\n";
    }
    else if (command == "vote")
    {
        const float threshold = argc > 3 ? std::stof(argv[3]) : 0.8F;
        const std::size_t stride = argc > 4 ? static_cast<std::size_t>(std::stoi(argv[4])) : 1;
        for (auto const &[id, s] : sessions)
        {
            int yes = 0, no = 0, used = 0;
            for (std::size_t i = 0; i < s.p.size() && yes < 2 && no < 2; i += stride)
            {
                (s.p[i] >= threshold ? yes : no)++;
                ++used;
            }
            ++n[s.label];
            observations[s.label] += used;
            if (yes >= 2)
                ++asLive[s.label];
            else if (no >= 2)
                ++asSpoof[s.label];
            else
                ++undecided[s.label]; // data habis sebelum 2 suara sama
        }
        std::cout << "VOTING 2-dari-3, ambang p=" << threshold << " stride=" << stride << "\n";
    }
    else
    {
        std::cerr << "perintah tidak dikenal: " << command << "\n";
        return 1;
    }

    std::cout << "  SPOOF  sesi=" << n[0] << "  FAR(lolos sebagai Live)=" << Percent(asLive[0], n[0])
              << "%  benar(Spoof)=" << Percent(asSpoof[0], n[0]) << "%  belum diputuskan=" << Percent(undecided[0], n[0])
              << "%  rata2 observasi=" << (n[0] ? observations[0] / n[0] : 0.0) << "\n";
    std::cout << "  ASLI   sesi=" << n[1] << "  FRR(ditolak sebagai Spoof)=" << Percent(asSpoof[1], n[1])
              << "%  benar(Live)=" << Percent(asLive[1], n[1]) << "%  belum diputuskan=" << Percent(undecided[1], n[1])
              << "%  rata2 observasi=" << (n[1] ? observations[1] / n[1] : 0.0) << "\n";
    return 0;
}
