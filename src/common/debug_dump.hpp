#pragma once
#include <filesystem>
#include <fstream>
#include <vector>

namespace FaceRuntime::Common
{
    // Tulis float mentah (little-endian, tanpa header) -> dibaca skrip tools/*.py
    inline void WriteFloatsToFile(std::filesystem::path const &path, std::vector<float> const &values)
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
}
