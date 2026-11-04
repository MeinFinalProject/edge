#pragma once
#include <windows.h>
#include <filesystem>
#include <string>

namespace FaceRuntime::Common
{
    // Folder tempat FaceRuntime.exe berada (bukan "current directory" yang bisa berubah-ubah).
    inline std::filesystem::path ExecutableDirectory()
    {
        wchar_t buffer[MAX_PATH]{};
        const DWORD length = ::GetModuleFileNameW(nullptr, buffer, MAX_PATH);
        return std::filesystem::path(std::wstring(buffer, length)).parent_path();
    }
}
