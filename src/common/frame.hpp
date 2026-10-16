#pragma once

#include <cstdint>
#include <d3d11.h>
#include <winrt/base.h>

namespace FaceRuntime
{

    struct Frame
    {
        winrt::com_ptr<ID3D11Texture2D> texture;
        std::uint64_t frameId;

        [[nodiscard]] bool valid() const noexcept
        {
            return texture != nullptr;
        }
    };

}