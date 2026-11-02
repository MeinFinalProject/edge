#pragma once
#include <cstdint>

// Struct di bawah HARUS identik urutannya dengan `cbuffer Params : register(b0)` di HLSL.
// Setiap field = 4 byte = 1 DWORD. Dikirim lewat SetComputeRoot32BitConstants.
namespace FaceRuntime::Preprocess
{
    // scrfd_tensorize.hlsl  (8 DWORD)
    struct ScrfdConstants
    {
        std::uint32_t srcWidth;
        std::uint32_t srcHeight;
        std::uint32_t dstWidth;
        std::uint32_t dstHeight;
        std::uint32_t resizedWidth;
        std::uint32_t resizedHeight;
        float reserved0; // (di HLSL lama bernama PadY; tidak dipakai karena padding kanan-bawah)
        float reserved1;
    };
    static_assert(sizeof(ScrfdConstants) == 8 * 4, "ScrfdConstants harus 8 DWORD");

    // pad_tensorize.hlsl dan arcface_tensorize.hlsl  (12 DWORD)
    struct WarpConstants
    {
        std::uint32_t srcWidth;
        std::uint32_t srcHeight;
        std::uint32_t dstWidth;
        std::uint32_t dstHeight;
        float m00, m01, m02; // baris 1 matriks dst -> src
        float m10, m11, m12; // baris 2
        float reserved1;
        float reserved2;
    };
    static_assert(sizeof(WarpConstants) == 12 * 4, "WarpConstants harus 12 DWORD");
}
