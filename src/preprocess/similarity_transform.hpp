#pragma once
#include <array>
#include "../common/types.hpp"

namespace FaceRuntime::Preprocess
{
    // Matriks 2x3 yang disimpan sebagai 6 float (row-major):
    //   [ m0 m1 m2 ]      x' = m0*x + m1*y + m2
    //   [ m3 m4 m5 ]      y' = m3*x + m4*y + m5
    struct SimilarityTransform2D
    {
        std::array<float, 6> srctoDst{};
        std::array<float, 6> dsttoSrc{};
        float rmse = 0.0F;
    };

    SimilarityTransform2D EstimateSimilarityTransform(
        std::array<Common::Point2f, 5> const &source,
        std::array<Common::Point2f, 5> const &target);

    // Berapa persen piksel crop (dstWidth x dstHeight) yang, setelah dipetakan balik
    // memakai dsttoSrc, jatuh DI LUAR frame sumber (srcWidth x srcHeight).
    // Logikanya sama persis dengan penghitung OOB di pad_tensorize.hlsl, tetapi di CPU.
    float ComputeOutOfBoundsRatio(
        SimilarityTransform2D const &transform,
        std::uint32_t dstWidth, std::uint32_t dstHeight,
        std::uint32_t srcWidth, std::uint32_t srcHeight);
}