#pragma once
#include <array>
#include "../common/point2f.hpp"

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

    constexpr std::array<FaceRuntime::Common::Point2f, 5> kLivenessTemplate80{
        {{32.03F, 38.06F},
         {47.89F, 37.98F},
         {40.01F, 47.08F},
         {33.50F, 56.36F},
         {46.63F, 56.29F}}};
    constexpr std::array<Common::Point2f, 5> kArcFaceTemplate112{
        {{38.2946F, 51.6963F},
         {73.5318F, 51.5014F},
         {56.0252F, 71.7366F},
         {41.5493F, 92.3655F},
         {70.7299F, 92.2041F}}};

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