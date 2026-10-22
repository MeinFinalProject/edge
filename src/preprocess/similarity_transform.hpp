#pragma once
#include <array>
#include "preprocess_contracts.hpp"
#include "../common/point2f.hpp"

namespace FaceRuntime::Preprocess
{
    constexpr std::array<FaceRuntime::Common::Point2f, 5> kLivenessTemplate80{{{32.03F, 38.06F}, {47.89F, 37.98F}, {40.01F, 47.08F}, {33.50F, 56.36F}, {46.63F, 56.29F}}};
    constexpr std::array<Common::Point2f, 5> kArcFaceTemplate112{{{38.2946F, 51.6963F}, {73.5318F, 51.5014F}, {56.0252F, 71.7366F}, {41.5493F, 92.3655F}, {70.7299F, 92.2041F}}};

    SimilarityTransform2D EstimateSimilarityTransform(
        std::array<Common::Point2f, 5> const &source,
        std::array<Common::Point2f, 5> const &target);
}