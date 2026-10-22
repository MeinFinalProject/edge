#include "similarity_transform.hpp"

namespace FaceRuntime::Preprocess
{
    SimilarityTransform2D EstimateSimilarityTransform(
        std::array<Common::Point2f, 5> const &source,
        std::array<Common::Point2f, 5> const &target)
    {
        double normalMatrix[4][4] = {0};
        double normalVector[4] = {0};

        auto accumulateRow = [&normalMatrix, &normalVector](std::array<double, 4> const &row, double val)
        {
            for (int i = 0; i < 4; ++i)
            {
                normalVector[i] += row[i] * val;
                for (int j = 0; j < 4; ++j)
                {
                    normalMatrix[i][j] += row[i] * row[j];
                }
            }
        };

        for (std::size_t i = 0; i < 5; ++i)
        {
            const auto [x, y] = source[i];
            const auto [u, v] = target[i];
            accumulateRow({x, -y, 1.0, 0.0}, u);
            accumulateRow({y, x, 0.0, 1.0}, v);
        }

        // partial pivoting gaussian elimination
        for (int i = 0; i < 4; ++i)
        {
            int pivot = i;
            for (int j = i + 1; j < 4; ++j)
            {
                if (std::abs(normalMatrix[j][i]) > std::abs(normalMatrix[pivot][i]))
                {
                    pivot = j;
                }
            }
            if (std::abs(normalMatrix[pivot][i]) < 1e-12)
            {
                throw std::runtime_error("Landmark tidak menghasilkan transform valid (Singular)");
            }
            if (pivot != i)
            {
                for (int k = 0; k < 4; ++k)
                {
                    std::swap(normalMatrix[i][k], normalMatrix[pivot][k]);
                }
                std::swap(normalVector[i], normalVector[pivot]);
            }
            for (int j = i + 1; j < 4; ++j)
            {
                double f = normalMatrix[j][i] / normalMatrix[i][i];
                for (int k = i + 1; k < 4; ++k)
                {
                    normalMatrix[j][k] -= f * normalMatrix[i][k];
                }
                normalVector[j] -= f * normalVector[i];
            }
        }

        std::array<double, 4> transformParams = {};
        for (int i = 3; i >= 0; --i)
        {
            transformParams[i] = normalVector[i];

            for (int j = i + 1; j < 4; ++j)
            {
                transformParams[i] -= normalMatrix[i][j] * transformParams[j];
            }
            transformParams[i] /= normalMatrix[i][i];
        }

        const double a = transformParams[0], b = transformParams[1], tx = transformParams[2], ty = transformParams[3];
        const double determinant = a * a + b * b;
        if (determinant < 1e-12)
        {
            throw std::runtime_error("Landmark tidak menghasilkan transform valid (Determinant terlalu kecil)");
        }

        SimilarityTransform2D result;
        result.srctoDst = {static_cast<float>(a),
                           static_cast<float>(-b),
                           static_cast<float>(tx),
                           static_cast<float>(b),
                           static_cast<float>(a),
                           static_cast<float>(ty)};
        result.dsttoSrc = {static_cast<float>(a / determinant),
                           static_cast<float>(b / determinant),
                           static_cast<float>(-(a * tx + b * ty) / determinant),
                           static_cast<float>(-b / determinant),
                           static_cast<float>(a / determinant),
                           static_cast<float>((b * tx - a * ty) / determinant)};

        float squaredErrorSum = 0.0F;
        for (std::size_t i = 0; i < 5; ++i)
        {
            float estimatedTargetX = result.srctoDst[0] * source[i].x + result.srctoDst[1] * source[i].y + result.srctoDst[2];
            float estimatedTargetY = result.srctoDst[3] * source[i].x + result.srctoDst[4] * source[i].y + result.srctoDst[5];

            squaredErrorSum += (estimatedTargetX - target[i].x) * (estimatedTargetX - target[i].x) +
                               (estimatedTargetY - target[i].y) * (estimatedTargetY - target[i].y);
        }
        result.rmse = std::sqrt(squaredErrorSum / 5.0F);

        return result;
    }
}