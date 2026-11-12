#include "warp.hpp"
#include <stdexcept>
#include "../gpu/d3d12_helpers.hpp"
#include "warp_tensorize.h"

#ifndef FACE_GROUP_SIZE
#error "FACE_GROUP_SIZE harus didefinisikan oleh CMake (target_compile_definitions)."
#endif

namespace FaceRuntime::Preprocess
{
    namespace
    {
        // Ukuran thread group. Sumbernya satu: GROUP_SIZE
        // di CMakeLists.txt, diteruskan ke fxc
        constexpr std::uint32_t kGroupSize = FACE_GROUP_SIZE;

        // HARUS identik urutannya dengan `cbuffer Params : register(b0)`
        // di warp_tensorize.hlsl.
        // Setiap field = 4 byte = 1 DWORD.
        // Dikirim lewat SetComputeRoot32BitConstants.
        struct WarpParams
        {
            std::uint32_t srcWidth;
            std::uint32_t srcHeight;
            std::uint32_t dstWidth;
            std::uint32_t dstHeight;
            float m00, m01, m02; // baris 1 matriks dst -> src
            float m10, m11, m12; // baris 2
            float normScale;
            float normBias;
            std::uint32_t oobMode;   // OobMode
            std::uint32_t reserved0; // padding ke 16 DWORD: cbuffer HLSL dibulatkan ke kelipatan 16 byte,
            std::uint32_t reserved1; // jadi root constants dibuat selebar itu agar ukurannya persis sama
            std::uint32_t reserved2;
        };
        static_assert(sizeof(WarpParams) == 16 * 4, "WarpParams harus 16 DWORD");
    };

    WarpContext::WarpContext(
        Gpu::GpuContext &gpu) : kernel(gpu.device(),
                                       g_warp_tensorize,
                                       sizeof(g_warp_tensorize),
                                       sizeof(WarpParams) / 4),
                                recorder(gpu) {};

    WarpPreprocessor::WarpPreprocessor(
        Gpu::GpuContext &gpu,
        WarpContext &context,
        FrameSize source,
        TensorSpec const &spec)
        : gpu(gpu),
          context(context),
          source(source),
          spec_(spec),
          tensor(gpu,
                 {1,
                  3,
                  static_cast<std::int64_t>(spec.height),
                  static_cast<std::int64_t>(spec.width)})
    {
        if (source.width == 0 || source.height == 0)
        {
            throw std::invalid_argument("WarpPreprocessor: ukuran frame sumber 0.");
        }
    };

    winrt::Windows::AI::MachineLearning::TensorFloat
    WarpPreprocessor::run(
        Gpu::Nv12Frame const &frame,
        std::array<float, 6> const &m)
    {
        WarpParams params{};
        params.srcWidth = source.width;
        params.srcHeight = source.height;
        params.dstWidth = spec().width;
        params.dstHeight = spec().height;
        params.m00 = m[0];
        params.m01 = m[1];
        params.m02 = m[2];
        params.m10 = m[3];
        params.m11 = m[4];
        params.m12 = m[5];
        params.normScale = spec_.normScale;
        params.normBias = spec_.normBias;
        params.oobMode = static_cast<std::uint32_t>(spec_.oob);

        // buat command list kosong
        auto *commandList = context.recorder.begin();

        // pasang semua colokan dan state ke command list
        context.kernel.bind(
            commandList,
            frame,
            &params,
            tensor.resource());

        // eksekusi dengan Dispatch.
        // Ukuran thread group = kGroupSize x kGroupSize.
        std::uint32_t groupX = (spec_.width + kGroupSize - 1) / kGroupSize;
        std::uint32_t groupY = (spec_.height + kGroupSize - 1) / kGroupSize;
        commandList->Dispatch(groupX, groupY, 1);

        // Pastikan semua penulisan shader tuntas sebelum WinML membaca tensor.
        D3D12_RESOURCE_BARRIER uav = Gpu::UavBarrier(tensor.resource());
        commandList->ResourceBarrier(1, &uav);

        // kirim command list ke antrean GPU
        context.recorder.submit();

        // kembalikan tensor input model (GPU)
        return tensor.tensor();
    }
}