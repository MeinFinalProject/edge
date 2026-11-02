#pragma once
#include <d3d12.h>
#include <winrt/base.h>
#include <cstddef>
#include <cstdint>
#include "nv12_bridge.hpp"

namespace FaceRuntime::Gpu
{
    // "Satu program shader siap pakai" = Root Signature + PSO.
    // Semua shader tensorize kita punya bentuk colokan yang SAMA:
    //   t0,t1  : SRV NV12 (Y dan UV)            -> descriptor table   (root param 0)
    //   b0     : konstanta (N x 32-bit)          -> root constants     (root param 1)
    //   u0     : tensor output                   -> root UAV           (root param 2)
    //   u1     : (opsional) counter OOB          -> root UAV           (root param 3)
    //   s0     : sampler linear + clamp          -> static sampler
    class ComputeKernel
    {
    public:
        ComputeKernel(
            ID3D12Device *device,
            const void *shaderBytecode,
            std::size_t shaderBytecodeSize,
            std::uint32_t numRootConstants, // jumlah DWORD (32-bit) di cbuffer b0
            std::uint32_t numUavs);         // 1 atau 2

        // Memasang semua colokan ke command list. Setelah ini tinggal cl->Dispatch(...).
        void bind(
            ID3D12GraphicsCommandList *commandList,
            Nv12Frame const &frame,
            const void *constants,
            ID3D12Resource *uav0,
            ID3D12Resource *uav1 = nullptr) const;

    private:
        winrt::com_ptr<ID3D12RootSignature> rootSignature;
        winrt::com_ptr<ID3D12PipelineState> pipelineState;
        std::uint32_t numRootConstants;
        std::uint32_t numUavs;
    };
}