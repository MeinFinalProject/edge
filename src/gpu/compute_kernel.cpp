#include "compute_kernel.hpp"
#include <stdexcept>
#include <string>

namespace FaceRuntime::Gpu
{
    ComputeKernel::ComputeKernel(
        ID3D12Device *device,
        const void *shaderBytecode,
        std::size_t shaderBytecodeSize,
        std::uint32_t numRootConstants,
        std::uint32_t numUavs) : numRootConstants(numRootConstants),
                                 numUavs(numUavs)
    {
        if (numUavs < 1 || numUavs > 2)
        {
            throw std::invalid_argument("ComputeKernel: numUavs harus 1 atau 2.");
        }

        // parameter root 0 = descriptor table untuk SRV NV12 (Y dan UV) -> t0 t1
        D3D12_DESCRIPTOR_RANGE srvRange{};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = 2;     // t0,t1
        srvRange.BaseShaderRegister = 0; // mulai dari t0
        srvRange.RegisterSpace = 0;      // ruang register 0
        srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        D3D12_ROOT_PARAMETER rootParams[4]{};

        // [Root Parameter 0] mendefinisikan Argumen ke-1 dari fungsi Shader.
        rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParams[0].DescriptorTable.NumDescriptorRanges = 1;
        rootParams[0].DescriptorTable.pDescriptorRanges = &srvRange;
        rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        // [Root Parameter 1] mendefinisikan Argumen ke-2 dari fungsi Shader.
        rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[1].Constants.ShaderRegister = 0; // b0
        rootParams[1].Constants.RegisterSpace = 0;  // ruang register 0
        rootParams[1].Constants.Num32BitValues = numRootConstants;
        rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        // --- Parameter 2 (dan 3): UAV buffer langsung lewat alamat GPU ---
        for (std::uint32_t i = 0; i < numUavs; ++i)
        {
            rootParams[2 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParams[2 + i].Descriptor.ShaderRegister = i; // u0, u1
            rootParams[2 + i].Descriptor.RegisterSpace = 0;  // ruang register 0
            rootParams[2 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }

        // --- Sampler s0: bilinear + clamp, ditanam langsung di root signature ---
        D3D12_STATIC_SAMPLER_DESC samplerDesc{};
        samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplerDesc.MipLODBias = 0.0f;
        samplerDesc.MaxAnisotropy = 1;
        samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        samplerDesc.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        samplerDesc.MinLOD = 0.0f;
        samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
        samplerDesc.ShaderRegister = 0; // s0
        samplerDesc.RegisterSpace = 0;  // ruang register 0
        samplerDesc.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        // --- Root Signature ---
        D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc{};
        rootSignatureDesc.NumParameters = 2 + numUavs;
        rootSignatureDesc.pParameters = rootParams;
        rootSignatureDesc.NumStaticSamplers = 1;
        rootSignatureDesc.pStaticSamplers = &samplerDesc;
        rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        // --- Serialisasi Root Signature ---
        winrt::com_ptr<ID3DBlob> signatureBlob;
        winrt::com_ptr<ID3DBlob> errorBlob;
        const HRESULT hr = D3D12SerializeRootSignature(
            &rootSignatureDesc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            signatureBlob.put(),
            errorBlob.put());
        if (FAILED(hr))
        {
            std::string message = "D3D12SerializeRootSignature gagal";
            if (errorBlob)
            {
                message += ": ";
                message.append(static_cast<const char *>(errorBlob->GetBufferPointer()),
                               errorBlob->GetBufferSize());
            }
            throw std::runtime_error(message);
        }

        winrt::check_hresult(device->CreateRootSignature(
            0, signatureBlob->GetBufferPointer(), signatureBlob->GetBufferSize(),
            IID_PPV_ARGS(rootSignature.put())));

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rootSignature.get();
        psoDesc.CS.pShaderBytecode = shaderBytecode;
        psoDesc.CS.BytecodeLength = shaderBytecodeSize;
        psoDesc.NodeMask = 0;
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        winrt::check_hresult(device->CreateComputePipelineState(
            &psoDesc, IID_PPV_ARGS(pipelineState.put())));
    }

    void ComputeKernel::bind(
        ID3D12GraphicsCommandList *commandList,
        Nv12Frame const &frame,
        const void *constants,
        ID3D12Resource *uav0,
        ID3D12Resource *uav1) const
    {
        commandList->SetComputeRootSignature(rootSignature.get());
        commandList->SetPipelineState(pipelineState.get());

        ID3D12DescriptorHeap *heaps[] = {frame.srvHeap};
        commandList->SetDescriptorHeaps(1, heaps);

        commandList->SetComputeRootDescriptorTable(
            0,
            frame.srvHeap->GetGPUDescriptorHandleForHeapStart());
        commandList->SetComputeRoot32BitConstants(
            1,
            numRootConstants,
            constants,
            0);
        commandList->SetComputeRootUnorderedAccessView(
            2,
            uav0->GetGPUVirtualAddress());
        if (numUavs == 2)
        {
            if (!uav1)
            {
                throw std::invalid_argument("ComputeKernel::bind: kernel butuh uav1 tetapi null.");
            }
            commandList->SetComputeRootUnorderedAccessView(3, uav1->GetGPUVirtualAddress());
        }
    }
}