#include "nv12_helpers.hlsl"

Texture2D<float> InputY : register(t0);
Texture2D<float2> InputUV : register(t1);
SamplerState LinearClamp : register(s0);
RWByteAddressBuffer OutputTensor : register(u0);

cbuffer Params : register(b0) {
    uint SrcWidth;
    uint SrcHeight;
    uint DstWidth;
    uint DstHeight;

    uint ResizedWidth;
    uint ResizedHeight;
    float PadY;
    float Reserved;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= DstWidth || tid.y >= DstHeight) {
        return;
    }

    const uint plane = DstWidth * DstHeight; // Total piksel per 1 channel (409,600)
    const uint pixel = tid.y * DstWidth + tid.x; // Indeks piksel unik thread ini
    const float2 dst = float2(tid.x + 0.5f, tid.y + 0.5f); // Titik tengah piksel target

    // normalize -127.5f / 128.0f black padding
    float3 value = float3(-0.99609375f, -0.99609375f, -0.99609375f);

    if (tid.x < ResizedWidth &&
        tid.y < ResizedHeight)
    {
        // UV berdasarkan ukuran resize aktual, bukan satu Scale.
        const float2 uv = dst / float2(
            (float)ResizedWidth,
            (float)ResizedHeight
        );

        const float ySample =
            InputY.SampleLevel(LinearClamp, uv, 0.0f);

        const float2 uvSample =
            InputUV.SampleLevel(LinearClamp, uv, 0.0f);
        
        // Normalisasi SCRFD: (val - 127.5) / 128.0
        value = (NV12ToRGB(ySample, uvSample) * 255.0f - 127.5f) / 128.0f;
    }

    OutputTensor.Store(pixel * 4, asuint(value.r));
    OutputTensor.Store((plane + pixel) * 4, asuint(value.g));
    OutputTensor.Store((2 * plane + pixel) * 4, asuint(value.b));
}