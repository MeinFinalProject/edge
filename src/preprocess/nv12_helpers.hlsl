#ifndef NV12_HELPERS_HLSL
#define NV12_HELPERS_HLSL

float3 NV12ToRGB(float ySample, float2 uvSample) {
    float y = (ySample * 255.0f - 16.0f) / 219.0f;
    float cb = (uvSample.x * 255.0f - 128.0f) / 224.0f;
    float cr = (uvSample.y * 255.0f - 128.0f) / 224.0f;

    return saturate(float3(
        y + 1.5748f * cr,
        y - 0.187324f * cb - 0.468124f * cr,
        y + 1.8556f * cb
    ));
}

#endif 