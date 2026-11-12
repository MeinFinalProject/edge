#include "nv12_helpers.hlsl"

Texture2D<float>  InputY  : register(t0);
Texture2D<float2> InputUV : register(t1);
SamplerState LinearClamp : register(s0);
RWByteAddressBuffer OutputTensor : register(u0);

// 16 DWORD = 4 register penuh.
cbuffer Params : register(b0) {
    uint SrcWidth;
    uint SrcHeight;
    uint DstWidth;
    uint DstHeight;

    float M00;   // baris 1 matriks dst -> src (koordinat kontinu, pusat piksel di +0.5)
    float M01;
    float M02;
    float M10;   // baris 2
    float M11;
    float M12;

    float NormScale;
    float NormBias;

    uint OobMode;       // 0 = Clamp (replikasi tepi), 1 = Fill (isi NormBias = hitam ternormalisasi)
    uint Reserved0;
    uint Reserved1;
    uint Reserved2;
}

#define OOB_CLAMP 0
#define OOB_FILL  1

[numthreads(GROUP, GROUP, 1)]   // GROUP didefinisikan CMake lewat fxc /D GROUP=...
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= DstWidth || tid.y >= DstHeight) {
        return;
    }

    const uint plane = DstWidth * DstHeight;
    const uint pixel = tid.y * DstWidth + tid.x;
    const float2 dst = float2(tid.x + 0.5f, tid.y + 0.5f);   // titik tengah piksel tujuan

    // Inverse warp: "piksel tujuan ini harus mengambil dari koordinat mana di frame kamera?"
    // ----------------------------------------------------------------------------------------------
    // MODALITAS 1: ALIGNMENT WAJAH BERDASARKAN LANDMARKS (Untuk ArcFace & Liveness)
    // ----------------------------------------------------------------------------------------------
    // 1. CPU menghitung matriks Transformasi geometris 2D berbasis rotasi, skala, dan translasi. 
    //    yang mencocokkan 5 titik landmark wajah (2 mata, hidung, 2 sudut bibir) dari SCRFD 
    //    ke template foto.
    // 2. CPU me-inverse matriks tersebut (dstToSrc matrix) dan mengirim angka (M00..M12) ke cbuffer ini.
    // 3. Efeknya di HLSL:
    //    - M00, M01, M10, M11 : Mengamputasi kemiringan kepala (Rotasi) & men-zoom wajah (Scale).
    //    - M02, M12           : Memindahkan posisi (Translasi) agar mata & hidung pas di tengah.
    //    - Hasil: Wajah miring 45 derajat di kamera otomatis menjadi "Foto Paspor Tegak Lurus".
    //
    // ----------------------------------------------------------------------------------------------
    // MODALITAS 2: RESIZE & LETTERBOXING PRESISI (Untuk Detector SCRFD 640x640)
    // ----------------------------------------------------------------------------------------------
    // SCRFD tidak butuh rotasi landmark, ia hanya butuh Resize + Aspect Ratio Preserved.
    // Tanpa perlu membuat shader baru, CPU cukup mengirimkan "Matriks Skala & Geser":
    //    - M01 = 0.0, M10 = 0.0 (Tidak ada rotasi).
    //    - M00 = SrcWidth / ScaledWidth  (Faktor lompatan piksel horizontal).
    //    - M11 = SrcHeight / ScaledHeight (Faktor lompatan piksel vertikal).
    //    - M02, M12 = Offset piksel (Margin hitam / Letterbox padding).
    float2 src = float2(M00 * dst.x + M01 * dst.y + M02,
                        M10 * dst.x + M11 * dst.y + M12);

    const bool outside = src.x < 0.0f || src.x >= (float)SrcWidth ||
                         src.y < 0.0f || src.y >= (float)SrcHeight;

    // Fill: area di luar frame berisi NormBias (SCRFD: hitam = (0 - 127.5) / 128 = -0.99609375).
    float3 value = float3(NormBias, NormBias, NormBias);

    if (!(outside && OobMode == OOB_FILL)) {
        if (OobMode == OOB_CLAMP) {
            // Replikasi tepi: koordinat di luar frame dipaksa mentok ke piksel paling pinggir.
            src = clamp(src, float2(0.0f, 0.0f),
                        float2((float)SrcWidth - 1.0f, (float)SrcHeight - 1.0f));
        }
        const float2 uv = src / float2((float)SrcWidth, (float)SrcHeight);
        const float  ySample  = InputY.SampleLevel(LinearClamp, uv, 0.0f);
        const float2 uvSample = InputUV.SampleLevel(LinearClamp, uv, 0.0f);
        value = NV12ToRGB(ySample, uvSample) * 255.0f * NormScale + NormBias;
    }

    // Layout NCHW: plane R, lalu G, lalu B.
    OutputTensor.Store(pixel * 4, asuint(value.r));
    OutputTensor.Store((plane + pixel) * 4, asuint(value.g));
    OutputTensor.Store((2 * plane + pixel) * 4, asuint(value.b));
}
