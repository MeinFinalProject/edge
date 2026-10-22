#include "nv12_helpers.hlsl"

Texture2D<float>  InputY  : register(t0);
Texture2D<float2> InputUV : register(t1);
SamplerState LinearClamp : register(s0);
RWByteAddressBuffer OutputTensor : register(u0);
RWByteAddressBuffer OobCounter : register(u1);

cbuffer Params : register(b0) {
    uint SrcWidth; 
    uint SrcHeight; 
    uint DstWidth; 
    uint DstHeight;
    float m00;
    float m01; 
    float m02; 
    float m10; 
    float m11; 
    float m12;
    float Reserved1; 
    float Reserved2;
};

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= DstWidth || tid.y >= DstHeight) return;

    const uint plane = DstWidth * DstHeight;
    const uint pixel = tid.y * DstWidth + tid.x;
    
    // Titik koordinat piksel di kotak 80x80 (ditambah 0.5 biar ngambil di titik tengah piksel)
    const float2 dst = float2(tid.x + 0.5f, tid.y + 0.5f);

    // ========================================================
    // TAHAP 1: INVERSE WARP (Membaca Rumus CPU)
    // ========================================================
    // GPU lagi berdiri di kanvas kosong 80x80.
    // Dia nanya: "Untuk piksel (x, y) di sini, gue harus nyomot di koordinat kamera mana?"
    // Angka m00 sampai m12 ini adalah array "dst_to_src" yang dikirim CPU!
    float srcX = m00 * dst.x + m01 * dst.y + m02;
    float srcY = m10 * dst.x + m11 * dst.y + m12;

    // ========================================================
    // TAHAP 2: SATPAM OUT-OF-BOUNDS (OOB)
    // ========================================================
    // Kalau hasil comotan lokasinya ada di luar layar kamera (misal X = -10, atau Y = 2000)
    if (srcX < 0.0f || srcX >= (float)SrcWidth || srcY < 0.0f || srcY >= (float)SrcHeight) {
        uint dummy;
        OobCounter.InterlockedAdd(0, 1, dummy); 
        // InterlockedAdd = Penambahan Atomik.
        // Karena ada puluhan ribu thread GPU yang jalan BERBARENGAN, 
        // kalau mereka nambah nilai bareng-bareng bisa tabrakan.
        // "Interlocked" memaksa GPU ngantre sepersekian nanodetik buat nambahin angka 1 ini,
        // biar perhitungannya akurat nggak ada yang ke-skip.
    }

    // ========================================================
    // TAHAP 3: BORDER REPLICATE (Maksa Nyomot Pinggiran)
    // ========================================================
    // Meskipun pikselnya keluar batas, GPU nggak boleh error. 
    // Kita paksa mentok (clamp) ke piksel paling pinggir dari kamera.
    // Jadi kalau koordinatnya X = -10, dia bakal nyomot piksel di X = 0.
    srcX = clamp(srcX, 0.0f, (float)SrcWidth - 1.0f);
    srcY = clamp(srcY, 0.0f, (float)SrcHeight - 1.0f);

    // ========================================================
    // TAHAP 4: NORMALISASI PAD
    // ========================================================
    const float2 uv = float2(srcX / (float)SrcWidth, srcY / (float)SrcHeight);
    const float ySample = InputY.SampleLevel(LinearClamp, uv, 0.0f);
    const float2 uvSample = InputUV.SampleLevel(LinearClamp, uv, 0.0f);
    
    // Kontrak model PAD minta normalisasi: Nilai_Warna / 255.0
    // KITA NGGAK PERLU NGITUNG APA-APA LAGI!
    // Kenapa? Karena fungsi `NV12ToRGB` kita itu hasilnya udah (0.0 sampai 1.0),
    // yang mana itu setara dengan warna yang udah dibagi 255!
    float3 value = NV12ToRGB(ySample, uvSample);

    // ========================================================
    // TAHAP 5: TULIS KE NCHW
    // ========================================================
    OutputTensor.Store(pixel * 4, asuint(value.r));
    OutputTensor.Store((plane + pixel) * 4, asuint(value.g));
    OutputTensor.Store((2 * plane + pixel) * 4, asuint(value.b));
}