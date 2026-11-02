#pragma once
#include <d3d12.h>
#include <winrt/base.h>
#include "gpu_context.hpp"

namespace FaceRuntime::Gpu
{
    // record command list gpu yang bisa dipake ulang.
    //
    //   auto *cl = recorder.begin();   // buka buku baru (kosong)
    //   ... isi perintah lewat `cl` ...
    //   recorder.submit();             // kirim ke GPU, TIDAK menunggu
    //   // atau:
    //   recorder.submitAndWait();      // kirim, lalu tunggu GPU selesai (dipakai bila CPU akan membaca hasilnya)
    class CommandRecorder
    {
    public:
        explicit CommandRecorder(
            GpuContext &gpu);
        // Bila kiriman sebelumnya mungkin masih berjalan, menunggu GPU idle dulu
        // (allocator tidak boleh di-reset selagi GPU memakainya).
        ID3D12GraphicsCommandList *begin();

        void submit();
        void submitAndWait();

    private:
        GpuContext &gpu;
        winrt::com_ptr<ID3D12CommandAllocator> commandAllocator;
        winrt::com_ptr<ID3D12GraphicsCommandList> commandList;

        bool mayBeRunning = false;
    };
}