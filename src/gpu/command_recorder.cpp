#include "command_recorder.hpp"

namespace FaceRuntime::Gpu
{
    CommandRecorder::CommandRecorder(
        GpuContext &gpu)
        : gpu(gpu)
    {
        auto *device = gpu.device();
        // Allocator = memori tempat isi buku perintah disimpan.
        winrt::check_hresult(
            device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(commandAllocator.put())));

        // Command list = buku perintah yang akan diisi.
        winrt::check_hresult(
            gpu.device()->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                commandAllocator.get(),
                nullptr,
                IID_PPV_ARGS(commandList.put())));
        winrt::check_hresult(commandList->Close());
    }

    ID3D12GraphicsCommandList *CommandRecorder::begin()
    {
        if (mayBeRunning)
        {
            gpu.waitIdle();
            mayBeRunning = false;
        }
        winrt::check_hresult(commandAllocator->Reset());
        winrt::check_hresult(commandList->Reset(commandAllocator.get(), nullptr));
        return commandList.get();
    }

    void CommandRecorder::submit()
    {
        winrt::check_hresult(commandList->Close());
        ID3D12CommandList *lists[] = {commandList.get()};
        gpu.queue()->ExecuteCommandLists(1, lists);
        mayBeRunning = true;
    }

    void CommandRecorder::submitAndWait()
    {
        submit();
        gpu.waitIdle();
        mayBeRunning = false;
    }
}