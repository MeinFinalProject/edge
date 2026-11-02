#include <iostream>
#include <winrt/base.h>
#include "gpu/compute_kernel.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/tensor_buffer.hpp"
#include "preprocess/kernel_constants.hpp"
#include "scrfd_tensorize.h"
#include "pad_tensorize.h"
#include "arcface_tensorize.h"

int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        using namespace FaceRuntime;
        Gpu::GpuContext gpu;

        // Jika root signature tidak cocok dengan shader, PSO gagal dibuat DI SINI.
        Gpu::ComputeKernel scrfd(gpu.device(), g_scrfd_tensorize, sizeof(g_scrfd_tensorize),
                                 sizeof(Preprocess::ScrfdConstants) / 4, 1);
        Gpu::ComputeKernel pad(gpu.device(), g_pad_tensorize, sizeof(g_pad_tensorize),
                               sizeof(Preprocess::WarpConstants) / 4, 2);
        Gpu::ComputeKernel arc(gpu.device(), g_arcface_tensorize, sizeof(g_arcface_tensorize),
                               sizeof(Preprocess::WarpConstants) / 4, 1);
        std::cout << "3 PSO berhasil dibuat\n";

        Gpu::TensorBuffer t(gpu, {1, 3, 640, 640});
        auto data = t.readbackToCpu(gpu);
        std::cout << "tensor " << data.size() << " float (diharapkan 1228800)\n";
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "HRESULT 0x" << std::hex << static_cast<unsigned long>(e.code().value)
                  << ": " << winrt::to_string(e.message()) << "\n";
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << "\n";
    }
}
