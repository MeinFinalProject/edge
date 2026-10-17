#include "camera/camera.hpp"

#include <chrono>
#include <iostream>

int main()
{
    FaceRuntime::Camera camera;
    camera.start();

    std::uint64_t last_frame_id = 0;
    for (int i = 0; i < 30; ++i)
    {
        auto frame = camera.waitForFrame(
            last_frame_id,
            std::chrono::seconds(2));
        if (!frame)
        {
            std::cerr << "timeout menunggu frame\n";
            return 1;
        }

        std::cout << "frame " << frame->frameId
                  << " diterima, texture: " << frame->texture.get() << "\n";

        D3D11_TEXTURE2D_DESC desc{};

        frame->texture->GetDesc(&desc);

        std::cout << "height dan width texture: "
                  << desc.Height << "x" << desc.Width << "\n";

        last_frame_id = frame->frameId;
    }

    camera.stop();
}
