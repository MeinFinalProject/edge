#include "camera/camera.hpp"

#include <chrono>
#include <iostream>

int main()
{
    FaceRuntime::Camera camera;
    camera.start();

    for (int i = 0; i < 30; ++i)
    {
        auto texture = camera.waitForFrame(
            std::chrono::seconds(2));
        if (!texture)
        {
            std::cerr << "timeout menunggu frame\n";
            return 1;
        }

        std::cout << " diterima, texture: " << texture.get() << "\n";

        D3D11_TEXTURE2D_DESC desc{};

        texture->GetDesc(&desc);

        std::cout << "height dan width texture: "
                  << desc.Height << "x" << desc.Width << "\n";
    }

    camera.stop();
}
