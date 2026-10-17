#include <cstdint>
#include <memory>
#include <optional>
#include <chrono>
#include <d3d11.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

namespace FaceRuntime
{

    struct CameraConfig
    {
        std::uint32_t cameraIndex = 0;
        std::uint32_t width = 1280;
        std::uint32_t height = 720;
        std::uint32_t fps = 30;
    };

    class Camera
    {
    public:
        explicit Camera(CameraConfig const &config = {});
        ~Camera();

        Camera(Camera const &) = delete;
        Camera &operator=(Camera const &) = delete;

        void start();
        void stop() noexcept;
        [[nodiscard]] winrt::com_ptr<ID3D11Texture2D> waitForFrame(
            std::chrono::milliseconds timeout);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}