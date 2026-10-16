#include <cstdint>
#include <memory>
#include <optional>
#include <chrono>

#include "common/frame.hpp"

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
        [[nodiscard]] std::optional<Frame> waitForFrame(
            std::uint64_t lastFrameId,
            std::chrono::milliseconds timeout);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}