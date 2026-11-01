#pragma once

#include <mutex>
#include <condition_variable>
#include <chrono>
#include <utility>
#include <winrt/Windows.Media.Capture.Frames.h>

namespace FaceRuntime
{
    class FrameMailbox
    {
    public:
        void publish(winrt::Windows::Media::Capture::Frames::MediaFrameReference texture)
        {
            {
                std::scoped_lock lock(mutex);
                latestTexture = std::move(texture);
            }
            readyCv.notify_one();
        }

        void reopen()
        {
            std::scoped_lock lock(mutex);
            closed = false;
            latestTexture = nullptr;
        }

        void close() noexcept
        {
            {
                std::scoped_lock lock(mutex);
                closed = true;
            }
            readyCv.notify_all();
        }

        [[nodiscard]] winrt::Windows::Media::Capture::Frames::MediaFrameReference
        waitForFrame(
            std::chrono::milliseconds timeout)
        {
            std::unique_lock lock(mutex);
            if (!readyCv.wait_for(lock, timeout, [this]
                                  { return static_cast<bool>(latestTexture) || closed; }))
            {
                return nullptr;
            }

            if (closed || !latestTexture)
            {
                return nullptr;
            }

            return std::exchange(latestTexture, nullptr);
        }

    private:
        winrt::Windows::Media::Capture::Frames::MediaFrameReference latestTexture{nullptr};
        std::mutex mutex;
        std::condition_variable readyCv;
        bool closed = false;
    };

}