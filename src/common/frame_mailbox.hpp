#pragma once

#include <mutex>
#include <condition_variable>
#include <optional>
#include <utility>

namespace FaceRuntime
{
    class FrameMailbox
    {
    public:
        void publish(winrt::com_ptr<ID3D11Texture2D> texture)
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

        [[nodiscard]] winrt::com_ptr<ID3D11Texture2D> waitForFrame(
            std::chrono::milliseconds timeout)
        {
            std::unique_lock lock(mutex);
            if (!readyCv.wait_for(lock, timeout, [this]
                                  { return latestTexture != nullptr || closed; }))
            {
                return nullptr;
            }

            if (closed)
            {
                return nullptr;
            }

            return std::move(latestTexture);
        }

    private:
        winrt::com_ptr<ID3D11Texture2D> latestTexture;
        std::mutex mutex;
        std::condition_variable readyCv;
        bool closed = false;
    };

}