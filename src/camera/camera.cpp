#include "camera.hpp"
#include <mutex>

#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Media.Capture.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Media.Capture.Frames.h>
#include <winrt/Windows.Media.MediaProperties.h>

namespace
{
    class ApartmentScope
    {
    public:
        ApartmentScope()
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
        };
        ~ApartmentScope()
        {
            winrt::uninit_apartment();
        };
        ApartmentScope(ApartmentScope const &) = delete;
        ApartmentScope &operator=(ApartmentScope const &) = delete;
    };

    winrt::com_ptr<ID3D11Texture2D> textureFromSurface(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface const &surface)
    {
        auto access = surface.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        winrt::com_ptr<ID3D11Texture2D> texture;

        winrt::check_hresult(access->GetInterface(IID_PPV_ARGS(texture.put())));
        return texture;
    }
}

namespace FaceRuntime
{
    struct Camera::Impl
    {
        explicit Impl(CameraConfig config) : cameraConfig(config) {}

        CameraConfig cameraConfig;
        std::thread captureThread;
        std::atomic_bool stopRequested{false};
        std::atomic_bool running{false};

        std::mutex mailboxMutex;
        std::condition_variable mailboxcv;
        Frame frame;
        bool hasFrame = false;

        std::mutex livecycleMutex;
        std::condition_variable livecyclecv;
        bool startFinished = false;
        std::exception_ptr startError;

        winrt::Windows::Media::Capture::MediaCapture mediaCapture{nullptr};
        winrt::Windows::Media::Capture::Frames::MediaFrameSource mediaFrameSource{nullptr};
        winrt::Windows::Media::Capture::Frames::MediaFrameReader mediaFrameReader{nullptr};
        winrt::event_token frameToken{};

        void captureLoop() noexcept;
        void initializeCapture();
        void registerFrameHandler();
        void onFrameArrived(winrt::Windows::Media::Capture::Frames::MediaFrameReader const &sender) noexcept;
        void cleanupCapture() noexcept;
    };

    void Camera::Impl::captureLoop() noexcept
    {
        try
        {
            ApartmentScope apartment;

            {
                std::scoped_lock lock(livecycleMutex);
                startFinished = false;
            }

            initializeCapture();
            registerFrameHandler();

            const auto status = mediaFrameReader.StartAsync().get();
            if (status != winrt::Windows::Media::Capture::Frames::MediaFrameReaderStartStatus::Success)
            {
                throw std::runtime_error("MediaFrameReader gagal distart");
            }

            running.store(true, std::memory_order_release);

            {
                std::scoped_lock lock(livecycleMutex);
                startFinished = true;
            }
            livecyclecv.notify_all();

            std::unique_lock lock(mailboxMutex);
            mailboxcv.wait(lock, [this]
                           { return stopRequested.load(std::memory_order_acquire); });
        }
        catch (...)
        {
            {
                std::scoped_lock lock(livecycleMutex);
                startError = std::current_exception();
                startFinished = true;
            }
            livecyclecv.notify_all();
        }

        running.store(false, std::memory_order_release);
        cleanupCapture();
    }

    void Camera::Impl::initializeCapture()
    {
        using namespace winrt::Windows::Media::Capture;
        using namespace winrt::Windows::Media::Capture::Frames;

        const auto groups = MediaFrameSourceGroup::FindAllAsync().get();
        std::vector<MediaFrameSourceGroup> colorGroups;

        for (auto const &group : groups)
        {
            for (auto const &info : group.SourceInfos())
            {
                const bool is_video =
                    info.MediaStreamType() == MediaStreamType::VideoPreview ||
                    info.MediaStreamType() == MediaStreamType::VideoRecord;

                if (info.SourceKind() == MediaFrameSourceKind::Color &&
                    is_video)
                {
                    colorGroups.push_back(group);
                    break;
                }
            }
        }

        if (cameraConfig.cameraIndex >= colorGroups.size())
        {
            throw std::runtime_error(
                "cameraIndex yang diminta tidak menunjuk color camera yang tersedia");
        }

        const auto selectedGroup = colorGroups[cameraConfig.cameraIndex];
        winrt::hstring sourceId;

        for (auto const preferred : {
                 MediaStreamType::VideoPreview,
                 MediaStreamType::VideoRecord})
        {
            for (auto const &info : selectedGroup.SourceInfos())
            {
                if (info.SourceKind() == MediaFrameSourceKind::Color &&
                    info.MediaStreamType() == preferred)
                {
                    sourceId = info.Id();
                    break;
                }
            }
            if (!sourceId.empty())
            {
                break;
            }
        }

        if (sourceId.empty())
        {
            throw std::runtime_error(
                "color camera tidak memiliki preview atau record stream");
        }

        MediaCaptureInitializationSettings settings;
        settings.SourceGroup(selectedGroup);
        settings.StreamingCaptureMode(StreamingCaptureMode::Video);
        settings.SharingMode(MediaCaptureSharingMode::ExclusiveControl);
        settings.MemoryPreference(MediaCaptureMemoryPreference::Auto);

        mediaCapture = MediaCapture();
        mediaCapture.InitializeAsync(settings).get();

        mediaFrameSource = mediaCapture.FrameSources().Lookup(sourceId);
        std::optional<winrt::Windows::Media::Capture::Frames::MediaFrameFormat> selectedFormat;
        double smallestFPSDelta = std::numeric_limits<double>::max();
        for (auto const &format : mediaFrameSource.SupportedFormats())
        {
            const auto videoFormat = format.VideoFormat();
            const auto rate = format.FrameRate();
            if (!videoFormat || rate.Denominator() == 0 || rate.Numerator() == 0)
            {
                continue;
            }

            if (videoFormat.Width() != cameraConfig.width ||
                videoFormat.Height() != cameraConfig.height)
            {
                continue;
            }

            const double aktualFPS =
                static_cast<double>(rate.Numerator()) /
                static_cast<double>(rate.Denominator());
            const double fpsError =
                std::abs(aktualFPS - static_cast<double>(cameraConfig.fps));

            if (!selectedFormat || fpsError < smallestFPSDelta)
            {
                selectedFormat = format;
                smallestFPSDelta = fpsError;
            }
        }

        if (!selectedFormat)
        {
            throw std::runtime_error(
                "camera tidak memiliki format dengan ukuran yang diminta");
        }

        mediaFrameSource.SetFormatAsync(*selectedFormat).get();

        mediaFrameReader = mediaCapture.CreateFrameReaderAsync(mediaFrameSource, L"NV12").get();
        mediaFrameReader.AcquisitionMode(winrt::Windows::Media::Capture::Frames::MediaFrameReaderAcquisitionMode::Realtime);
    }

    void Camera::Impl::registerFrameHandler()
    {
        frameToken = mediaFrameReader.FrameArrived([this](winrt::Windows::Media::Capture::Frames::MediaFrameReader const &sender, winrt::Windows::Media::Capture::Frames::MediaFrameArrivedEventArgs const &)
                                                   { onFrameArrived(sender); });
    }

    void Camera::Impl::onFrameArrived(winrt::Windows::Media::Capture::Frames::MediaFrameReader const &sender) noexcept
    {
        try
        {
            const auto frameReference = sender.TryAcquireLatestFrame();
            if (!frameReference)
            {
                return;
            }

            const auto videoFrame = frameReference.VideoMediaFrame();
            if (!videoFrame)
            {
                return;
            }

            const auto surface = videoFrame.Direct3DSurface();
            if (!surface)
            {
                throw std::runtime_error(
                    "frame tidak memiliki Direct3DSurface");
            }

            auto texture = textureFromSurface(surface);

            Frame newFrame;
            newFrame.texture = std::move(texture);
            {
                std::scoped_lock lock(mailboxMutex);
                newFrame.frameId = frame.frameId + 1;
                frame = std::move(newFrame);
                hasFrame = true;
            }
            mailboxcv.notify_all();
        }
        catch (...)
        {
            stopRequested.store(true, std::memory_order_release);
            mailboxcv.notify_all();
        }
    }

    void Camera::Impl::cleanupCapture() noexcept
    {
        try
        {
            if (mediaFrameReader)
            {
                mediaFrameReader.FrameArrived(frameToken);
                mediaFrameReader.StopAsync().get();
                mediaFrameReader.Close();
                mediaFrameReader = nullptr;
            }
            if (mediaCapture)
            {
                mediaCapture.Close();
                mediaCapture = nullptr;
            }
        }
        catch (...)
        {
        }
    }

    Camera::Camera(CameraConfig const &config) : impl_(std::make_unique<Impl>(config))
    {
    }

    Camera::~Camera()
    {
        stop();
    }

    void Camera::stop() noexcept
    {
        if (!impl_ || !impl_->captureThread.joinable())
        {
            return;
        }
        impl_->stopRequested.store(true, std::memory_order_release);
        impl_->mailboxcv.notify_all();
        if (impl_->captureThread.joinable())
        {
            impl_->captureThread.join();
        }
    }

    void Camera::start()
    {
        impl_->stopRequested.store(false, std::memory_order_release);
        impl_->captureThread = std::thread([this]
                                           { impl_->captureLoop(); });

        std::unique_lock lock(impl_->livecycleMutex);
        impl_->livecyclecv.wait(lock, [this]
                                { return impl_->startFinished; });

        if (impl_->startError)
        {
            auto error = impl_->startError;
            lock.unlock();
            impl_->captureThread.join();
            std::rethrow_exception(error);
        }
    }

    std::optional<Frame> Camera::waitForFrame(
        std::uint64_t last_frame_id,
        std::chrono::milliseconds timeout)
    {
        std::unique_lock lock(impl_->mailboxMutex);
        const bool changed = impl_->mailboxcv.wait_for(lock, timeout, [this, last_frame_id]
                                                       { return impl_->hasFrame &&
                                                                impl_->frame.frameId > last_frame_id; });

        if (!changed)
        {
            return std::nullopt;
        }
        return impl_->frame;
    }
}