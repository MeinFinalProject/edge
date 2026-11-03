#include "camera.hpp"
#include "common/frame_mailbox.hpp"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Media.Capture.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Media.Capture.Frames.h>
#include <winrt/Windows.Media.MediaProperties.h>

#include <cwchar>
#include <string>

#include <dshow.h>
#include <ks.h>
#include <ksmedia.h>
#include <ocidl.h>
#include <oleauto.h>

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

    // 0 = frame rate dijaga konstan, 1 = driver boleh menurunkan fps saat gelap
    constexpr long kLowLightProperty =
        static_cast<long>(KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY);
    constexpr long kFixedFrameRateValue = 0;

    // Mencari filter DirectShow dengan nama yang diberikan. Mengembalikan nullptr bila tidak ditemukan.
    winrt::com_ptr<IBaseFilter> findDirectShowFilter(winrt::hstring const &friendlyName)
    {
        winrt::com_ptr<ICreateDevEnum> deviceEnum;
        winrt::check_hresult(CoCreateInstance(
            CLSID_SystemDeviceEnum,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(deviceEnum.put())));

        winrt::com_ptr<IEnumMoniker> enumMoniker;
        const HRESULT hr = deviceEnum->CreateClassEnumerator(
            CLSID_VideoInputDeviceCategory,
            enumMoniker.put(),
            0);
        if (hr == S_FALSE)
        {
            return nullptr;
        }
        winrt::check_hresult(hr);

        winrt::com_ptr<IMoniker> moniker;
        while (enumMoniker->Next(1, moniker.put(), nullptr) == S_OK)
        {
            winrt::com_ptr<IPropertyBag> propertyBag;
            if (FAILED(moniker->BindToStorage(nullptr, nullptr, IID_PPV_ARGS(propertyBag.put()))))
            {
                continue;
            }

            VARIANT name;
            VariantInit(&name);
            const bool matched =
                SUCCEEDED(propertyBag->Read(L"FriendlyName", &name, nullptr)) &&
                name.vt == VT_BSTR &&
                name.bstrVal != nullptr &&
                _wcsicmp(name.bstrVal, friendlyName.c_str()) == 0;
            VariantClear(&name);

            if (matched)
            {
                winrt::com_ptr<IBaseFilter> filter;
                winrt::check_hresult(moniker->BindToObject(
                    nullptr, nullptr, IID_PPV_ARGS(filter.put())));
                return filter;
            }
        }
        return nullptr;
    }

    // Mengatur low-light priority ke 0 (frame rate dijaga konstan) untuk kamera DirectShow yang dipilih.
    void setFixedFrameRate(winrt::hstring const &cameraName)
    {
        const auto filter = findDirectShowFilter(cameraName);
        if (!filter)
        {
            throw std::runtime_error(
                "kamera \"" + winrt::to_string(cameraName) +
                "\" tidak ditemukan di DirectShow");
        }

        const auto control = filter.as<IAMCameraControl>();
        winrt::check_hresult(control->Set(
            kLowLightProperty,
            kFixedFrameRateValue,
            CameraControl_Flags_Manual));

        long observedValue = -1;
        long observedFlags = 0;
        winrt::check_hresult(control->Get(kLowLightProperty, &observedValue, &observedFlags));
        if (observedValue != kFixedFrameRateValue)
        {
            throw std::runtime_error(
                "driver tidak mempertahankan low-light priority yang diminta");
        }
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

        FrameMailbox mailbox;

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

            stopRequested.wait(false, std::memory_order_acquire);
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
        setFixedFrameRate(selectedGroup.DisplayName());

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

            mailbox.publish(std::move(frameReference));
        }
        catch (...)
        {
            stopRequested.store(true, std::memory_order_release);
            stopRequested.notify_all();
            mailbox.close();
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
        impl_->stopRequested.notify_all();
        impl_->mailbox.close();
        if (impl_->captureThread.joinable())
        {
            impl_->captureThread.join();
        }
    }

    void Camera::start()
    {
        stop();
        {
            std::scoped_lock lock(impl_->livecycleMutex);
            impl_->startFinished = false;
            impl_->startError = nullptr;
        }
        impl_->mailbox.reopen();
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

    [[nodiscard]] winrt::Windows::Media::Capture::Frames::MediaFrameReference Camera::waitForFrame(
        std::chrono::milliseconds timeout)
    {
        return impl_->mailbox.waitForFrame(timeout);
    }

    winrt::com_ptr<ID3D11Texture2D> TextureOf(
        winrt::Windows::Media::Capture::Frames::MediaFrameReference const &frame)
    {
        const auto video = frame.VideoMediaFrame();
        if (!video)
        {
            throw std::runtime_error("frame kamera bukan frame video");
        }
        const auto surface = video.Direct3DSurface();
        if (!surface)
        {
            throw std::runtime_error(
                "frame tidak memiliki Direct3DSurface (kamera memakai jalur memori CPU?)");
        }
        auto access = surface.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        winrt::com_ptr<ID3D11Texture2D> texture;
        winrt::check_hresult(access->GetInterface(IID_PPV_ARGS(texture.put())));
        return texture;
    }
}