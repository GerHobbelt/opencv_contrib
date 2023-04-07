#include "util.hpp"

#include "viz2d.hpp"
#include "nvg.hpp"

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <fstream>
#else
# include <opencv2/core/ocl.hpp>
# include <va/va.h>
# include <csignal>
#endif

namespace kb {
namespace viz2d {

std::string get_gl_info() {
    return reinterpret_cast<const char*>(glGetString(GL_VERSION));
}

std::string get_cl_info() {
    std::stringstream ss;
#ifndef __EMSCRIPTEN__
    std::vector<cv::ocl::PlatformInfo> plt_info;
    cv::ocl::getPlatfomsInfo(plt_info);
    const cv::ocl::Device &defaultDevice = cv::ocl::Device::getDefault();
    cv::ocl::Device current;
    ss << endl;
    for (const auto &info : plt_info) {
        for (int i = 0; i < info.deviceNumber(); ++i) {
            ss << "\t";
            info.getDevice(current, i);
            if (defaultDevice.name() == current.name())
                ss << "* ";
            else
                ss << "  ";
            ss << info.version() << " = " << info.name() << endl;
            ss << "\t\t  GL sharing: " << (current.isExtensionSupported("cl_khr_gl_sharing") ? "true" : "false") << endl;
            ss << "\t\t  VAAPI media sharing: " << (current.isExtensionSupported("cl_intel_va_api_media_sharing") ? "true" : "false") << endl;
        }
    }
#endif
    return ss.str();
}

bool is_intel_va_supported() {
#ifndef __EMSCRIPTEN__
    try {
        std::vector<cv::ocl::PlatformInfo> plt_info;
        cv::ocl::getPlatfomsInfo(plt_info);
        cv::ocl::Device current;
        for (const auto &info : plt_info) {
            for (int i = 0; i < info.deviceNumber(); ++i) {
                info.getDevice(current, i);
                return current.isExtensionSupported("cl_intel_va_api_media_sharing");
            }
        }
    } catch (std::exception& ex) {
        cerr << "Intel VAAPI query failed: " << ex. what() << endl;
    } catch (...) {
        cerr << "Intel VAAPI query failed" << endl;
    }
#endif
    return false;
}

bool is_cl_gl_sharing_supported() {
#ifndef __EMSCRIPTEN__
    try {
        std::vector<cv::ocl::PlatformInfo> plt_info;
        cv::ocl::getPlatfomsInfo(plt_info);
        cv::ocl::Device current;
        for (const auto &info : plt_info) {
            for (int i = 0; i < info.deviceNumber(); ++i) {
                info.getDevice(current, i);
                return current.isExtensionSupported("cl_khr_gl_sharing");
            }
        }
    } catch (std::exception& ex) {
        cerr << "CL-GL sharing query failed: " << ex. what() << endl;
    } catch (...) {
        cerr << "CL-GL sharing query failed" << endl;
    }
#endif
    return false;
}

void print_system_info() {
    cerr << "OpenGL Version: " << get_gl_info() << endl;
    cerr << "OpenCL Platforms: " << get_cl_info() << endl;
}

static bool finish_requested = false;
static bool signal_handlers_installed = false;

static void request_finish(int ignore) {
    finish_requested = true;
}

static void install_signal_handlers() {
    signal(SIGINT, request_finish);
    signal(SIGTERM, request_finish);
}

bool keep_running() {
    if(!signal_handlers_installed) {
        install_signal_handlers();
    }
    return !finish_requested;
}

void update_fps(cv::Ptr<kb::viz2d::Viz2D> v2d, bool graphically) {
    static uint64_t cnt = 0;
    static cv::TickMeter tick;
    static float fps;

    if (cnt > 0) {
        tick.stop();

        if (tick.getTimeMilli() > 50) {
            cerr << "FPS : " << (fps = tick.getFPS());
#ifndef __EMSCRIPTEN__
            cerr << '\r';
#else
            cerr << endl;
#endif
            cnt = 0;
            tick.reset();
        }

        if (graphically) {
            v2d->nvg([&](const cv::Size &size) {
                using namespace kb;
                string text = "FPS: " + std::to_string(fps);
                nvg::beginPath();
                nvg::roundedRect(5, 5, 15 * text.size() + 5, 30, 5);
                nvg::fillColor(cv::Scalar(255, 255, 255, 180));
                nvg::fill();

                nvg::fontSize(30.0f);
                nvg::fontFace("mono");
                nvg::fillColor(cv::Scalar(90, 90, 90, 255));
                nvg::textAlign(NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
                nvg::text(10, 20, text.c_str(), nullptr);
            });
        }
    }

    tick.start();
    ++cnt;
}

#ifndef __EMSCRIPTEN__
Sink make_va_sink(cv::Ptr<Viz2D> v2d, const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, frameSize, { cv::VIDEOWRITER_PROP_HW_DEVICE, vaDeviceIndex, cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    v2d->setVideoFrameSize(frameSize);

    return Sink([=](const cv::UMat& frame){
        (*writer) << frame;
        return writer->isOpened();
    });
}

Source make_va_source(cv::Ptr<Viz2D> v2d, const string &inputFilename, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG, { cv::CAP_PROP_HW_DEVICE, vaDeviceIndex, cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    float fps = capture->get(cv::CAP_PROP_FPS);
    float w = capture->get(cv::CAP_PROP_FRAME_WIDTH);
    float h = capture->get(cv::CAP_PROP_FRAME_HEIGHT);
    v2d->setVideoFrameSize(cv::Size(w,h));

    return Source([=](cv::UMat& frame){
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}
#else
Sink make_va_sink(cv::Ptr<Viz2D> v2d, const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize, const int vaDeviceIndex) {
    return Sink([=](const cv::UMat& frame){
        return false;
    });
}

Source make_va_source(cv::Ptr<Viz2D> v2d, const string &inputFilename, const int vaDeviceIndex) {
    return Source([=](cv::UMat& frame){
        return false;
    }, fps);
#endif

#ifndef __EMSCRIPTEN__
Sink make_writer_sink(cv::Ptr<Viz2D> v2d, const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize) {
    if(is_intel_va_supported()) {
        return make_va_sink(v2d, outputFilename, fourcc, fps, frameSize, 0);
    }

    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, frameSize);
    v2d->setVideoFrameSize(frameSize);

    return Sink([=](const cv::UMat& frame){
        (*writer) << frame;
        return writer->isOpened();
    });
}

Source make_capture_source(cv::Ptr<Viz2D> v2d, const string &inputFilename) {
    if(is_intel_va_supported()) {
        return make_va_source(v2d, inputFilename, 0);
    }

    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG);
    float fps = capture->get(cv::CAP_PROP_FPS);
    float w = capture->get(cv::CAP_PROP_FRAME_WIDTH);
    float h = capture->get(cv::CAP_PROP_FRAME_HEIGHT);
    v2d->setVideoFrameSize(cv::Size(w,h));

    return Source([=](cv::UMat& frame){
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

#else
Source make_capture_source(cv::Ptr<Viz2D> v2d, int width, int height) {
    using namespace std;
    static cv::Mat tmp(height, width, CV_8UC4);

    return Source([=](cv::UMat& frame) {
        try {
            if (frame.empty())
                frame.create(v2d->getInitialSize(), CV_8UC3);
            std::ifstream fs("viz2d_rgba_canvas.raw", std::fstream::in | std::fstream::binary);
            fs.seekg(0, std::ios::end);
            auto length = fs.tellg();
            fs.seekg(0, std::ios::beg);

            if (length == (frame.elemSize() + 1) * frame.total()) {
                cv::Mat v = frame.getMat(cv::ACCESS_WRITE);
                fs.read((char*) (tmp.data), tmp.elemSize() * tmp.total());
                cvtColor(tmp, v, cv::COLOR_BGRA2RGB);
                v.release();
            } else if(length == 0) {
                std::cerr << "Error: empty webcam frame received!" << endl;
            } else {
                std::cerr << "Error: webcam frame size mismatch!" << endl;
            }
        } catch(std::exception& ex) {
            cerr << ex.what() << endl;
        }
        return true;
    }, 0);
}
#endif

}
}

