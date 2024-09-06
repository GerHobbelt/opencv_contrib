// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_MUSA

#include <musa_runtime.h>

#include "opencv2/core/musa.hpp"
#include "opencv2/core/musa_stream_accessor.hpp"
#include "opencv2/ts/musa_test.hpp"

namespace opencv_test { namespace {

struct AsyncEvent : testing::TestWithParam<cv::musa::DeviceInfo>
{
    cv::musa::HostMem src;
    cv::musa::GpuMat d_src;

    cv::musa::HostMem dst;
    cv::musa::GpuMat d_dst;

    cv::musa::Stream stream;

    virtual void SetUp()
    {
        cv::musa::DeviceInfo devInfo = GetParam();
        cv::musa::setDevice(devInfo.deviceID());

        src = cv::musa::HostMem(cv::musa::HostMem::PAGE_LOCKED);

        cv::Mat m = randomMatMusa(cv::Size(128, 128), CV_8UC1);
        m.copyTo(src);
    }
};

void deviceWork(void* userData)
{
    AsyncEvent* test = reinterpret_cast<AsyncEvent*>(userData);
    test->d_src.upload(test->src, test->stream);
    test->d_src.convertTo(test->d_dst, CV_32S, test->stream);
    test->d_dst.download(test->dst, test->stream);
}

MUSA_TEST_P(AsyncEvent, WrapEvent)
{
    musaEvent_t musa_event = NULL;
    ASSERT_EQ(musaSuccess, musaEventCreate(&musa_event));
    {
        cv::musa::Event musaEvent = cv::musa::EventAccessor::wrapEvent(musa_event);
        deviceWork(this);
        musaEvent.record(stream);
        musaEvent.waitForCompletion();
        cv::Mat dst_gold;
        src.createMatHeader().convertTo(dst_gold, CV_32S);
        ASSERT_MAT_NEAR(dst_gold, dst, 0);
    }
    ASSERT_EQ(musaSuccess, musaEventDestroy(musa_event));
}

MUSA_TEST_P(AsyncEvent, WithFlags)
{
    cv::musa::Event musaEvent = cv::musa::Event(cv::musa::Event::CreateFlags::BLOCKING_SYNC);
    deviceWork(this);
    musaEvent.record(stream);
    musaEvent.waitForCompletion();
    cv::Mat dst_gold;
    src.createMatHeader().convertTo(dst_gold, CV_32S);
    ASSERT_MAT_NEAR(dst_gold, dst, 0);
}

MUSA_TEST_P(AsyncEvent, Timing)
{
    const std::vector<cv::musa::Event::CreateFlags> eventFlags = { cv::musa::Event::CreateFlags::BLOCKING_SYNC , cv::musa::Event::CreateFlags::BLOCKING_SYNC | cv::musa::Event::CreateFlags::DISABLE_TIMING };
    const std::vector<bool> shouldFail = { false, true };
    for (size_t i = 0; i < eventFlags.size(); i++) {
        const auto& flags = eventFlags.at(i);
        cv::musa::Event startEvent = cv::musa::Event(flags);
        cv::musa::Event stopEvent = cv::musa::Event(flags);
        startEvent.record(stream);
        deviceWork(this);
        stopEvent.record(stream);
        stopEvent.waitForCompletion();
        cv::Mat dst_gold;
        src.createMatHeader().convertTo(dst_gold, CV_32S);
        ASSERT_MAT_NEAR(dst_gold, dst, 0);
        bool failed = false;
        try {
            const double elTimeMs = cv::musa::Event::elapsedTime(startEvent, stopEvent);
            ASSERT_GT(elTimeMs, 0);
        }
        catch (cv::Exception ex) {
            failed = true;
        }
        ASSERT_EQ(failed, shouldFail.at(i));
    }
}

INSTANTIATE_TEST_CASE_P(MUSA_Event, AsyncEvent, ALL_DEVICES);

}} // namespace
#endif // HAVE_MUSA
