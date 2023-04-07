// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "framebuffercontext.hpp"

#include "../util.hpp"
#include "../viz2d.hpp"

namespace cv {
namespace viz {
namespace detail {

//FIXME use cv::ogl
FrameBufferContext::FrameBufferContext(const cv::Size& frameBufferSize) :
        frameBufferSize_(frameBufferSize) {
#ifndef __EMSCRIPTEN__
    glewExperimental = true;
    glewInit();
    try {
        if(is_cl_gl_sharing_supported())
            cv::ogl::ocl::initializeContextFromGL();
        else
            clglSharing_ = false;
    } catch (...) {
        cerr << "CL-GL sharing failed." << endl;
        clglSharing_ = false;
    }
#else
    clglSharing_ = false;
#endif
    frameBufferID_ = 0;
    GL_CHECK(glGenFramebuffers(1, &frameBufferID_));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
    GL_CHECK(glGenRenderbuffers(1, &renderBufferID_));
    textureID_ = 0;
    GL_CHECK(glGenTextures(1, &textureID_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
    texture_ = new cv::ogl::Texture2D(frameBufferSize_, cv::ogl::Texture2D::RGBA, textureID_);
    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frameBufferSize_.width, frameBufferSize_.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));

    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, frameBufferSize_.width, frameBufferSize_.height));
    GL_CHECK(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));

    GL_CHECK(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
    assert(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
#ifndef __EMSCRIPTEN__
    context_ = CLExecContext_t::getCurrent();
#endif
}

FrameBufferContext::~FrameBufferContext() {
    end();
    glDeleteTextures(1, &textureID_);
    glDeleteRenderbuffers( 1, &renderBufferID_);
    glDeleteFramebuffers( 1, &frameBufferID_);
}

cv::Size FrameBufferContext::getSize() {
    return frameBufferSize_;
}

void FrameBufferContext::execute(std::function<void(cv::UMat&)> fn) {
#ifndef __EMSCRIPTEN__
    CLExecScope_t clExecScope(getCLExecContext());
#endif
    FrameBufferContext::GLScope glScope(*this);
    FrameBufferContext::FrameBufferScope fbScope(*this, frameBuffer_);
    fn(frameBuffer_);
}

cv::ogl::Texture2D& FrameBufferContext::getTexture2D() {
    return *texture_;
}

#ifndef __EMSCRIPTEN__
CLExecContext_t& FrameBufferContext::getCLExecContext() {
    return context_;
}
#endif

void FrameBufferContext::blitFrameBufferToScreen(const cv::Rect& viewport, const cv::Size& windowSize, bool stretch) {
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferID_));
    GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    GL_CHECK(glBlitFramebuffer(
            viewport.x, viewport.y, viewport.x + viewport.width, viewport.y + viewport.height,
            stretch ? 0 : windowSize.width - frameBufferSize_.width,
            stretch ? 0 : windowSize.height - frameBufferSize_.height,
            stretch ? windowSize.width : frameBufferSize_.width,
            stretch ? windowSize.height : frameBufferSize_.height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}

void FrameBufferContext::begin() {
    GL_CHECK(glGetIntegerv( GL_VIEWPORT, viewport_ ));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

void FrameBufferContext::end() {
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, 0));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
}

void FrameBufferContext::download(cv::UMat& m) {
    cv::Mat tmp = m.getMat(cv::ACCESS_WRITE);
    assert(tmp.data != nullptr);
    //this should use a PBO for the pixel transfer, but i couldn't get it to work for both opengl and webgl at the same time
    GL_CHECK(glReadPixels(0, 0, tmp.cols, tmp.rows, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data));
    tmp.release();
}

void FrameBufferContext::upload(const cv::UMat& m) {
    cv::Mat tmp = m.getMat(cv::ACCESS_READ);
    assert(tmp.data != nullptr);
    GL_CHECK(glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        tmp.cols,
        tmp.rows,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        tmp.data)
    );
    tmp.release();
}

void FrameBufferContext::acquireFromGL(cv::UMat &m) {
    if(clglSharing_) {
        GL_CHECK(cv::ogl::convertFromGLTexture2D(getTexture2D(), m));
    } else {
        if(m.empty())
            m.create(getSize(), CV_8UC4);
        download(m);
        GL_CHECK(glFlush());
        GL_CHECK(glFinish());
    }
    //FIXME
    cv::flip(m, m, 0);
}

void FrameBufferContext::releaseToGL(cv::UMat &m) {
    //FIXME
    cv::flip(m, m, 0);
    if(clglSharing_) {
        GL_CHECK(cv::ogl::convertToGLTexture2D(m, getTexture2D()));
    } else {
        if(m.empty())
            m.create(getSize(), CV_8UC4);
        upload(m);
        GL_CHECK(glFlush());
        GL_CHECK(glFinish());
    }
}
}
}
}
