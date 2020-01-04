/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../pch.h"
#include "demoutils.h"

using namespace snn;
using namespace gl;

class NightVisionFrameViz::Impl
{
public:
    FullScreenQuad _quad;
    SimpleGlslProgram _program = gl::SimpleGlslProgram("quad");
    GLsizei _width, _height;

public:
    Impl()
    {
#ifdef __ANDROID__
        // On android, we need to rotate the image 90 degree
        const char* vscode = R"(#version 320 es

            out vec2 v_uv;

            void main()
            {
                const vec4 v[] = vec4[](
                    vec4(-1., -1.,  1., 1.),
                    vec4( 3., -1.,  1.,-1.),
                    vec4(-1.,  3., -1., 1.));
                gl_Position = vec4(v[gl_VertexID].xy, 0., 1.);
                v_uv = v[gl_VertexID].zw;
            }
        )";
#else
        const char * vscode = R"(#version 320 es

            out vec2 v_uv;

            void main()
            {
                const vec4 v[] = vec4[](
                    vec4(-1., -1.,  0., 0.),
                    vec4( 3., -1.,  2., 0.),
                    vec4(-1.,  3.,  0., 2.));
                gl_Position = vec4(v[gl_VertexID].xy, 0., 1.);
                v_uv = v[gl_VertexID].zw;
            }
        )";
#endif

        const char * pscode = R"(#version 320 es
precision mediump float;

layout(binding = 0) uniform sampler2D i_color;
in vec2 v_uv;
out vec3 o_color;

void main()
{
    o_color = texture(i_color, v_uv).xyz;
}
        )";
        _program.loadVsPs(vscode, pscode);
        _quad.allocate();
    }

    virtual ~Impl()
    {
    }

    void resize(uint32_t w, uint32_t h)
    {
        _width = (GLsizei)w;
        _height = (GLsizei)h;
    }

    virtual void render(const RenderParameters & rp)
    {
        if (!_program) {
            return;
        }
        auto g = rp.frame->getGpuData();
        if (g.empty()) {
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0); // make sure we are rendering to screen.
        glViewport(0, 0, _width, _height);
        glDisable(GL_DEPTH_TEST);
        clearScreen(GL_COLOR_BUFFER_BIT);
        _program.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(g.target, g.texture);
        _quad.draw();
    }
};

NightVisionFrameViz::NightVisionFrameViz() : _impl(new Impl()) {}

NightVisionFrameViz::NightVisionFrameViz(NightVisionFrameViz::Impl *impl) : _impl(impl) {}

NightVisionFrameViz::~NightVisionFrameViz() { delete _impl; }

void NightVisionFrameViz::resize(uint32_t w, uint32_t h)
{
    _impl->resize(w, h);
}

void NightVisionFrameViz::render(const RenderParameters & p)
{
    return _impl->render(p);
}

class NightVisionFrameRec::Impl : public NightVisionFrameViz::Impl {
protected:
    // Context this is recording to.
    std::unique_ptr<gl::RenderContext> _sc;
    intptr_t _window = 0;
    bool _windowChanged = false;

public:
    Impl() {
        //Quad needs to be created in the context of the window, which we do not have yet.
        //Cleanup the quad created in the base class until we can get a window.
        _quad.cleanup();
    }

    void setWindow(intptr_t window) {
        if (window != _window) {
            _window = window;
            _windowChanged = true;
        }
    }

    void ensureRecordContext(RenderContextStack & rcs) {
        if (_sc) {
            _sc->makeCurrent();
        }
        if (!_windowChanged) {
            return;
        }
        _windowChanged = false;

        // Free the existing quad.
        _quad.cleanup();

        // delete the old context
        _sc.reset();

        // apply main rendering context
        rcs.apply();

        // Create a new context associated with  tied to the window
        _sc.reset(new gl::RenderContext(gl::RenderContext::SHARED, _window));

        // Startup the new context.
        _sc->makeCurrent();

        // Create the new quad.
        _quad.allocate();
    }

    void render(const RenderParameters &rp) override {
        // store current context
        RenderContextStack rcs;
        rcs.push();

        ensureRecordContext(rcs);

        // Draw everything.
        NightVisionFrameViz::Impl::render(rp);

        // TODO: is this step necessary?
        _sc->swapBuffers();

        // Revert to whatever context was being used before.
        rcs.pop();
    }
};

void NightVisionFrameRec::render(const NightVisionFrameViz::RenderParameters & rp) {
    return _impl->render(rp);
}

NightVisionFrameRec::NightVisionFrameRec() : NightVisionFrameViz(new NightVisionFrameRec::Impl()) {}

void NightVisionFrameRec::setWindow(intptr_t window) {
    static_cast<Impl *>(_impl)->setWindow(window);
}
