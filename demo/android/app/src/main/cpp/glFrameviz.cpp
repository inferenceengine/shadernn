/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
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
#include "glFrameviz.h"
#include <glad/glad.h>
#include "snn/glImageHandle.h"

namespace snn {

GlNightVisionFrameViz::GlNightVisionFrameViz()
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
            v_uv = v[gl_VertexID].zw;NightVisionFrameRec
-            }
-        )";
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


void GlNightVisionFrameViz::resize(uint32_t w, uint32_t h)
{
    _width = (GLsizei)w;
    _height = (GLsizei)h;
}

void GlNightVisionFrameViz::render(const NightVisionFrameViz::RenderParameters & rp)
{
    if (!_program) {
        return;
    }
    GlImageHandle g;
    rp.frame->getGpuImageHandle(g);
    if (g.empty()) {
        return;
    }
    SNN_LOGD("%s:%d, texture id = %d", __FILENAME__, __LINE__, g.textureId);
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // make sure we are rendering to screen.
    glViewport(0, 0, _width, _height);
    glDisable(GL_DEPTH_TEST);
    gl::clearScreen(GL_COLOR_BUFFER_BIT);
    _program.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(g.target, g.textureId);
    _quad.draw();
}

GlNightVisionFrameRec::GlNightVisionFrameRec() {
    //Quad needs to be created in the context of the window, which we do not have yet.
    //Cleanup the quad created in the base class until we can get a window.
    _quad.cleanup();
}

void GlNightVisionFrameRec::setWindow(intptr_t window) {
    if (window != _window) {
        _window = window;
        _windowChanged = true;
    }
}

void GlNightVisionFrameRec::ensureRecordContext(gl::RenderContextStack & rcs) {
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

void GlNightVisionFrameRec::render(const NightVisionFrameViz::RenderParameters &rp) {
    // store current context
    gl::RenderContextStack rcs;
    rcs.push();

    ensureRecordContext(rcs);

    // Draw everything.
    GlNightVisionFrameViz::render(rp);

    // TODO: is this step necessary?
    _sc->swapBuffers();

    // Revert to whatever context was being used before.
    rcs.pop();
}

}
