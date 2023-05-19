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

#pragma once
#include "framevizImpl.h"
#include "glUtils.h"

namespace snn
{

class GlNightVisionFrameViz : public NightVisionFrameVizImpl
{
public:
    gl::FullScreenQuad _quad;
    gl::SimpleGlslProgram _program = gl::SimpleGlslProgram("quad");
    GLsizei _width, _height;

    GlNightVisionFrameViz();

    void resize(uint32_t w, uint32_t h) override;

    virtual void render(const NightVisionFrameViz::RenderParameters & rp) override;

    virtual void setWindow(intptr_t /*window*/) override {}

    virtual void reset() override {}
};

class GlNightVisionFrameRec : public GlNightVisionFrameViz {
protected:
    // Context this is recording to.
    std::unique_ptr<gl::RenderContext> _sc;
    intptr_t _window = 0;
    bool _windowChanged = false;

public:
    GlNightVisionFrameRec();

    void render(const NightVisionFrameViz::RenderParameters & rp) override;

    void setWindow(intptr_t window) override;

private:
    void ensureRecordContext(gl::RenderContextStack & rcs);
};

}
