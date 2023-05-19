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
#include "snn/utils.h"
#include "snn/image.h"
#include "colorGL.h"
#include "snn/deviceTimer.h"
#include <memory>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <variant>
#include <algorithm>
#include <sstream>
#include <KHR/khrplatform.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>

#include <iostream>
#include <inttypes.h>

// This class is a c++ wrapper around OpenGL API

inline void checkGlError(const char* funcName) {
    if (glGetError == NULL) {
        SNN_LOGE("gl not initialized properly...");
    } else {
        GLenum err = glGetError();
        if (GL_NO_ERROR != err) {
            SNN_LOGE("function %s failed. (error=0x%x)", funcName, err);
        }
    }
}

// Check OpenGL error. This check is enabled in both debug and release build.
#define GLCHK(func)                                                                                                                                        \
    func;                                                                                                                                                  \
    checkGlError(#func);                                                                                                                                   \

// Use GLCHKDBG() at where that you want to have some sanity check in debug build,
// but don't want to pay the overhead in release build.
#ifdef _DEBUG
    #define GLCHKDBG(x) GLCHK(x)
#else
    #define GLCHKDBG(x) x
#endif

// TODO: these are obsolete GL related macros that needs to be removed from code base.
#define CHECK_GL_ERROR(target)    GLCHK(;)
#define CHECK_AND_WRAP(func, ...) GLCHK(func(__VA_ARGS__))
#define GL_WRAPPER(func, ...)     func(__VA_ARGS__)

namespace gl {
void initGLExtensions(bool printGLInfo = false);

// The 'optionalFilename' parameter is optional and is only used when printing
// shader compilation error.
GLuint loadShaderFromString(const char* source, size_t length, GLenum shaderType, const char* optionalFilename = nullptr);

// the program name parameter is optional and is only used to print link error.
GLuint linkProgram(const std::vector<GLuint>& shaders, const char* optionalProgramName = nullptr);

// a utility function to upload uniform values
template<typename T>
void updateUniformValue(GLint location, const T& value) {
    if (location < 0) {
        return;
    }

    if constexpr (std::is_same<int, T>()) {
        GLCHKDBG(glUniform1i(location, (int) value));
    } else if constexpr (std::is_same<unsigned int, T>()) {
        GLCHKDBG(glUniform1ui(location, (unsigned int) value));
    } else if constexpr (std::is_same<float, T>()) {
        GLCHKDBG(glUniform1f(location, value));
    } else if constexpr (std::is_same<glm::vec2, T>()) {
        GLCHKDBG(glUniform2fv(location, 1, (const float*) &value));
    } else if constexpr (std::is_same<glm::vec3, T>()) {
        GLCHKDBG(glUniform3fv(location, 1, (const float*) &value));
    } else if constexpr (std::is_same<glm::vec4, T>()) {
        GLCHKDBG(glUniform4fv(location, 1, (const float*) &value));
    } else if constexpr (std::is_same<glm::mat3, T>()) {
        GLCHKDBG(glUniformMatrix3fv(location, 1, false, (const float*) &value));
    } else if constexpr (std::is_same<glm::mat4, T>()) {
        GLCHKDBG(glUniformMatrix4fv(location, 1, false, (const float*) &value));
    } else if constexpr (std::is_same<std::vector<float>, T>()) {
        //auto count = static_cast<GLsizei>(value.size());
        GLCHKDBG(glUniform1fv(location, static_cast<GLsizei>(value.size()), value.data()));
    } else {
        struct DependentFalse : public std::false_type {};
        static_assert(DependentFalse::value, "unsupported uniform type");
    }
}

inline void clearScreen(GLbitfield flags, const glm::vec4& color = {0.f, 0.f, 0.f, 1.f}, float depth = 1.0f, uint32_t stencil = 0) {
    if (flags | GL_COLOR_BUFFER_BIT) {
        glClearColor(color.x, color.y, color.z, color.w);
    }
    if (flags | GL_DEPTH_BUFFER_BIT) {
        glClearDepthf(depth);
    }
    if (flags | GL_STENCIL_BUFFER_BIT) {
        glClearStencil(stencil);
    }
    GLCHKDBG(glClear(flags));
}

// -----------------------------------------------------------------------------
//
inline GLint getInt(GLenum name) {
    GLint value;
    glGetIntegerv(name, &value);
    return value;
}

inline GLint getInt(GLenum name, GLint i) {
    GLint value;
    glGetIntegeri_v(name, i, &value);
    return value;
}

template<GLenum TARGET>
struct QueryObject {
    enum Status {
        EMPTY,   // the query object is not created yet.
        IDLE,    // the query object is idle and ready to use.
        RUNNING, // in between begin() and end()
        PENDING, // query is issued. but result is yet to returned.
    };

    GLuint qo     = 0;
    Status status = EMPTY;

    QueryObject() = default;
    ~QueryObject() { cleanup(); }

    SNN_NO_COPY(QueryObject);

    // can move
    QueryObject(QueryObject&& that) {
        qo          = that.qo;
        status      = that.status;
        that.qo     = 0;
        that.status = EMPTY;
    }
    QueryObject& operator=(QueryObject&& that) {
        if (this != &that) {
            qo          = that.qo;
            status      = that.status;
            that.qo     = 0;
            that.status = EMPTY;
        }
    }

    bool empty() const { return EMPTY == status; }
    bool idle() const { return IDLE == status; }
    bool running() const { return RUNNING == status; }
    bool pending() const { return PENDING == status; }

    void cleanup() {
#ifdef __ANDROID__
        if (qo) {
            glDeleteQueriesEXT(1, &qo), qo = 0;
        }
#else
        if (qo) {
            glDeleteQueries(1, &qo), qo = 0;
        }
#endif
        status = IDLE;
    }

    void allocate() {
        cleanup();
#ifdef __ANDROID__
        GLCHKDBG(glGenQueriesEXT(1, &qo));
#else
        GLCHKDBG(glGenQueries(1, &qo));
#endif
        status = IDLE;
    }

    void begin() {
        if (IDLE == status) {
#ifdef __ANDROID__
            GLCHKDBG(glBeginQueryEXT(TARGET, qo));
#else
            GLCHKDBG(glBeginQuery(TARGET, qo));
#endif
            status = RUNNING;
        } else
            SNN_LOGI("start failed!");
    }

    void end() {
        if (RUNNING == status) {
#ifdef __ANDROID__
            GLCHKDBG(glEndQueryEXT(TARGET));
#else
            GLCHKDBG(glEndQuery(TARGET));
#endif
            status = PENDING;
        }
    }

    void mark() {
        if (IDLE == status) {
#ifdef __ANDROID__
            glQueryCounterEXT(qo, TARGET);
#else
            glQueryCounter(qo, TARGET);
#endif
            status = PENDING;
        }
    }

    bool getResult(uint64_t& result) {
        if (PENDING != status) {
            SNN_LOGI("get result error, not pending");
            return false;
        }
        GLint available = false;
        while (!available) {
#ifdef __ANDROID__
            glGetQueryObjectivEXT(qo, GL_QUERY_RESULT_AVAILABLE, &available);
#else
            glGetQueryObjectiv(qo, GL_QUERY_RESULT_AVAILABLE, &available);
#endif
        }

#ifdef __ANDROID__
        GLCHKDBG(glGetQueryObjectui64vEXT(qo, GL_QUERY_RESULT, &result));
#else
        GLCHKDBG(glGetQueryObjectui64v(qo, GL_QUERY_RESULT, &result));
#endif
        status = IDLE;
        // SNN_LOGD("get result status set to idle(1) %" PRIu64 "\n", result);
        return true;
    }

    // returns 0, if the query is still pending.
    template<uint64_t DEFAULT_VALUE = 0>
    uint64_t getResult() const {
        uint64_t ret = DEFAULT_VALUE;
        return getResult(ret) ? ret : DEFAULT_VALUE;
    }
};

// -----------------------------------------------------------------------------
// Helper class to manage GL buffer object.
template<GLenum TARGET, size_t MIN_GPU_BUFFER_LENGH = 0>
struct BufferObject {
    GLuint bo            = 0;
    size_t length        = 0; // buffer length in bytes.
    GLenum mapped_target = 0;

    SNN_NO_COPY(BufferObject);
    SNN_NO_MOVE(BufferObject);

    BufferObject() {}

    ~BufferObject() { cleanup(); }

    static GLenum GetTarget() { return TARGET; }

    template<typename T, GLenum T2 = TARGET>
    void allocate(size_t count, const T* ptr, GLenum usage = GL_STATIC_DRAW) {
        cleanup();
        GLCHK(glGenBuffers(1, &bo));
        // Note: ARM Mali GPU doesn't work well with zero sized buffers. So
        // we create buffer that is large enough to hold at least one element.
        length = std::max(count, MIN_GPU_BUFFER_LENGH) * sizeof(T);
        GLCHK(glBindBuffer(T2, bo));
        GLCHK(glBufferData(T2, length, ptr, usage));
        GLCHK(glBindBuffer(T2, 0)); // unbind
    }

    void cleanup() {
        if (bo) {
            glDeleteBuffers(1, &bo), bo = 0;
        }
        length = 0;
    }

    bool empty() const { return 0 == bo; }

    template<typename T, GLenum T2 = TARGET>
    void update(const T* ptr, size_t offset = 0, size_t count = 1) const {
        GLCHKDBG(glBindBuffer(T2, bo));
        GLCHKDBG(glBufferSubData(T2, offset * sizeof(T), count * sizeof(T), ptr));
    }

    template<GLenum T2 = TARGET>
    void bind() const {
        GLCHKDBG(glBindBuffer(T2, bo));
    }

    template<GLenum T2 = TARGET>
    static void unbind() {
        GLCHKDBG(glBindBuffer(T2, 0));
    }

    template<GLenum T2 = TARGET>
    void bindBase(GLuint base) const {
        GLCHKDBG(glBindBufferBase(T2, base, bo));
    }

    template<typename T, GLenum T2 = TARGET>
    void getData(T* ptr, size_t offset, size_t count) const {
        GLCHKDBG(glBindBuffer(T2, bo));
        void* mapped = nullptr;
        GLCHKDBG(mapped = glMapBufferRange(T2, offset * sizeof(T), count * sizeof(T), GL_MAP_READ_BIT));
        if (mapped) {
            memcpy(ptr, mapped, count * sizeof(T));
            GLCHKDBG(glUnmapBuffer(T2));
        }
    }

    template<GLenum T2 = TARGET>
    void* map(size_t offset, size_t count, GLbitfield bufMask = GL_MAP_READ_BIT) {
        bind();
        void* ptr = nullptr;
        GLCHKDBG(ptr = glMapBufferRange(T2, offset, count, bufMask));
        assert(ptr);
        mapped_target = TARGET;
        return ptr;
    }

    template<GLenum T2 = TARGET>
    void* map(GLbitfield bufMask = GL_MAP_READ_BIT) {
        return map<T2>(0, length, bufMask);
    }

    void unmap() {
        if (mapped_target) {
            bind();
            GLCHKDBG(glUnmapBuffer(mapped_target));
            mapped_target = 0;
        }
    }

    operator GLuint() const { return bo; }

    GLuint getId() const { return bo; }
};

// -----------------------------------------------------------------------------
//
template<typename T, GLenum TARGET, size_t MIN_GPU_BUFFER_LENGTH = 0>
struct TypedBufferObject {
    std::vector<T> c;                                  // CPU data
    gl::BufferObject<TARGET, MIN_GPU_BUFFER_LENGTH> g; // GPU data

    void allocateGpuBuffer() { g.allocate(c.size(), c.data()); }

    void syncGpuBuffer() { g.update(c.data(), 0, c.size()); }

    // Synchornosly copy buffer content from GPU to CPU.
    // Note that this call is EXTREMELY expensive, since it stalls both CPU and GPU.
    void syncToCpu() {
        glFinish();
        g.getData(c.data(), 0, c.size());
    }

    void cleanup() {
        c.clear();
        g.cleanup();
    }
};

// -----------------------------------------------------------------------------
//
class VertexArrayObject {
    GLuint _va = 0;

public:
    ~VertexArrayObject() { cleanup(); }

    void allocate() {
        cleanup();
        GLCHK(glGenVertexArrays(1, &_va));
    }

    void cleanup() {
        if (_va) {
            glDeleteVertexArrays(1, &_va), _va = 0;
        }
    }

    void bind() const { GLCHKDBG(glBindVertexArray(_va)); }

    void unbind() const { GLCHKDBG(glBindVertexArray(0)); }

    operator GLuint() const { return _va; }
};

// -----------------------------------------------------------------------------
//
struct AutoShader {
    GLuint shader;

    AutoShader(GLuint s = 0): shader(s) {}
    ~AutoShader() { cleanup(); }

    void cleanup() {
        if (shader) {
            glDeleteShader(shader), shader = 0;
        }
    }

    SNN_NO_COPY(AutoShader);

    // can move
    AutoShader(AutoShader&& rhs): shader(rhs.shader) { rhs.shader = 0; }
    AutoShader& operator=(AutoShader&& rhs) {
        if (this != &rhs) {
            cleanup();
            shader     = rhs.shader;
            rhs.shader = 0;
        }
        return *this;
    }

    operator GLuint() const { return shader; }
};

class SamplerObject {
    GLuint _id = 0;

public:
    SamplerObject() {}
    ~SamplerObject() { cleanup(); }

    SNN_NO_COPY(SamplerObject);

    // can move
    SamplerObject(SamplerObject&& that) {
        _id      = that._id;
        that._id = 0;
    }
    SamplerObject& operator=(SamplerObject&& that) {
        if (this != &that) {
            cleanup();
            _id      = that._id;
            that._id = 0;
        }
        return *this;
    }

    operator GLuint() const { return _id; }

    void allocate() {
        cleanup();
        GLCHKDBG(glGenSamplers(1, &_id));
        SNN_ASSERT(glIsSampler(_id));
    }
    void cleanup() {
        if (_id) {
            glDeleteSamplers(1, &_id), _id = 0;
        }
    }
    void bind(size_t unit) const {
        SNN_ASSERT(glIsSampler(_id));
        glBindSampler((GLuint) unit, _id);
    }
};

inline void bindTexture(GLenum target, uint32_t stage, GLuint texture) {
    GLCHKDBG(glActiveTexture(GL_TEXTURE0 + stage));
    GLCHKDBG(glBindTexture(target, texture));
}

class TextureObject {
public:
    // no copy
    TextureObject(const TextureObject&) = delete;
    TextureObject& operator=(const TextureObject&) = delete;

    // can move
    TextureObject(TextureObject&& rhs) noexcept: _desc(rhs._desc) { rhs._desc.id = 0; }
    TextureObject& operator=(TextureObject&& rhs) noexcept {
        if (this != &rhs) {
            cleanup();
            _desc        = rhs._desc;
            rhs._desc.id = 0;
            rhs._owned = false;

            _owned = true;
        }
        return *this;
    }

    // default constructor
    TextureObject() { cleanup(); }

    ~TextureObject() { cleanup(); }

    struct TextureDesc {
        GLuint id = 0; // all other fields are undefined, if id is 0.
        GLenum target;
        snn::ColorFormat format;
        uint32_t width;
        uint32_t height;
        uint32_t depth; // this is number of layers for 2D array texture and is always 6 for cube texture.
        uint32_t channels;
        uint32_t mips;
    };

    const TextureDesc& getDesc() const { return _desc; }

    GLenum target() const { return _desc.target; }
    GLenum id() const { return _desc.id; }

    bool empty() const { return 0 == _desc.id; }

    bool is2D() const { return GL_TEXTURE_2D == _desc.target; }

    bool isArray() const { return GL_TEXTURE_2D_ARRAY == _desc.target; }

    void attach(GLenum target, GLuint id);

    void attach(const TextureObject& that) { attach(that._desc.target, that._desc.id); }

    // In case some other object starts owning this texture.
    void detach() { _owned = false; }

    void allocate2D(snn::ColorFormat f, size_t w, size_t h, size_t channels, size_t m = 1);

    void allocate2D(snn::ColorFormat f, size_t w, size_t h);

    void allocate2DArray(snn::ColorFormat f, size_t w, size_t h, size_t l, size_t channels, size_t m = 1);

    void allocate2DArray(snn::ColorFormat f, size_t w, size_t h, size_t l);

    void allocate3D(snn::ColorFormat f, size_t w, size_t h, size_t l, size_t channels, size_t m = 1);

    void allocate3D(snn::ColorFormat f, size_t w, size_t h, size_t l);

    void allocateCube(snn::ColorFormat f, size_t w, size_t channels, size_t m = 1);

    void allocateCube(snn::ColorFormat f, size_t w);

    void setPixels(size_t level, size_t x, size_t y, size_t w, size_t h,
                    size_t rowPitchInBytes, // set to 0, if pixels are tightly packed.
                    const void* pixels) const;

    // Set to rowPitchInBytes 0, if pixels are tightly packed.
    void setPixels(size_t layer, size_t level, size_t x, size_t y, size_t w, size_t h, size_t rowPitchInBytes, const void* pixels) const;

    snn::ManagedRawImage getBaseLevelPixels(bool convertFp16ToFp32 = false) const;

    void cleanup() {
        if (_owned && _desc.id) {
            GLCHK(glDeleteTextures(1, &_desc.id));
        }
        _desc.id       = 0;
        _desc.target   = GL_NONE;
        _desc.format   = snn::ColorFormat::NONE;
        _desc.width    = 0;
        _desc.height   = 0;
        _desc.depth    = 0;
        _desc.channels = 0;
        _desc.mips     = 0;
    }

    void bind(size_t stage) const {
        GLCHKDBG(glActiveTexture(GL_TEXTURE0 + (int) stage));
        GLCHKDBG(glBindTexture(_desc.target, _desc.id));
    }

    void unbind() const { glBindTexture(_desc.target, 0); }

    operator GLuint() const { return _desc.id; }

private:
    TextureDesc _desc;
    bool _owned = false;
    void applyDefaultParameters();
};

// SSBO for in-shader debug output. Check out ftl/main_ps.glsl for example usage.
// It is currently working on Windows only. Running it on Android crashes the driver.
#if defined(_DEBUG)
    #define DEBUG_SSBO_ENABLED 1
#else
    #define DEBUG_SSBO_ENABLED 0
#endif
struct DebugSSBO {
#if DEBUG_SSBO_ENABLED
    std::vector<float> buffer;
    mutable std::vector<float> printed;
    int* counter = nullptr;
    gl::BufferObject<GL_SHADER_STORAGE_BUFFER> g;
#endif

    static constexpr bool isEnabled() {
#if DEBUG_SSBO_ENABLED
        return true;
#else
        return false;
#endif
    }

    ~DebugSSBO() { cleanup(); }

    void allocate(size_t n) {
#if DEBUG_SSBO_ENABLED
        cleanup();
        buffer.resize(n + 1);
        printed.resize(buffer.size());
        counter = (int*) &buffer[0];
        g.allocate(buffer.size(), buffer.data(), GL_STATIC_READ);
#else
        (void) n;
#endif
    }

    void bind(int slot = 15) const {
#if DEBUG_SSBO_ENABLED
        if (g) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, g);
        }
#else
        (void) slot;
#endif
    }

    void cleanup() {
#if DEBUG_SSBO_ENABLED
        buffer.clear();
        printed.clear();
        counter = nullptr;
        g.cleanup();
#endif
    }

    void clearCounter() {
#if DEBUG_SSBO_ENABLED
        if (!counter) {
            return;
        }
        *counter = 0;
        g.update(counter, 0, 1);
#endif
    }

    void pullDataFromGPU() {
#if DEBUG_SSBO_ENABLED
        if (buffer.empty()) {
            return;
        }
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        g.getData(buffer.data(), 0, buffer.size());
#endif
    }

    void printLastResult() const;
};

struct FullScreenQuad {
    // vertex array
    GLuint va = 0;
    gl::BufferObject<GL_ARRAY_BUFFER> vb;

    FullScreenQuad() {}

    ~FullScreenQuad() { cleanup(); }

    void allocate();

    void cleanup();

    void draw() const {
        SNN_ASSERT(va);
        GLCHKDBG(glBindVertexArray(va));
        GLCHKDBG(glDrawArrays(GL_TRIANGLES, 0, 3));
    }
};

class SimpleGlslProgram {
    GLuint _program = 0;

public:
    // optional program name (for debug log)
    std::string name;

#ifdef _DEBUG
    std::string vsSource, psSource, csSource;
#endif

    SNN_NO_COPY(SimpleGlslProgram);
    SNN_NO_MOVE(SimpleGlslProgram);

    SimpleGlslProgram(const char* optionalProgramName = nullptr) {
        if (optionalProgramName) {
            name = optionalProgramName;
        }
    }

    ~SimpleGlslProgram() { cleanup(); }

    bool loadVsPs(const char* vscode, const char* pscode) {
#ifdef _DEBUG
        if (vscode) {
            vsSource = vscode;
        }
        if (pscode) {
            psSource = pscode;
        }
#endif
        cleanup();
        AutoShader vs = loadShaderFromString(vscode, 0, GL_VERTEX_SHADER, name.c_str());
        AutoShader ps = loadShaderFromString(pscode, 0, GL_FRAGMENT_SHADER, name.c_str());
        if ((vscode && !vs) || (pscode && !ps)) {
            return false;
        }
        _program = linkProgram({vs, ps}, name.c_str());
        return _program != 0;
    }

    bool loadCs(const char* code) {
#ifdef _DEBUG
        if (code) {
            csSource = code;
        }
#endif
        cleanup();
        AutoShader cs = loadShaderFromString(code, 0, GL_COMPUTE_SHADER, name.c_str());
        if (!cs) {
            return false;
        }
        _program = linkProgram({cs}, name.c_str());
        return _program != 0;
    }

    void use() const { GLCHKDBG(glUseProgram(_program)); }

    void cleanup() {
        //_uniforms.clear();
        if (_program) {
            glDeleteProgram(_program), _program = 0;
        }
    }

    GLint getUniformLocation(const char* name_) const { return glGetUniformLocation(_program, name_); }

    GLint getUniformBinding(const char* name_) const {
        GLCHKDBG(auto loc = glGetUniformLocation(_program, name_));
        // SNN_LOGD("%s:%d bind %s loc: %d\n",__FILENAME__, __LINE__, name_, loc);
        if (-1 == loc) {
            return -1;
        }
        GLint binding;
        glGetUniformiv(_program, loc, &binding);
        return binding;
    }

    operator GLuint() const { return _program; }
};

class SimpleUniform {
public:
    using Value = std::variant<int, unsigned int, float, glm::vec2, glm::vec3, glm::vec4, glm::ivec2, glm::ivec3, glm::ivec4, glm::uvec2, glm::uvec3,
                               glm::uvec4, glm::mat3x3, glm::mat4x4, std::vector<float>>;

    Value value;

    SimpleUniform(const std::string& name): _name(name) {
        SNN_ASSERT(!_name.empty());
    }

    template<typename T>
    SimpleUniform(const std::string& name, const T& v): value(v), _name(name) {
        SNN_ASSERT(!_name.empty());
    }

    bool init(GLuint program) {
        if (program > 0) {
            GLCHKDBG(_location = glGetUniformLocation(program, _name.c_str()));
        } else {
            _location = -1;
        }
        return _location > -1;
    }

    template<typename T>
    void update(const T& v) {
        value = v;
    }

    std::string getName() {
        return _name;
    }

    void apply() const {
        SNN_ASSERT(!_name.empty());
        if (_location < 0) {
            return;
        }
        std::visit(
            [&](auto&& v) {
                // SNN_LOGD("%s:%d uniform apply  %s: %d\n",__FILENAME__, __LINE__, _name.c_str(), _location);
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, int>)
                    glUniform1i(_location, v);
                else if constexpr (std::is_same_v<T, unsigned int>)
                    glUniform1ui(_location, v);
                else if constexpr (std::is_same_v<T, float>)
                    glUniform1f(_location, v);
                else if constexpr (std::is_same_v<T, glm::vec2>)
                    glUniform2f(_location, v.x, v.y);
                else if constexpr (std::is_same_v<T, glm::vec3>)
                    glUniform3f(_location, v.x, v.y, v.z);
                else if constexpr (std::is_same_v<T, glm::vec4>)
                    glUniform4f(_location, v.x, v.y, v.z, v.w);
                else if constexpr (std::is_same_v<T, glm::ivec2>)
                    glUniform2i(_location, v.x, v.y);
                else if constexpr (std::is_same_v<T, glm::ivec3>)
                    glUniform3i(_location, v.x, v.y, v.z);
                else if constexpr (std::is_same_v<T, glm::ivec4>)
                    glUniform4i(_location, v.x, v.y, v.z, v.w);
                else if constexpr (std::is_same_v<T, glm::uvec2>)
                    glUniform2ui(_location, v.x, v.y);
                else if constexpr (std::is_same_v<T, glm::uvec3>)
                    glUniform3ui(_location, v.x, v.y, v.z);
                else if constexpr (std::is_same_v<T, glm::uvec4>)
                    glUniform4ui(_location, v.x, v.y, v.z, v.w);
                else if constexpr (std::is_same_v<T, glm::mat3x3>)
                    glUniformMatrix3fv(_location, 1, false, glm::value_ptr(v));
                else if constexpr (std::is_same_v<T, glm::mat4x4>)
                    glUniformMatrix4fv(_location, 1, false, glm::value_ptr(v));
                else if constexpr (std::is_same_v<T, const std::vector<float>>)
                    glUniform1fv(_location, (GLsizei) v.size(), v.data());
            },
            value);
    }

private:
    const std::string _name;
    GLint _location = -1;
};

// -----------------------------------------------------------------------------
// For asynchronous timer (not time stamp) queries
struct GpuTimeElapsedQuery : virtual DeviceTimer {
    std::string name;

    explicit GpuTimeElapsedQuery(std::string n = std::string("")): name(n) { _q.allocate(); }

    ~GpuTimeElapsedQuery() {}

    // returns duration in nanoseconds
    GLuint64 duration() const  override { return _result; }

    void start() override {
        // SNN_LOGI("%s before start status: %d", name.c_str(), _q.status);
        _q.begin();
        // SNN_LOGI("%s after start status: %d", name.c_str(), _q.status);
    }

    void stop() override;

    void getTime() override;

    // Print stats to string
    std::string print() const override;

    friend inline std::ostream& operator<<(std::ostream& s, const GpuTimeElapsedQuery& t) {
        s << t.print();
        return s;
    }

private:
    QueryObject<GL_TIME_ELAPSED> _q;
    uint64_t _result = 0;
};

// -----------------------------------------------------------------------------
// GPU time stamp query
class GpuTimestamps {
public:
    explicit GpuTimestamps(std::string name = std::string("")): _name(name) {}

    void start() {
        SNN_ASSERT(!_started);
        if (!_started) {
            _started = true;
            _count   = 0;
            mark("start time");
        }
    }

    void stop() {
        SNN_ASSERT(_started);
        if (_started) {
            mark("end time");
            _started = false;
        }
    }

    void mark(std::string name) {
        SNN_ASSERT(_started);
        if (!_started) {
            return;
        }

        if (_count == _marks.size()) {
            _marks.emplace_back();
            _marks.back().name = name;
        }

        SNN_ASSERT(_count < _marks.size());
        _marks[_count++].mark();
    }

    // Print stats of timestamps to string
    std::string print(const char* ident = nullptr) const;

private:
    struct Query {
        std::string name;
        QueryObject<GL_TIMESTAMP> q;
        uint64_t result = 0;
        Query() { q.allocate(); }
        SNN_NO_COPY(Query);
        SNN_DEFAULT_MOVE(Query);
        void mark() {
            if (q.idle()) {
                q.mark();
            } else {
                q.getResult(result);
            }
        }
    };

    std::string _name;
    std::vector<Query> _marks;
    size_t _count = 0;
    bool _started = false;
};

// -----------------------------------------------------------------------------
// Creates an OpenGL context
class RenderContext {
    class Impl;
    Impl* _impl;

public:
    using WindowHandle = intptr_t;

    enum Type {
        STANDALONE,
        SHARED,
    };

    RenderContext(Type type, WindowHandle externalWindow = 0);
    ~RenderContext();

    SNN_NO_COPY(RenderContext);

    // can move
    RenderContext(RenderContext&& that): _impl(that._impl) { that._impl = nullptr; }
    RenderContext& operator=(RenderContext&& that);

    void makeCurrent();

    void swapBuffers();
};

// Store and restore OpenGL context
class RenderContextStack {
    class Impl;
    Impl* _impl;

public:
    SNN_NO_COPY(RenderContextStack);
    SNN_NO_MOVE(RenderContextStack);

    RenderContextStack();

    // the destructor will automatically pop out any previously pushed context.
    ~RenderContextStack();

    // push store current context to the top of the stack
    void push();

    // apply previous stored context without pop it out of the stack.
    void apply();

    // pop previously stored context
    void pop();
};

#define OPENGL_CHECK_ERROR void()
#define GLASSERT           assert
class GLSSBOBuffer {
public:
    GLSSBOBuffer(GLsizeiptr size, GLenum type = GL_SHADER_STORAGE_BUFFER, GLenum usage = GL_DYNAMIC_DRAW) {
        mType = type;
        GLASSERT(size > 0);
        glGenBuffers(1, &mId);
        OPENGL_CHECK_ERROR;
        glBindBuffer(mType, mId);
        OPENGL_CHECK_ERROR;
        GLASSERT(mId > 0);
        glBufferData(mType, size, NULL, usage);
        OPENGL_CHECK_ERROR;
        mSize = size;
    }

    ~GLSSBOBuffer() {
        glDeleteBuffers(1, &mId);
        OPENGL_CHECK_ERROR;
    }

    GLuint getId() const { return mId; }
    void* map(GLbitfield bufMask);
    void unmap();

    GLsizeiptr size() const { return mSize; }

private:
    GLuint mId = 0;
    GLsizeiptr mSize;
    GLenum mType;
};

} // namespace gl
