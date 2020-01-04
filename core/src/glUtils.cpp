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
#include "pch.h"
#include "snn/glUtils.h"
#include <opencv2/opencv.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <stack>

#ifdef __ANDROID__
    #include <dlfcn.h>
#endif

using namespace snn;

#ifdef _DEBUG
// -----------------------------------------------------------------------------
//
static void initializeOpenGLDebugRuntime() {
    struct OGLDebugOutput {
        static const char* source2String(GLenum source) {
            switch (source) {
            case GL_DEBUG_SOURCE_API_ARB:
                return "GL API";
            case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
                return "Window System";
            case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
                return "Shader Compiler";
            case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
                return "Third Party";
            case GL_DEBUG_SOURCE_APPLICATION_ARB:
                return "Application";
            case GL_DEBUG_SOURCE_OTHER_ARB:
                return "Other";
            default:
                return "INVALID_SOURCE";
            }
        }

        static const char* type2String(GLenum type) {
            switch (type) {
            case GL_DEBUG_TYPE_ERROR_ARB:
                return "Error";
            case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
                return "Deprecation";
            case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
                return "Undefined Behavior";
            case GL_DEBUG_TYPE_PORTABILITY_ARB:
                return "Portability";
            case GL_DEBUG_TYPE_PERFORMANCE_ARB:
                return "Performance";
            case GL_DEBUG_TYPE_OTHER_ARB:
                return "Other";
            default:
                return "INVALID_TYPE";
            }
        }

        static const char* severity2String(GLenum severity) {
            switch (severity) {
            case GL_DEBUG_SEVERITY_HIGH_ARB:
                return "High";
            case GL_DEBUG_SEVERITY_MEDIUM_ARB:
                return "Medium";
            case GL_DEBUG_SEVERITY_LOW_ARB:
                return "Low";
            default:
                return "INVALID_SEVERITY";
            }
        }

        static void GLAPIENTRY messageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                               GLsizei, // length,
                                               const GLchar* message,
                                               const void*) // userParam)
        {
            // Determine log level
            bool error_  = false;
            bool warning = false;
            bool info    = false;
            switch (type) {
            case GL_DEBUG_TYPE_ERROR_ARB:
            case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
                error_ = true;
                break;

            case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
            case GL_DEBUG_TYPE_PORTABILITY:
                switch (severity) {
                case GL_DEBUG_SEVERITY_HIGH_ARB:
                case GL_DEBUG_SEVERITY_MEDIUM_ARB:
                    warning = true;
                    break;
                case GL_DEBUG_SEVERITY_LOW_ARB:
                    break;
                default:
                    error_ = true;
                    break;
                }
                break;

            case GL_DEBUG_TYPE_PERFORMANCE_ARB:
                switch (severity) {
                case GL_DEBUG_SEVERITY_HIGH_ARB:
                    warning = true;
                    break;
                case GL_DEBUG_SEVERITY_MEDIUM_ARB: // shader recompiliation, buffer data read back.
                case GL_DEBUG_SEVERITY_LOW_ARB:
                    break; // verbose: performance warnings from redundant state changes
                default:
                    error_ = true;
                    break;
                }
                break;

            case GL_DEBUG_TYPE_OTHER_ARB:
                switch (severity) {
                case GL_DEBUG_SEVERITY_HIGH_ARB:
                    error_ = true;
                    break;
                case GL_DEBUG_SEVERITY_MEDIUM_ARB:
                    warning = true;
                    break;
                case GL_DEBUG_SEVERITY_LOW_ARB:
                case GL_DEBUG_SEVERITY_NOTIFICATION:
                    break; // verbose
                default:
                    error_ = true;
                    break;
                }
                break;

            default:
                error_ = true;
                break;
            }

            std::string s = formatString("(id=[%d] source=[%s] type=[%s] severity=[%s]): %s\n%s", id, source2String(source), type2String(type),
                                         severity2String(severity), message, dumpCallStack().c_str());
            if (error_) {
                SNN_LOGE("[GL ERROR] %s", s.c_str());
            } else if (warning) {
                // SNN_LOGW("[GL WARNING] %s", s.c_str());
                (void) warning; // Disable it for debug
            } else if (info) {
                SNN_LOGI("[GL INFO] %s", s.c_str());
            }
        }
    };

    if (GLAD_GL_KHR_debug) {
        GLCHK(glDebugMessageCallback(&OGLDebugOutput::messageCallback, nullptr));
        GLCHK(glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS));
    } else if (GLAD_GL_ARB_debug_output) {
        GLCHK(glDebugMessageCallbackARB(&OGLDebugOutput::messageCallback, nullptr));
        // enable all messages
        GLCHK(glDebugMessageControlARB(GL_DONT_CARE, // source
                                    GL_DONT_CARE, // type
                                    GL_DONT_CARE, // severity
                                    0,            // count
                                    nullptr,      // ids
                                    GL_TRUE));
    }
}
#endif

// -----------------------------------------------------------------------------
//
static void printGLInfo(bool printExtensionList) {
    std::stringstream info;

    // vendor and version info.
    const char* vendor   = (const char*) glGetString(GL_VENDOR);
    const char* version  = (const char*) glGetString(GL_VERSION);
    const char* renderer = (const char*) glGetString(GL_RENDERER);
    const char* glsl     = (const char*) glGetString(GL_SHADING_LANGUAGE_VERSION);
    GLint maxsls = -1, maxslsFast = -1;
    if (GLAD_GL_EXT_shader_pixel_local_storage) {
        glGetIntegerv(GL_MAX_SHADER_PIXEL_LOCAL_STORAGE_SIZE_EXT, &maxsls);
        glGetIntegerv(GL_MAX_SHADER_PIXEL_LOCAL_STORAGE_FAST_SIZE_EXT, &maxslsFast);
    }
    info << "\n\n"
        "===================================================\n"
        "        OpenGL Implementation Informations\n"
        "---------------------------------------------------\n"
        "               OpenGL vendor : "
        << vendor
        << "\n"
        "              OpenGL version : "
        << version
        << "\n"
        "             OpenGL renderer : "
        << renderer
        << "\n"
        "                GLSL version : "
        << glsl
        << "\n"
        "       Max VT uniform blocks : "
        << gl::getInt(GL_MAX_VERTEX_UNIFORM_BLOCKS)
        << "\n"
        "       Max GM uniform blocks : "
        << gl::getInt(GL_MAX_GEOMETRY_UNIFORM_BLOCKS)
        << "\n"
        "       Max FS uniform blocks : "
        << gl::getInt(GL_MAX_FRAGMENT_UNIFORM_BLOCKS)
        << "\n"
        "      Max uniform block size : "
        << gl::getInt(GL_MAX_UNIFORM_BLOCK_SIZE)
        << " bytes\n"
        "           Max texture units : "
        << gl::getInt(GL_MAX_TEXTURE_IMAGE_UNITS)
        << "\n"
        "    Max array texture layers : "
        << gl::getInt(GL_MAX_ARRAY_TEXTURE_LAYERS)
        << "\n"
        "       Max color attachments : "
        << gl::getInt(GL_MAX_COLOR_ATTACHMENTS)
        << "\n"
        "           Max SSBO binding  : "
        << gl::getInt(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)
        << "\n"
        "         Max SSBO FS blocks  : "
        << gl::getInt(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS)
        << "\n"
        "        Max SSBO block size  : "
        << gl::getInt(GL_MAX_SHADER_STORAGE_BLOCK_SIZE) * 4
        << " bytes\n"
        "       Max CS WorkGroup size : "
        << gl::getInt(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0) << "," << gl::getInt(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1) << ","
        << gl::getInt(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2)
        << "\n"
        "      Max CS WorkGroup count : "
        << gl::getInt(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0) << "," << gl::getInt(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1) << ","
        << gl::getInt(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2)
        << "\n"
        "    Max shader local storage : total="
        << maxsls << ", fast=" << maxslsFast << "\n";

    if (printExtensionList) {
        info << "---------------------------------------------------\n";
        std::vector<std::string> extensions;
        GLint num = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &num);
        for (int i = 0; i < num; ++i) {
            extensions.push_back((const char*) glGetStringi(GL_EXTENSIONS, i));
        }
        std::sort(extensions.begin(), extensions.end());
        for (int i = 0; i < num; ++i) {
            info << "    " << extensions[i] << "\n";
        }
    }

    info << "===================================================\n";

    SNN_LOGI(info.str().c_str());
}

// -----------------------------------------------------------------------------
//
void gl::initGLExtensions(bool printExtensionList) {
    static std::atomic_bool initialized = false;
    if (initialized.exchange(true)) {
        return;
    }

#ifdef __ANDROID__
    typedef void* (*GetProcAddress)(const char* name);
    auto gpa = (GetProcAddress) dlsym(nullptr, "eglGetProcAddress");
    SNN_CHK(gpa);
    SNN_CHK(gladLoadGLES2Loader(gpa));
#else
    SNN_CHK(gladLoadGL());
#endif

#if _DEBUG
    initializeOpenGLDebugRuntime();
#endif

    printGLInfo(printExtensionList);
}

// -----------------------------------------------------------------------------
//
void gl::TextureObject::attach(GLenum target, GLuint id) {
    cleanup();
    _owned       = false;
    _desc.target = target;
    _desc.id     = id;
    bind(0);
    SNN_LOGD("%s:%d attach TEXTURE %x:%d, w:%d, h:%d, d:%d\n", __FUNCTION__, __LINE__, target, id, _desc.width, _desc.height, _desc.depth);
    // std::cout << "Texture Width: " << _desc.width << std::endl;

    glGetTexLevelParameteriv(_desc.target, 0, GL_TEXTURE_WIDTH, (GLint*) &_desc.width);
    SNN_ASSERT(_desc.width);

    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_HEIGHT, (GLint*) &_desc.height);
    SNN_ASSERT(_desc.height);

    // determine depth
    switch (target) {
    case GL_TEXTURE_2D_ARRAY:
    case GL_TEXTURE_3D:
        glGetTexLevelParameteriv(target, 0, GL_TEXTURE_DEPTH, (GLint*) &_desc.depth);
        SNN_ASSERT(_desc.depth);
        break;
    case GL_TEXTURE_CUBE_MAP:
        _desc.depth = 6;
        break;
    default:
        _desc.depth = 1;
        break;
    }

    GLint maxLevel;
    glGetTexParameteriv(target, GL_TEXTURE_MAX_LEVEL, &maxLevel);
    _desc.mips = (uint32_t) maxLevel + 1;

    int internalFormat = 0;
    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_INTERNAL_FORMAT, &internalFormat);
    _desc.format   = fromGLInternalFormat(internalFormat);
    _desc.channels = getColorFormatDesc(_desc.format).ch * _desc.depth;
    // SNN_LOGI("%s:%d attach TEXTURE %d:%d, w:%d, h:%d, d:%d\n", __FUNCTION__, __LINE__,target, id, _desc.width, _desc.height, _desc.depth);
    unbind();
}

// -----------------------------------------------------------------------------
//
void gl::TextureObject::allocate2D(ColorFormat f, size_t w, size_t h, size_t channels, size_t m) {
    cleanup();
    _desc.target   = GL_TEXTURE_2D;
    _desc.format   = f;
    _desc.width    = (uint32_t) w;
    _desc.height   = (uint32_t) h;
    _desc.depth    = (uint32_t) 1;
    _desc.channels = (uint32_t) channels;
    _desc.mips     = (uint32_t) m;
    _owned         = true;
    GLCHK(glGenTextures(1, &_desc.id));
    GLCHK(glBindTexture(_desc.target, _desc.id));
    applyDefaultParameters();
    const auto& cd = getColorFormatDesc(_desc.format);
    GLCHK(glTexStorage2D(_desc.target, (GLsizei) _desc.mips, cd.glInternalFormat, (GLsizei) _desc.width, (GLsizei) _desc.height));
    GLCHK(glBindTexture(_desc.target, 0));
}

// -----------------------------------------------------------------------------
//
void gl::TextureObject::allocate2DArray(ColorFormat f, size_t w, size_t h, size_t l, size_t channels, size_t m) {
    cleanup();
    _desc.target   = GL_TEXTURE_2D_ARRAY;
    _desc.format   = f;
    _desc.width    = (uint32_t) w;
    _desc.height   = (uint32_t) h;
    _desc.depth    = (uint32_t) l;
    _desc.channels = (uint32_t) channels;
    _desc.mips     = (uint32_t) m;
    _owned         = true;
    GLCHK(glGenTextures(1, &_desc.id));
    GLCHK(glBindTexture(_desc.target, _desc.id));
    applyDefaultParameters();
    const auto& cd = getColorFormatDesc(_desc.format);
    GLCHK(glTexStorage3D(_desc.target, (GLsizei) _desc.mips, cd.glInternalFormat, (GLsizei) _desc.width, (GLsizei) _desc.height, (GLsizei) _desc.depth));
    GLCHK(glBindTexture(_desc.target, 0));
}

// -----------------------------------------------------------------------------
//
void gl::TextureObject::allocate3D(ColorFormat f, size_t w, size_t h, size_t l, size_t m, size_t channels) {
    cleanup();
    _desc.target   = GL_TEXTURE_3D;
    _desc.format   = f;
    _desc.width    = (uint32_t) w;
    _desc.height   = (uint32_t) h;
    _desc.depth    = (uint32_t) l;
    _desc.channels = (uint32_t) channels;
    _desc.mips     = (uint32_t) m;
    _owned         = true;
    GLCHK(glGenTextures(1, &_desc.id));
    GLCHK(glBindTexture(_desc.target, _desc.id));
    applyDefaultParameters();
    const auto& cd = getColorFormatDesc(_desc.format);
    GLCHK(glTexStorage3D(_desc.target, (GLsizei) _desc.mips, cd.glInternalFormat, (GLsizei) _desc.width, (GLsizei) _desc.height, (GLsizei) _desc.depth));
    GLCHK(glBindTexture(_desc.target, 0));
}

// -----------------------------------------------------------------------------
//
void gl::TextureObject::allocateCube(ColorFormat f, size_t w, size_t channels, size_t m) {
    cleanup();
    _desc.target   = GL_TEXTURE_CUBE_MAP;
    _desc.format   = f;
    _desc.width    = (uint32_t) w;
    _desc.height   = (uint32_t) w;
    _desc.depth    = 6;
    _desc.channels = (uint32_t) channels;
    _desc.mips     = (uint32_t) m;
    _owned         = true;
    GLCHK(glGenTextures(1, &_desc.id));
    GLCHK(glBindTexture(GL_TEXTURE_CUBE_MAP, _desc.id));
    applyDefaultParameters();
    const auto& cd = snn::getColorFormatDesc(_desc.format);
    GLCHK(glTexStorage2D(GL_TEXTURE_CUBE_MAP, (GLsizei) _desc.mips, cd.glInternalFormat, (GLsizei) _desc.width, (GLsizei) _desc.width));
    GLCHK(glBindTexture(_desc.target, 0));
}

void gl::TextureObject::applyDefaultParameters() {
    SNN_ASSERT(_desc.width > 0);
    SNN_ASSERT(_desc.height > 0);
    SNN_ASSERT(_desc.depth > 0);
    SNN_ASSERT(_desc.mips > 0);
    GLCHK(glTexParameteri(_desc.target, GL_TEXTURE_BASE_LEVEL, 0));
    GLCHK(glTexParameteri(_desc.target, GL_TEXTURE_MAX_LEVEL, _desc.mips - 1));
    GLCHK(glTexParameteri(_desc.target, GL_TEXTURE_MIN_FILTER, _desc.mips > 1 ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST));
    GLCHK(glTexParameteri(_desc.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GLCHK(glTexParameteri(_desc.target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GLCHK(glTexParameteri(_desc.target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
}

// -----------------------------------------------------------------------------
//
void gl::TextureObject::setPixels(size_t level, size_t x, size_t y, size_t w, size_t h, size_t rowPitchInBytes, const void* pixels) const {
    if (empty()) {
        return;
    }
    GLCHK(glBindTexture(_desc.target, _desc.id));
    auto& cf = getColorFormatDesc(_desc.format);
    SNN_ASSERT(0 == (rowPitchInBytes * 8 % cf.bits));
    GLCHK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GLCHK(glPixelStorei(GL_UNPACK_ROW_LENGTH, (int) (rowPitchInBytes * 8 / cf.bits)));
    GLCHK(glTexSubImage2D(_desc.target, (GLint) level, (GLint) x, (GLint) y, (GLsizei) w, (GLsizei) h, cf.glFormat, cf.glType, pixels));
    GLCHK(glPixelStorei(GL_UNPACK_ROW_LENGTH, 0));
    //GLCHK(;);
}

void gl::TextureObject::setPixels(size_t layer, size_t level, size_t x, size_t y, size_t w, size_t h, size_t rowPitchInBytes, const void* pixels) const {
    if (empty()) {
        return;
    }

    GLCHKDBG(glBindTexture(_desc.target, _desc.id));
    auto& cf = getColorFormatDesc(_desc.format);
    SNN_ASSERT(0 == (rowPitchInBytes * 8 % cf.bits));
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, (int) (rowPitchInBytes * 8 / cf.bits));
    GLCHKDBG(glTexSubImage3D(_desc.target, (GLint) level, (GLint) x, (GLint) y, (GLint) layer, (GLsizei) w, (GLsizei) h, 1, cf.glFormat, cf.glType, pixels));

    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    //GLCHK(;);
}

void readTexture(int buf_size, GLuint textureId) {
    char* outBuffer = (char*) malloc(buf_size * 16);
    glActiveTexture(GL_TEXTURE0);
    CHECK_GL_ERROR("glActiveTexture");
    glBindTexture(GL_TEXTURE_2D_ARRAY, textureId);
    CHECK_GL_ERROR("glBindTexture");

    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_FLOAT, outBuffer);
    CHECK_GL_ERROR("glGetTexImage");

    float* dest = (float*) outBuffer;
    SNN_LOGD(">>>>>>>>>>readTexture>>>>>>>>>>>>>\n");
    for (int i = 0; i < buf_size * 4; i += 4) {
        SNN_LOGD("%d, %f\n", i, *(dest + i));
    }
    SNN_LOGD("<<<<<<<<<<<<<<<<<<<<<<<\n");
    free(outBuffer);
}

static inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
    return as_float(w);
#elif defined(__CUDA_ARCH__)
    return __uint_as_float((unsigned int) w);
#elif defined(__INTEL_COMPILER)
    return _castu32_f32(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
    return _CopyFloatFromInt32((__int32) w);
#else
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = {w};
    return fp32.as_value;
#endif
}

static inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
    return as_uint(f);
#elif defined(__CUDA_ARCH__)
    return (uint32_t) __float_as_uint(f);
#elif defined(__INTEL_COMPILER)
    return _castf32_u32(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
    return (uint32_t) _CopyInt32FromFloat(f);
#else
    union {
        float as_value;
        uint32_t as_bits;
    } fp32 = {f};
    return fp32.as_bits;
#endif
}

static inline double fp64_from_bits(uint64_t w) {
#if defined(__OPENCL_VERSION__)
    return as_double(w);
#elif defined(__CUDA_ARCH__)
    return __longlong_as_double((long long) w);
#elif defined(__INTEL_COMPILER)
    return _castu64_f64(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
    return _CopyDoubleFromInt64((__int64) w);
#else
    union {
        uint64_t as_bits;
        double as_value;
    } fp64 = {w};
    return fp64.as_value;
#endif
}

static inline uint64_t fp64_to_bits(double f) {
#if defined(__OPENCL_VERSION__)
    return as_ulong(f);
#elif defined(__CUDA_ARCH__)
    return (uint64_t) __double_as_longlong(f);
#elif defined(__INTEL_COMPILER)
    return _castf64_u64(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
    return (uint64_t) _CopyInt64FromDouble(f);
#else
    union {
        double as_value;
        uint64_t as_bits;
    } fp64 = {f};
    return fp64.as_bits;
#endif
}

// -----------------------------------------------------------------------------
//
snn::ManagedRawImage gl::TextureObject::getBaseLevelPixels() const {
    if (empty()) {
        return {};
    }
    // SNN_LOGI("%s:%d %d,%d,%d,%d,%d\n", __FUNCTION__,__LINE__, _desc.format, _desc.width, _desc.height, _desc.depth, _desc.channels);
    snn::ManagedRawImage image(ImageDesc(_desc.format, _desc.width, _desc.height, _desc.depth, _desc.channels));
    GLint pixelBufferIndex = -1;
    GLuint _frameBuffer    = 0;
    GLCHK(glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pixelBufferIndex));
    if (pixelBufferIndex != 0) {
        glBindTexture(_desc.target, _desc.id);
        GLCHK(glGenBuffers(1, (GLuint*) &pixelBufferIndex));
        GLCHK(glBindBuffer(GL_PIXEL_PACK_BUFFER, (GLuint) pixelBufferIndex));
        GLCHK(glReadBuffer(GL_COLOR_ATTACHMENT0));
        auto& cf = snn::getColorFormatDesc(_desc.format);
        GLCHK(glReadPixels(0, 0, _desc.width, _desc.height, GL_RGBA, cf.glInternalFormat, 0));
        SNN_LOGD("Downloading PBO: %d", pixelBufferIndex);
        glGetTexImage(_desc.target, 0, cf.glFormat, cf.glType, 0);
        uint8_t* pbo = (uint8_t*) glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        if (_desc.format == ColorFormat::RGBA16F) {
            snn::ManagedRawImage rgba16fImage(ImageDesc(_desc.format, _desc.width, _desc.height, _desc.depth, _desc.channels), pbo, image.size());
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            auto rgba32fImage = snn::toRgba32f(rgba16fImage);
            snn::ManagedRawImage retImage(ImageDesc(ColorFormat::RGBA32F, _desc.width, _desc.height, _desc.depth, _desc.channels), rgba32fImage.data(),
                                          rgba32fImage.size());
            return retImage;
        } else {
            snn::ManagedRawImage retImage(ImageDesc(_desc.format, _desc.width, _desc.height, _desc.depth, _desc.channels), pbo, image.size());
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            return retImage;
        }
    } else {
#ifdef __ANDROID__
        if (_desc.target == GL_TEXTURE_2D) {
            auto& cf = getColorFormatDesc(_desc.format);
            // SNN_LOGD("INPUT TEXTURE 2D FORMAT is: %s", cf.name);
            GLuint _frameBuffer = 0;
            GLCHKDBG(glGenFramebuffers(1, &_frameBuffer));
            GLCHKDBG(glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer));
            GLCHKDBG(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _desc.target, _desc.id, 0));
            GLCHKDBG(glReadBuffer(GL_COLOR_ATTACHMENT0));
            if (cf.glType == GL_FLOAT || cf.glType == GL_HALF_FLOAT) {
                if (_desc.format == ColorFormat::RGBA16F) {
                    std::vector<uint8_t> tempBuffer(_desc.width * _desc.height * 8);
                    std::vector<uint8_t> floatBuffer(_desc.width * _desc.height * 16);
                    GLCHKDBG(glReadPixels(0, 0, _desc.width, _desc.height, GL_RGBA, GL_HALF_FLOAT, tempBuffer.data()));
                    for (std::size_t i = 0; i < tempBuffer.size(); i += 2) {
                        uint16_t val;
                        std::memcpy(&val, &tempBuffer.at(i), 2);
                        float flt = snn::convertToHighPrecision(val);
                        std::memcpy(&floatBuffer.at(2 * i), &flt, 4);
                    }
                    snn::ManagedRawImage retImage(ImageDesc(ColorFormat::RGBA32F, _desc.width, _desc.height, _desc.depth, _desc.channels), floatBuffer.data(),
                                                  floatBuffer.size());
                    return retImage;
                } else {
                    GLCHKDBG(glReadPixels(0, 0, _desc.width, _desc.height, GL_RGBA, GL_FLOAT, image.data()));
                }
            } else {
                GLCHKDBG(glReadPixels(0, 0, _desc.width, _desc.height, GL_RGBA, GL_UNSIGNED_BYTE, image.data()));
            }
            //GLCHK(;);
        } else {
            auto& cf            = getColorFormatDesc(_desc.format);
            std::size_t bitSize = (std::size_t)(cf.bits / (cf.ch * 8));
            glBindTexture(_desc.target, _desc.id);
            std::vector<uint8_t> dataBuffer;
            GLint maxFBOLayers, maxFBOSamples;
            glGetIntegerv(GL_MAX_FRAMEBUFFER_LAYERS, &maxFBOLayers);
            glGetIntegerv(GL_MAX_FRAMEBUFFER_SAMPLES, &maxFBOSamples);
            SNN_LOGD("Max FBO Layer is %d", maxFBOLayers);
            SNN_LOGD("Max FBO Sample count is %d", maxFBOSamples);
            SNN_ASSERT(_desc.depth < maxFBOLayers);
            SNN_ASSERT(_desc.width);
            SNN_ASSERT(_desc.height);
            uint32_t planeSize = _desc.width * _desc.height * 4 * bitSize;
            for (std::size_t i = 0; i < _desc.depth; i++) {
                std::size_t offsetStart = planeSize * i;
                std::size_t offsetEnd   = planeSize * (i + 1);
                // uint8_t* tempBuffer = (uint8_t*)malloc(planeSize * sizeof(uint8_t));
                std::vector<uint8_t> tempBuffer(planeSize);
                GLuint frameBuffer = 0;
                GLCHK(glGenFramebuffers(1, &frameBuffer));
                GLCHK(glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer));
                GLCHK(glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _desc.id, 0, (GLint) i));
                GLCHK(glReadBuffer(GL_COLOR_ATTACHMENT0));
                // GLint fboSamples;
                // glFrame
                auto completeNess = glCheckFramebufferStatus(GL_FRAMEBUFFER);
                switch (completeNess) {
                case GL_FRAMEBUFFER_COMPLETE:
                    break;

                case GL_FRAMEBUFFER_UNDEFINED:
                    SNN_RIP("Frame buffer does not exist");
                    break;

                case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                    SNN_RIP("Framebuffer attachment points are incomplete %d", i);
                    break;

                case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                    SNN_RIP("Framebuffer attachments are missing");
                    break;

                case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                    SNN_RIP("Issue with framebuffer read object type");
                    break;

                case GL_FRAMEBUFFER_UNSUPPORTED:
                    SNN_RIP("Framebuffer combination unsupported");
                    break;

                case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
                    SNN_RIP("Issue with layer targets");
                    break;

                default:
                    SNN_RIP("Something else wit framebuffer");
                    break;
                }
                if (cf.glType == GL_FLOAT || cf.glType == GL_HALF_FLOAT) {
                    if (_desc.format == ColorFormat::RGBA16F) {
                        GLCHK(glReadPixels(0, 0, _desc.width, _desc.height, GL_RGBA, GL_HALF_FLOAT, tempBuffer.data()));
                        // SNN_LOGD("Pixel Dump complete, converting to high precision");
                        std::vector<uint8_t> floatBuffer(_desc.width * _desc.height * 4 * 4);
                        for (std::size_t i = 0; i < planeSize; i += bitSize) {
                            uint16_t val;
                            std::memcpy(&val, &tempBuffer.at(i), bitSize);
                            float flt = snn::convertToHighPrecision(val);
                            std::memcpy(&floatBuffer.at(2 * i), &flt, 4);
                        }
                        // SNN_LOGD("Completed conversion to high precision");
                        dataBuffer.insert(dataBuffer.end(), floatBuffer.begin(), floatBuffer.end());
                    } else {
                        glReadPixels(0, 0, _desc.width, _desc.height, GL_RGBA, GL_FLOAT, tempBuffer.data());
                        dataBuffer.insert(dataBuffer.end(), tempBuffer.begin(), tempBuffer.end());
                    }
                } else {
                    GLCHK(glReadPixels(0, 0, _desc.width, _desc.height, GL_RGBA, GL_UNSIGNED_BYTE, tempBuffer.data()));
                    dataBuffer.insert(dataBuffer.end(), tempBuffer.begin(), tempBuffer.end());
                }
                glDeleteFramebuffers(1, &frameBuffer);
            }
            if (cf.glType == GL_FLOAT || cf.glType == GL_HALF_FLOAT) {
                image = snn::ManagedRawImage(ImageDesc(snn::ColorFormat::RGBA32F, _desc.width, _desc.height, _desc.depth, _desc.channels), dataBuffer.data(),
                                             dataBuffer.size());
            } else {
                image =
                    snn::ManagedRawImage(ImageDesc(_desc.format, _desc.width, _desc.height, _desc.depth, _desc.channels), dataBuffer.data(), dataBuffer.size());
            }
        }
#else
        auto& cf = getColorFormatDesc(_desc.format);
        glPixelStorei(GL_UNPACK_ALIGNMENT, image.alignment());
        CHECK_GL_ERROR("glPixelStorei");
        glPixelStorei(GL_UNPACK_ROW_LENGTH, (int) image.pitch() * 8 / (int) cf.bits);
        CHECK_GL_ERROR("glPixelStorei");
        glBindTexture(_desc.target, _desc.id);
        CHECK_GL_ERROR("glBindTexture");
        glGetTexImage(_desc.target, 0, cf.glFormat, cf.glType, image.data());

        CHECK_GL_ERROR("glGetTexImage");
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        //GLCHK(;);
        CHECK_GL_ERROR("glPixelStorei");

#endif
        return image;
    }
}

// -----------------------------------------------------------------------------
//
static void SaveImageToPNG(const float* pixels, uint32_t w, uint32_t h, uint32_t channels, const std::string& filepath) {
    // convert data to RGB8
    size_t numPixels = w * h;
    std::vector<uint8_t> rgb8(numPixels * 3);
    for (size_t i = 0; i < numPixels; ++i) {
        uint8_t* d     = &rgb8[i * 3];
        const float* s = &pixels[i * channels];
        for (size_t c = 0; c < 3; ++c) {
            if (c < channels) {
                auto f = s[c] * 255.0f;
                if (f < 0.f) {
                    d[c] = 0;
                } else if (f > 255.f) {
                    d[c] = 255;
                } else {
                    d[c] = (uint8_t) f;
                }
            } else {
                d[c] = 0;
            }
        }
    }
    SNN_LOGI("Save texture content to %s", filepath.c_str());
    stbi_write_png(filepath.c_str(), (int) w, (int) h, 3, rgb8.data(), 0);
}

// -----------------------------------------------------------------------------
//
static void saveTextureToFile(uint32_t w, uint32_t h, uint32_t channels, GLenum target, GLenum format, GLenum type, const std::string& filepath) {
    SNN_LOGI("Save texture content to %s", filepath.c_str());
    std::vector<float> pixels(w * h * channels);
    GLCHK(glGetTexImage(target, 0, format, type, pixels.data()));

    // Flip the image vertically, since OpenGL texture is bottom-up.
    std::vector<float> flipped(w * h * channels);
    for (size_t i = 0; i < h; ++i) {
        const float* s = &pixels[w * channels * (h - i - 1)];
        float* d       = &flipped[w * channels * i];
        memcpy(d, s, w * channels * sizeof(float));
    }

    std::ofstream file(filepath, std::ofstream::binary);
    if (!file.good()) {
        SNN_LOGE("Failed to open file %s for writing", filepath.c_str());
        return;
    }
    struct FileHeader {
        char FILE_TAG[8];
        uint32_t w, h, channels;
        GLenum type;
    } header = {{'F', 'T', 'L', 'I', 'M', 'A', 'G', 'E'}, w, h, channels, type};
    file.write((const char*) &header, sizeof(header));
    file.write((const char*) flipped.data(), flipped.size() * sizeof(float));
    file.close();

    // Also save to png file just for easy previewing.
    SaveImageToPNG(flipped.data(), w, h, channels, filepath + ".png");
}

void gl::DebugSSBO::printLastResult() const {
#if DEBUG_SSBO_ENABLED
    if (!counter) {
        return;
    }
    auto count    = std::min<size_t>((*counter), buffer.size() - 1);
    auto dataSize = sizeof(float) * (count + 1);
    if (0 != memcmp(buffer.data(), printed.data(), dataSize)) {
        memcpy(printed.data(), buffer.data(), dataSize);
        std::stringstream ss;
        ss << "count = " << *counter << " [";
        for (size_t i = 0; i < count; ++i) {
            auto value = printed[i + 1];
            if (std::isnan(value)) {
                ss << std::endl;
            } else {
                ss << value << ", ";
            }
        }
        ss << "]";
        SNN_LOGI("%s", ss.str().c_str());
    }
#endif
}

// -----------------------------------------------------------------------------
//
void gl::FullScreenQuad::allocate() {
    const glm::vec4 vertices[] = {{-1, -1, 0, 1}, {3, -1, 0, 1}, {-1, 3, 0, 1}};

    // Cleanup previous array if any.
    cleanup();

    // Create new array.
    GLCHK(glGenVertexArrays(1, &va));
    GLCHK(glBindVertexArray(va));
    vb.allocate(sizeof(vertices), vertices);
    GLCHK(vb.bind());
    GLCHK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (const void*) 0));
    GLCHK(glEnableVertexAttribArray(0));
    GLCHK(glBindVertexArray(0)); // unbind
}

// -----------------------------------------------------------------------------
//
void gl::FullScreenQuad::cleanup() {
    vb.cleanup();

    // If we actually have a vertex array to cleanup.
    if (va) {
        // Delete the vertex array.
        glDeleteVertexArrays(1, &va);

        // Reset this to mark it as cleaned up.
        va = 0;
    }
    // GLCHK(;);
}

// -----------------------------------------------------------------------------
//
static const char* shaderType2String(GLenum shaderType) {
    switch (shaderType) {
    case GL_VERTEX_SHADER:
        return "vertex";
    case GL_FRAGMENT_SHADER:
        return "fragment";
    case GL_COMPUTE_SHADER:
        return "compute";
    default:
        return "";
    }
}

// -----------------------------------------------------------------------------
static std::string addLineCount(const std::string& in) {
    std::stringstream ss;
    ss << "(  1) : ";
    int line = 1;
    for (auto ch : in) {
        if ('\n' == ch) {
            ss << formatString("\n(%3d) : ", ++line);
        } else {
            ss << ch;
        }
    }
    return ss.str();
}

// -----------------------------------------------------------------------------
//
GLuint gl::loadShaderFromString(const char* source, size_t length, GLenum shaderType, const char* optionalFilename) {
    if (!source) {
        return 0;
    }
    const char* sources[] = {source};
    if (0 == length) {
        length = strlen(source);
    }
    GLint sizes[] = {(GLint) length};
    auto shader   = glCreateShader(shaderType);
    glShaderSource(shader, 1, sources, sizes);
    glCompileShader(shader);
    // check for shader compile errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[4096 * 16];
        glGetShaderInfoLog(shader, 4096 * 16, NULL, infoLog);
        glDeleteShader(shader);
        SNN_LOGE("\n================== Failed to compile %s shader '%s' ====================\n"
                 "%s\n"
                 "\n============================= GLSL shader source ===============================\n"
                 "%s\n"
                 "\n================================================================================\n",
                 shaderType2String(shaderType), optionalFilename ? optionalFilename : "<no-name>", infoLog, addLineCount(source).c_str());
        // std::cout << std::endl << "================== Failed to compile "<< shaderType2String(shaderType);
        // std::cout <<" shader '" << (optionalFilename ? optionalFilename : "<no-name>") << "' ====================";
        // std::cout << std::endl << infoLog << std::endl << "============================= GLSL shader source ===============================";
        // std::cout << addLineCount(source).c_str() << std::endl;
        // std::cout << "================================================================================" << std::endl;
        SNN_LOGE("");
        return 0;
    }
    // done
    SNN_ASSERT(shader);
    return shader;
}

// -----------------------------------------------------------------------------
//
GLuint gl::linkProgram(const std::vector<GLuint>& shaders, const char* optionalProgramName) {
    auto program = glCreateProgram();
    for (auto s : shaders) {
        if (s) {
            glAttachShader(program, s);
        }
    }
    glLinkProgram(program);
    for (auto s : shaders) {
        if (s) {
            glDetachShader(program, s);
        }
    }
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        glDeleteProgram(program);
        SNN_LOGE("Failed to link program %s:\n%s", optionalProgramName ? optionalProgramName : "", infoLog);
        return 0;
    }

    // Enable the following code to dump GL program binary to disk.
#if 0
    if (optionalProgramName) {
        std::string outfilename = std::string(optionalProgramName) + ".bin";
        std::ofstream fs;
        fs.open(outfilename);
        if (fs.good()) {
            std::vector<uint8_t> buffer(1024 * 1024 * 1024); // allocate 1MB buffer.
            GLsizei len;
            GLenum dummyFormat;
            GLCHK(glGetProgramBinary(program, (GLsizei)buffer.size(), &len, &dummyFormat, buffer.data()));
            fs.write((const char*)buffer.data(), len);
        }
    }
#endif

    // done
    SNN_ASSERT(program);
    return program;
}

// -----------------------------------------------------------------------------
//
void gl::GpuTimeElapsedQuery::stop() {
    if (_q.running()) {
        // SNN_LOGI("%s still running: %d", name.c_str(), _q.status);
        _q.end();
        // SNN_LOGI("%s end running: %d", name.c_str(), _q.status);
    }
}

void gl::GpuTimeElapsedQuery::getTime() {
    if (_q.running()) {
        return;
    }
    _q.getResult(_result);
}

// -----------------------------------------------------------------------------
//
std::string gl::GpuTimeElapsedQuery::print() const { return formatString("%s : %s"), name.c_str(), ns2s(duration()).c_str(); }

// -----------------------------------------------------------------------------
//
std::string gl::GpuTimestamps::print(const char* ident) const {
    if (_marks.size() < 2) {
        return {};
    }
    std::stringstream ss;
    GLuint64 startTime = _marks[0].result;
    GLuint64 prevTime  = startTime;
    if (!ident) {
        ident = "";
    }
    if (0 == startTime) {
        ss << ident << "all timestamp queries are pending...\n";
    } else {
        auto getDuration = [](uint64_t a, uint64_t b) { return b >= a ? ns2s(b - a) : "  <n/a>"; };

        size_t maxlen = 0;
        for (size_t i = 1; i < _marks.size(); ++i) {
            maxlen = std::max(_marks[i].name.size(), maxlen);
        }
        for (size_t i = 1; i < _marks.size(); ++i) {
            auto current = _marks[i].result;
            if (0 == current) {
                ss << ident << "pending...\n";
                break;
            }
            // auto fromStart = current > startTime ? (current - startTime) : 0;
            auto delta = getDuration(prevTime, current);
            ss << ident << std::setw(maxlen) << std::left << _marks[i].name << std::setw(0) << " : " << delta << std::endl;
            prevTime = current;
        }
        ss << ident << "total = " << getDuration(_marks.front().result, _marks.back().result) << std::endl;
    }
    return ss.str();
}

// -----------------------------------------------------------------------------
//
#if 0
// This code path creates shared GL context using native Win32 API w/o using any 3rd party libraries.
// It is not currenty being used, but kept as reference.
    #include <windows.h>
class gl::RenderContext::Impl
{
    HWND _window = 0;
    HDC _dc = 0;
    HGLRC _rc = 0;

public:
    ~Impl()
    {
        destroy();
    }

    bool create(void *)
    {
        destroy();

        auto currentRC = wglGetCurrentContext();
        auto currentDC = wglGetCurrentDC();
        auto currentPF = GetPixelFormat(currentDC);
        PIXELFORMATDESCRIPTOR currentPfd;
        currentPfd.nSize = sizeof(currentPfd);
        if (0 == DescribePixelFormat(currentDC, currentPF, sizeof(currentPfd), &currentPfd)) {
            SNN_LOGE("Failed to get current PFD");
            return false;
        }

        // get class name
        auto className = "shared context class";

        WNDCLASSA wc = {};
        wc.lpfnWndProc    = (WNDPROC)&DefWindowProc;
        wc.cbClsExtra     = 0;
        wc.cbWndExtra     = 0;
        wc.hInstance      = (HINSTANCE)GetModuleHandleW(nullptr);
        wc.hIcon          = LoadIcon(0, IDI_APPLICATION);
        wc.hCursor        = LoadCursor(0, IDC_ARROW);
        wc.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
        wc.lpszMenuName   = 0;
        wc.lpszClassName  = className;
        wc.hIcon          = LoadIcon(0, IDI_APPLICATION);
        RegisterClassA(&wc);
        _window = CreateWindowA(className, "shared context window", 0, CW_USEDEFAULT, CW_USEDEFAULT, 1, 1, nullptr, 0, wc.hInstance, 0);
        _dc = GetDC(_window);
        if (!SetPixelFormat(_dc, currentPF, &currentPfd)) {
            SNN_LOGE("SetPixelFormat failed!");
            return false;
        }
        _rc = wglCreateContext(_dc);
        if (!_rc) {
            SNN_LOGE("wglCreateContext failed!");
            return false;
        }
        if (!wglShareLists(currentRC, _rc)) {
            SNN_LOGE("wglShareLists failed!");
            return false;
        }
        return true;
    }

    void makeCurrent()
    {
        if (!_rc) {
            SNN_LOGE("shared GL context is not properly initialized.");
            return;
        }

        if (!wglMakeCurrent(_dc, _rc)) {
            SNN_LOGE("wglMakeCurrent() failed.");
        }
    }

private:
    void destroy()
    {
        if (_rc) {
            wglDeleteContext(_rc), _rc = 0;
        }
        if (_dc) {
            ::ReleaseDC(_window, _dc), _dc = 0;
        }
        if (_window) {
            ::DestroyWindow(_window), _window = 0;
        }
    }
};
#elif defined(__ANDROID__) || defined(__linux__)
    #include <EGL/egl.h>
    #include <EGL/eglext.h>
const char* eglError2String(EGLint err) {
    switch (err) {
    case EGL_SUCCESS:
        return "The last function succeeded without error.";
    case EGL_NOT_INITIALIZED:
        return "EGL is not initialized, or could not be initialized, for the specified EGL display connection.";
    case EGL_BAD_ACCESS:
        return "EGL cannot access a requested resource (for example a context is bound in another thread).";
    case EGL_BAD_ALLOC:
        return "EGL failed to allocate resources for the requested operation.";
    case EGL_BAD_ATTRIBUTE:
        return "An unrecognized attribute or attribute value was passed in the attribute list.";
    case EGL_BAD_CONTEXT:
        return "An EGLContext argument does not name a valid EGL rendering context.";
    case EGL_BAD_CONFIG:
        return "An EGLConfig argument does not name a valid EGL frame buffer configuration.";
    case EGL_BAD_CURRENT_SURFACE:
        return "The current surface of the calling thread is a window, pixel buffer or pixmap that is no longer valid.";
    case EGL_BAD_DISPLAY:
        return "An EGLDisplay argument does not name a valid EGL display connection.";
    case EGL_BAD_SURFACE:
        return "An EGLSurface argument does not name a valid surface (window, pixel buffer or pixmap) configured for GL rendering.";
    case EGL_BAD_MATCH:
        return "Arguments are inconsistent (for example, a valid context requires buffers not supplied by a valid surface).";
    case EGL_BAD_PARAMETER:
        return "One or more argument values are invalid.";
    case EGL_BAD_NATIVE_PIXMAP:
        return "A NativePixmapType argument does not refer to a valid native pixmap.";
    case EGL_BAD_NATIVE_WINDOW:
        return "A NativeWindowType argument does not refer to a valid native window.";
    case EGL_CONTEXT_LOST:
        return "A power management event has occurred. The application must destroy all contexts and reinitialise OpenGL ES state and objects to continue"
        "rendering.";
    default:
        return "unknown error";
    }
}
    #define EGLCHK_R(x, returnValueWhenFailed)                                                                                                                 \
        if (!(x)) {                                                                                                                                            \
            SNN_LOGE(#x " failed: %s", eglError2String(eglGetError()));                                                                                        \
            return (returnValueWhenFailed);                                                                                                                    \
        } else                                                                                                                                                 \
            void(0)
    #define EGLCHK(x)                                                                                                                                          \
        if (!(x)) {                                                                                                                                            \
            SNN_RIP(#x " failed: %s", eglError2String(eglGetError()));                                                                                         \
        } else                                                                                                                                                 \
            void(0)
class gl::RenderContext::Impl {
public:
    Impl(gl::RenderContext::WindowHandle window, bool shared): _window(NativeWindowType(window)) {
        if (shared) {
            initSharedContext();
        } else {
            initStandaloneContext();
        }
    }

    ~Impl() { destroy(); }

    void makeCurrent() {
        if (!eglMakeCurrent(_disp, _surf, _surf, _rc)) {
            SNN_LOGE("Failed to set current EGL context.");
        }
    }

    void swapBuffers() {
        if (!eglSwapBuffers(_disp, _surf)) {
            int error = eglGetError();
            SNN_LOGE("Post record render swap fail. ERROR: %x", error);
        }
    }

private:
    // The context represented by this object.
    bool _new_disp = false;
    EGLDisplay _disp = 0;
    EGLContext _rc = 0;
    EGLSurface _surf = 0;
    NativeWindowType _window = (NativeWindowType) nullptr;

    void initSharedContext() {
        _disp = eglGetCurrentDisplay();
        auto currentRC = eglGetCurrentContext();
        if (!_disp || !currentRC) {
            SNN_RIP("no current display and/or EGL context found.");
        }

        auto currentConfig = getCurrentConfig(_disp, currentRC);
        if (!currentConfig) {
            SNN_RIP("failed to get EGL config.");
        }

        if (_window) {
            SNN_CHK(_surf = eglCreateWindowSurface(_disp, getCurrentConfig(_disp, currentRC), _window, nullptr));
        } else {
            EGLint pbufferAttribs[] = {
                EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE,
            };
            SNN_CHK(_surf = eglCreatePbufferSurface(_disp, currentConfig, pbufferAttribs));
        }

        // create context
        EGLint contextAttribs[] = {
            EGL_CONTEXT_CLIENT_VERSION,
            3,
    #ifdef _DEBUG
            EGL_CONTEXT_FLAGS_KHR,
            EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR,
    #endif
            EGL_NONE,
        };
        SNN_CHK(_rc = eglCreateContext(_disp, currentConfig, currentRC, contextAttribs));
    }

    void initStandaloneContext() {
        _new_disp = true;
        _disp = findBestHardwareDisplay();
        if (0 == _disp) {
            EGLCHK(_disp = eglGetDisplay(EGL_DEFAULT_DISPLAY));
        }
        EGLCHK(eglInitialize(_disp, nullptr, nullptr));
        const EGLint configAttribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 8, EGL_NONE};
        EGLint numConfigs;
        EGLConfig config;
        EGLCHK(eglChooseConfig(_disp, configAttribs, &config, 1, &numConfigs));
        SNN_CHK(numConfigs > 0);
        EGLint pbufferAttribs[] = {
            EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE,
        };
        EGLCHK(_surf = eglCreatePbufferSurface(_disp, config, pbufferAttribs));
        SNN_CHK(_surf);
    #ifdef __ANDROID__
        EGLCHK(eglBindAPI(EGL_OPENGL_ES_API));
    #else
        EGLCHK(eglBindAPI(EGL_OPENGL_API));
    #endif
        EGLint contextAttribs[] = {
            EGL_CONTEXT_CLIENT_VERSION,
            3,
    #ifdef _DEBUG
            EGL_CONTEXT_FLAGS_KHR,
            EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR,
    #endif
            EGL_NONE,
        };
        SNN_CHK(_rc = eglCreateContext(_disp, config, 0, contextAttribs));
    }

    void destroy() {
        if (_surf) {
            eglDestroySurface(_disp, _surf), _surf = 0;
        }
        if (_rc) {
            eglDestroyContext(_disp, _rc), _rc = 0;
        }
        if (_new_disp) {
            eglTerminate(_disp), _new_disp = false;
        }
        _disp = 0;
    }

    static EGLConfig getCurrentConfig(EGLDisplay d, EGLContext c) {
        EGLint currentConfigID = 0;
        EGLCHK_R(eglQueryContext(d, c, EGL_CONFIG_ID, &currentConfigID), 0);
        EGLint numConfigs;
        EGLCHK_R(eglGetConfigs(d, nullptr, 0, &numConfigs), 0);
        std::vector<EGLConfig> configs(numConfigs);
        EGLCHK_R(eglGetConfigs(d, configs.data(), numConfigs, &numConfigs), 0);
        for (auto config : configs) {
            EGLint id;
            eglGetConfigAttrib(d, config, EGL_CONFIG_ID, &id);
            if (id == currentConfigID) {
                return config;
            }
        }
        SNN_LOGE("Couldn't find current EGL config.");
        return 0;
    }

    // Return the display that represents the best GPU hardware available on current system.
    static EGLDisplay findBestHardwareDisplay() {
        // query required extension
        auto eglQueryDevicesEXT = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
        auto eglGetPlatformDisplayExt = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (!eglQueryDevicesEXT || !eglGetPlatformDisplayExt) {
            SNN_LOGE("Required EGL extension(s) are missing.");
            return 0;
        }

        EGLDeviceEXT devices[32];
        EGLint num_devices;
        EGLCHK_R(eglQueryDevicesEXT(32, devices, &num_devices), 0);
        if (num_devices == 0) {
            SNN_LOGE("No EGL devices found.");
            return 0;
        }
        SNN_LOGI("Total %d EGL devices found.", num_devices);

        // try find the NVIDIA device
        EGLDisplay nvidia = 0;
        for (int i = 0; i < num_devices; ++i) {
            auto display = eglGetPlatformDisplayExt(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
            EGLint major, minor;
            eglInitialize(display, &major, &minor);
            auto vendor = eglQueryString(display, EGL_VENDOR);
            if (vendor && 0 == strcmp(vendor, "NVIDIA")) {
                nvidia = display;
            }
            eglTerminate(display);
        }

        return nvidia;
    }
};
#else
    #include <GLFW/glfw3.h>
class gl::RenderContext::Impl {
public:
    Impl(gl::RenderContext::WindowHandle, bool shared) {
        GLFWwindow* current = nullptr;
        if (shared) {
            current = glfwGetCurrentContext();
            if (!current) {
                SNN_RIP("No current GLFW window found.");
                return;
            }
        } else {
            glfwInit();
        }
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        _window = glfwCreateWindow(1, 1, "", nullptr, current);
        if (!_window) {
            SNN_RIP("Failed to create shared GLFW window.");
        }
    }

    virtual ~Impl() {
        if (_window) {
            glfwDestroyWindow(_window), _window = nullptr;
        }
    }

    void makeCurrent() {
        if (_window) {
            glfwMakeContextCurrent(_window);
        }
        else {
            SNN_LOGE("shared GL context wasn't properly initlaized.");
        }
    }

    void swapBuffers() { glfwSwapBuffers(_window); }

private:
    GLFWwindow* _window = nullptr;
};
#endif

gl::RenderContext::RenderContext(Type t, WindowHandle w) {
    // store current context
    RenderContextStack rcs;
    rcs.push();

    _impl = new Impl(w, t == SHARED);
    makeCurrent();
    gl::initGLExtensions();

    // switch back to previous context
    rcs.pop();
}
gl::RenderContext::~RenderContext() {
    delete _impl;
    _impl = nullptr;
}
gl::RenderContext& gl::RenderContext::operator=(RenderContext&& that) {
    if (this != &that) {
        delete _impl;
        _impl      = that._impl;
        that._impl = nullptr;
    }
    return *this;
}
void gl::RenderContext::makeCurrent() {
    if (_impl) {
        _impl->makeCurrent();
    }
}
void gl::RenderContext::swapBuffers() {
    if (_impl) {
        _impl->swapBuffers();
    }
}

class gl::RenderContextStack::Impl {
    struct OpenGLRC {
#if defined(__ANDROID__) || defined(__linux__)
        EGLDisplay display;
        EGLSurface drawSurface;
        EGLSurface readSurface;
        EGLContext context;

        void store() {
            display     = eglGetCurrentDisplay();
            drawSurface = eglGetCurrentSurface(EGL_DRAW);
            readSurface = eglGetCurrentSurface(EGL_READ);
            context     = eglGetCurrentContext();
        }

        void restore() const {
            if (display && context) {
                if (!eglMakeCurrent(display, drawSurface, readSurface, context)) {
                    EGLint error = eglGetError();
                    SNN_LOGE("Failed to restore EGL context. ERROR: %x", error);
                }
            }
        }
#else
        GLFWwindow* window;

        void store() { window = glfwGetCurrentContext(); }
        void restore() { glfwMakeContextCurrent(window); }
#endif
    };

    std::stack<OpenGLRC> _stack;

public:
    ~Impl() {
        while (_stack.size() > 1) {
            _stack.pop();
        }
        if (1 == _stack.size()) {
            pop();
        }
        SNN_ASSERT(_stack.empty());
    }

    void push() {
        _stack.push({});
        _stack.top().store();
    }

    void apply() {
        if (!_stack.empty()) {
            _stack.top().restore();
        }
    }

    void pop() {
        if (!_stack.empty()) {
            _stack.top().restore();
            _stack.pop();
        }
    }
};
gl::RenderContextStack::RenderContextStack(): _impl(new Impl()) {}
gl::RenderContextStack::~RenderContextStack() { delete _impl; }
void gl::RenderContextStack::push() { _impl->push(); }
void gl::RenderContextStack::apply() { _impl->apply(); }
void gl::RenderContextStack::pop() { _impl->pop(); }

void* gl::GLSSBOBuffer::map(GLbitfield bufMask) {
    glBindBuffer(mType, mId);
    OPENGL_CHECK_ERROR;
    auto ptr = glMapBufferRange(mType, 0, mSize, bufMask);
    OPENGL_CHECK_ERROR;
    return ptr;
}

void gl::GLSSBOBuffer::unmap() {
    glBindBuffer(mType, mId);
    glUnmapBuffer(mType);
    OPENGL_CHECK_ERROR;
}
