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
#include "pch.h"
#include "snn/snn.h"
#include "snn/glUtils.h"
#include <deque>
#include <mutex>
#include <future>
#include <set>

using namespace snn;

class FenceManager::Impl {
    struct GpuSync {
        uint64_t index;
        GLsync sync;
    };

    std::deque<GpuSync> _gpuSyncs;
    std::mutex _gpuLock;
    uint64_t _gpuNext = 0;

    std::set<uint64_t> _cpuSyncs;
    std::mutex _cpuLock;
    uint64_t _cpuNext = 0;

public:
    Impl() = default;

    ~Impl() {}

    Fence insertGpuFence() {
        std::unique_lock<std::mutex> lock(_gpuLock);
        auto sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        SNN_ASSERT(sync);
        Fence fence;
        fence.type  = Device::GPU;
        fence.value = ++_gpuNext;
        _gpuSyncs.push_back({fence.value, sync});
        return fence;
    }

    bool isFencePending(Fence fence) {
        if (0 == fence) {
            return false;
        }
        if (Device::GPU == fence.type) {
            return isGpuFencePending(fence.value);
        } else {
            return isCpuFencePending(fence.value);
        }
    }

    Fence createCpuFence() {
        std::unique_lock<std::mutex> lock(_cpuLock);
        auto id = ++_cpuNext;
        _cpuSyncs.insert(id);
        Fence fence;
        fence.type  = Device::CPU;
        fence.value = id;
        return fence;
    }

    void signalCpuFence(Fence fence) {
        if (Device::CPU != fence.type) {
            SNN_LOGE("not a CPU fence.");
            return;
        }
        std::unique_lock<std::mutex> lock(_cpuLock);
        _cpuSyncs.erase(fence.value);
    }

    void waitForFence(Fence f) {
        if (0 == f) {
            return;
        }
        if (Device::GPU == f.type) {
            waitForGPUFence(f.value);
        } else {
            waitForCPUFence(f.value);
        }
        SNN_ASSERT(!isFencePending(f));
    }

private:
    bool isGpuFencePending(uint64_t index) {
        std::unique_lock<std::mutex> lock(_gpuLock);

        if (_gpuSyncs.empty()) {
            return false; // all fences have passed already.
        }

        const auto& theOldestFence = _gpuSyncs.front();
        if (index < theOldestFence.index) {
            return false; // The parameter fence is older then the oldest fence. So it must have passed.
        }

        size_t arrayOffset = index - theOldestFence.index;
        SNN_ASSERT(arrayOffset < _gpuSyncs.size());
        const auto& fence = _gpuSyncs[arrayOffset];

        SNN_ASSERT(glIsSync(fence.sync)); // make sure that this is a valid sync object.

        GLint status;
        GLCHK(glGetSynciv(fence.sync, GL_SYNC_STATUS, sizeof(GLint), nullptr, &status));
        if (GL_UNSIGNALED == status) {
            return true; // stil pending
        }

        // if a fence is passed, delete it along with all other GPU fences older than it.
        // TODO: is fence object reuseable?
        for (size_t i = 0; i <= arrayOffset; ++i) {
            glDeleteSync(_gpuSyncs[i].sync);
        }
        _gpuSyncs.erase(_gpuSyncs.begin(), _gpuSyncs.begin() + arrayOffset + 1);

        // done
        return false;
    }

    void waitForGPUFence(uint64_t index) {
        GLsync sync = nullptr;
        {
            std::unique_lock<std::mutex> lock(_gpuLock);

            if (_gpuSyncs.empty()) {
                return; // all fences have passed already.
            }
            const auto& theOldestFence = _gpuSyncs.front();
            if (index < theOldestFence.index) {
                return; // The parameter fence is older then the oldest fence. So it must have passed.
            }

            size_t arrayOffset = index - theOldestFence.index;
            SNN_ASSERT(arrayOffset < _gpuSyncs.size());
            const auto& fence = _gpuSyncs[arrayOffset];

            SNN_ASSERT(glIsSync(fence.sync));

            GLint status;
            GLCHK(glGetSynciv(fence.sync, GL_SYNC_STATUS, sizeof(GLint), nullptr, &status));
            if (GL_SIGNALED == status) {
                return; // not pending anymore.
            }

            sync = fence.sync;
        }

        // Block calling thread until the sync object is signaled or deleted.
        // Note this function is expected to trigger GL error, if sync object is deleted while wating.
        glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, GLuint64(-1));

        // clear our potential GL error.
        glGetError();
    }

    bool isCpuFencePending(uint64_t id) {
        std::unique_lock<std::mutex> lock(_cpuLock);
        return _cpuSyncs.end() != _cpuSyncs.find(id);
    }

    void waitForCPUFence(uint64_t id) {
        // TODO: wait on an event.
        while (isCpuFencePending(id)) {
            std::this_thread::sleep_for(std::chrono::microseconds(0));
        }
    }
};

// -----------------------------------------------------------------------------
//
FenceManager::FenceManager(): _impl(new Impl()) {}

// -----------------------------------------------------------------------------
//
FenceManager::~FenceManager() { delete _impl; }

// -----------------------------------------------------------------------------
//
std::shared_ptr<FenceManager> FenceManager::getInstance() {
    static std::weak_ptr<FenceManager> p;
    auto s = p.lock();
    if (!s) {
        s.reset(new FenceManager());
        p = s;
    }
    SNN_ASSERT(s);
    return s;
}

// -----------------------------------------------------------------------------
//
Fence FenceManager::insertGpuFence() { return _impl->insertGpuFence(); }

// -----------------------------------------------------------------------------
//
Fence FenceManager::createCpuFence() { return _impl->createCpuFence(); }

// -----------------------------------------------------------------------------
//
void FenceManager::signalCpuFence(Fence f) { _impl->signalCpuFence(f); }

// -----------------------------------------------------------------------------
//
bool FenceManager::isFencePending(Fence f) { return _impl->isFencePending(f); }

// -----------------------------------------------------------------------------
//
void FenceManager::waitForFence(Fence f) { _impl->waitForFence(f); }
