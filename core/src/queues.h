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
#pragma once
#include <array>
#include <vector>
#include <chrono>
#include <optional>
#include "snn/utils.h"
#include "readerwriterqueue.h"

namespace snn {
namespace queue {
// foundational queues
template<class Value>
class ReaderWriter : moodycamel::ReaderWriterQueue<Value> {
public:
    using Base      = moodycamel::ReaderWriterQueue<Value>;
    using ValueType = Value;
    using Base::Base;
    using Base::size_approx;

    template<class QueueOp, class... Args>
    auto operator()(QueueOp op, Args&&... vs) -> auto {
        return op(static_cast<Base&>(*this), std::forward<Args>(vs)...);
    }
};
template<class Value>
class BlockingReaderWriter : moodycamel::BlockingReaderWriterQueue<Value> {
public:
    using Base      = moodycamel::BlockingReaderWriterQueue<Value>;
    using ValueType = Value;
    using Base::Base;
    using Base::size_approx;

    template<class QueueOp, class... Args>
    auto operator()(QueueOp op, Args&&... vs) -> auto {
        return op(static_cast<Base&>(*this), std::forward<Args>(vs)...);
    }
};

namespace detail {
template<class Queue, class = void>
struct Value;

template<class Queue>
struct Value<Queue, std::void_t<typename Queue::ValueType>> {
    using Type = typename Queue::ValueType;
};
template<class T>
struct Value<moodycamel::ReaderWriterQueue<T>, void> {
    using Type = T;
};
template<class T>
struct Value<moodycamel::BlockingReaderWriterQueue<T>, void> {
    using Type = T;
};
} // namespace detail
template<class Queue>
using ValueType = typename detail::Value<std::decay_t<Queue>>::Type;

// Base classes for queue operations
// They are functionally just tags used for overload routing.
class Op {};
class ReadOp {
public:
    template<class Queue>
    auto operator()(Queue&, ValueType<Queue>&) const -> bool;
};
class WriteOp {
public:
    template<class Queue, class... Args>
    auto operator()(Queue&, Args&&...) const -> bool;
};

// Enqueues a if there is room in the queue.
// Returns true if the element was enqueued, false otherwise.
// Does not allocate memory.
struct TryEnqueue : WriteOp {
    template<class Queue, class Value>
    auto operator()(Queue& q, Value&& v) const -> bool {
        return q.try_enqueue(std::forward<Value>(v));
    }
};

// Like try_enqueue() but with emplace semantics (i.e. construct-in-place).
struct TryEmplace : WriteOp {
    template<class Queue, class... Args>
    auto operator()(Queue& q, Args&&... vs) const -> bool {
        return q.try_emplace(std::forward<Args>(vs)...);
    }
};

// Allocates an additional block of memory if needed.
// Only fails (returns false) if memory allocation fails.
struct Enqueue : WriteOp {
    template<class Queue, class Value>
    auto operator()(Queue& q, Value&& v) const -> bool {
        return q.enqueue(std::forward<Value>(v));
    }
};

// Like enqueue() but with emplace semantics (i.e. construct-in-place).
struct Emplace : WriteOp {
    template<class Queue, class... Args>
    auto operator()(Queue& q, Args&&... vs) const -> bool {
        return q.emplace(std::forward<Args>(vs)...);
    }
};

// Attempts to dequeue an element; if the queue is empty,
// returns false instead. If the queue has at least one element,
// moves front to result using operator=, then returns true.
struct TryDequeue : ReadOp {
    template<class Queue>
    auto operator()(Queue& q, ValueType<Queue>& v) const -> bool {
        return q.try_dequeue(v);
    }
};

// Attempts to dequeue an element; if the queue is empty,
// waits until an element is available, then dequeues it.
struct WaitDequeue : ReadOp {
    template<class Value>
    auto operator()(typename BlockingReaderWriter<Value>::Base& q, Value& v) const -> bool {
        q.wait_dequeue(v);
        return true;
    }
};

// Attempts to dequeue an element; if the queue is empty,
// waits until an element is available up to the specified timeout,
// then dequeues it and returns true, or returns false if the timeout
// expires before an element can be dequeued.
// Using a negative timeout indicates an indefinite timeout,
// and is thus functionally equivalent to calling wait_dequeue.
template<class Rep, class Period>
struct WaitDequeueTimed : ReadOp {
    const std::chrono::duration<Rep, Period> timeout;

    WaitDequeueTimed(const std::chrono::duration<Rep, Period>& timeout): timeout(timeout) {}

    template<class Value>
    auto operator()(typename BlockingReaderWriter<Value>::Base& q, Value& v) const -> bool {
        return q.wait_dequeue_timed(v, timeout);
    }
};

// Copies the front element in the queue (the one that
// would be removed next by a call to `try_dequeue` or `pop`). If the
// queue appears empty at the time the method is called, false is
// returned.
// Must be called only from the consumer thread.
struct Peek : ReadOp {
    template<class Queue>
    auto operator()(Queue& q, ValueType<Queue>& v) const -> bool {
        if (auto ptr = q.peek(); ptr != nullptr) {
            v = *ptr;
            return true;
        } else
            return false;
    }
};

static constexpr auto tryEnqueue  = TryEnqueue {};
static constexpr auto tryEmplace  = TryEmplace {};
static constexpr auto enqueue     = Enqueue {};
static constexpr auto emplace     = Emplace {};
static constexpr auto tryDequeue  = TryDequeue {};
static constexpr auto waitDequeue = WaitDequeue {};
template<class Rep, class Period>
static constexpr auto waitDequeueTimed(const std::chrono::duration<Rep, Period>& timeout) -> WaitDequeueTimed<Rep, Period> {
    return {timeout};
}
static constexpr auto peek = Peek {};

// base class for dequeuing adapters
class ReadAdapter {
    template<class Read, class Value>
    auto operator()(const Read& read, Value& v) -> bool;

    auto size_approx() const -> size_t;
};

// join two queues by a key computed from a key getter.
// it will drop an item from the front of its queue
// when its key is less than the front of the other queue's key
template<class QueueA, class QueueB, class Key = snn::Get<0>>
struct JoinedMonotonic : ReadAdapter {
    QueueA queueA;
    QueueB queueB;

    std::optional<queue::ValueType<QueueA>> cacheA;
    std::optional<queue::ValueType<QueueB>> cacheB;

    JoinedMonotonic(QueueA queueA, QueueB queueB): queueA(std::forward<QueueA>(queueA)), queueB(std::forward<QueueB>(queueB)) {}

    using ValueType = std::tuple<queue::ValueType<QueueA>, queue::ValueType<QueueB>>;

    template<class Read>
    auto operator()(const Read& read, ValueType& v) -> std::enable_if_t<std::is_base_of_v<ReadOp, Read>, bool> {
        const auto key = Key {};

        if (!cacheA) {
            cacheA.emplace();
            if (!queueA(read, *cacheA)) {
                cacheA = {};
            }
        }
        if (!cacheB) {
            cacheB.emplace();
            if (!queueB(read, *cacheB)) {
                cacheB = {};
            }
        }

        while (cacheA && cacheB) {
            if (key(*cacheA) < key(*cacheB)) {
                cacheA.emplace();
                if (!queueA(read, *cacheA)) {
                    cacheA = {};
                }
            } else if (key(*cacheB) < key(*cacheA)) {
                cacheB.emplace();
                if (!queueB(read, *cacheB)) {
                    cacheB = {};
                }
            } else {
                std::get<0>(v) = std::move(*cacheA);
                std::get<1>(v) = std::move(*cacheB);
                cacheA         = {};
                cacheB         = {};
                return true;
            }
        }

        return false;
    }

    auto size_approx() const -> size_t { return std::min(queueA.size_approx(), queueB.size_approx()); }
};
template<class QueueA, class QueueB>
static constexpr auto joinedMonotonic(QueueA&& a, QueueB&& b) -> JoinedMonotonic<QueueA, QueueB> {
    return {std::forward<QueueA>(a), std::forward<QueueB>(b)};
}

// accumulate n queue items in a buffer
// adapter is empty until n items are accumulated
// after which, each subsequent item pushed to the back
// will pop the front of the buffer
// as if one is sliding a n-length window over a streaming queue
template<size_t n, class Queue>
struct SlidingWindow : ReadAdapter {
    Queue queue;

    SlidingWindow(Queue queue): queue(std::forward<Queue>(queue)) {}

    using ValueType = std::vector<std::reference_wrapper<queue::ValueType<Queue>>>;
    using StoreType = std::array<queue::ValueType<Queue>, n>;

    template<class Read>
    auto operator()(const Read& read, ValueType& v) -> std::enable_if_t<std::is_base_of_v<ReadOp, Read>, bool> {
        if (_fillIndex == n) {
            std::move(front.begin() + 1, front.end(), front.begin());
            --_fillIndex;
        }
        if (queue(read, front[_fillIndex])) {
            ++_fillIndex;
        } else
            return false;
        if (_fillIndex < n) {
            return false;
        }
        v.clear();
        v.reserve(n);
        std::copy(front.begin(), front.end(), std::back_inserter(v));
        return true;
    }

    auto size_approx() const -> size_t { throw std::runtime_error("unimplemented"); }

private:
    StoreType front;
    size_t _fillIndex = 0;
};
template<size_t n, class Queue>
static constexpr auto slidingWindow(Queue&& q) -> SlidingWindow<n, Queue> {
    return {std::forward<Queue>(q)};
}
} // namespace queue
} // namespace snn
