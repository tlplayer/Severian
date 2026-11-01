# SIP-0035: Scoped concurrency and executors

Status: Draft
Type: Language | Runtime
Tags: async | await | ownership | completion | cancellation | virtual | thread | process

## Appendix

- task: one invocation with an owned completion result.
- owner: lifecycle responsible for joining every registered task.
- local: current function invocation; nested tasks complete before it exits.
- self: receiver lifecycle in a method; current invocation in a free function.
- module: module lifecycle; completion precedes module storage destruction.
- scope: explicit lifecycle owner for a request, connection, or batch.
- virtual: runtime-scheduled task; waiting on readiness suspends the task.
- thread: task executed on an OS thread sharing the process address space.
- process: isolated worker; inputs and outputs cross a checked IPC boundary.
- runtime: explicitly unsafe detached lifecycle; it is not an executor kind.
- join: observe completion and release execution resources.
- await: join and consume a task's value or propagate its error.
- cancellation: cooperative request, followed by acknowledged completion.
- completion: returned value, failure, or acknowledged cancellation; exactly one.
- deadline: monotonic expiration; distinct from a promise to finish cleanup by then.
- scope failure: primary failure plus secondary task/cleanup diagnostics.

## Goal(s)

```text
spawn -> register owner -> execute -> complete once -> join -> release captures
normal owner exit -> join all remaining children -> release owner storage
exceptional owner exit -> request cancellation -> join all children -> propagate
cancel(task) -> request; await(task) -> proof that borrowed storage is reusable
executor selection -> placement choice; ownership rules remain the same
```

## Examples, Problems, & Tests

### One ownership rule, three execution frameworks

```sev
def double(value: int) -> int:
    return value * 2

test "execution placement preserves the result":
    light = async double(21) with local, virtual
    native = async double(21) with local, thread
    isolated = async double(21) with local, process
    assert(await light == 42)
    assert(await native == 42)
    assert(await isolated == 42)

# `and` remains accepted for existing `with self and lock` source.
# Omitted owner -> local. Omitted executor -> runtime's configured default.
# An explicit executor must be honoured or rejected as Unsupported.
# `virtual` must not silently create one OS thread per task.
# `process` must not silently use a thread sharing the caller's heap.
```

```text
virtual -> scheduler + readiness reactor + bounded blocking-worker pool
thread  -> OS thread + join; bounded admission may wait or return Overloaded
process -> worker process + serialization + IPC + wait/reap

Executor configuration -> worker count, maximum queued work, admission policy
Queue full -> suspend with deadline or return Overloaded; never grow without bound
CPU-bound virtual work -> explicit yield/checkpoints; no fairness by assumption
Process worker death -> WorkerExited; surviving owner can still join other children
Process inputs -> transferable values, not pointers/loans/locks/open sockets
External commands -> process.spawn(argv), not async(command_string)
GPU/SIMD -> placement in SIP-0039; not interchangeable with these task executors
```

### Function scope completes even without explicit await

```sev
class Counter:
    value: int

    def increment():
        value += 1

def finish_two(counter: borrow Counter):
    async counter.increment() with local, thread, lock
    async counter.increment() with local, thread, lock
    # Local task registry retains both discarded handles until completion.

test "return is delayed until both increments complete":
    counter = Counter(0)
    finish_two(borrow counter)
    assert(counter.value == 2)

# A return expression is evaluated before cleanup/join. Read shared state only
# after an explicit await or after the owning function has returned.
# Dropping a handle does not detach a task or abandon its failure.
# Unit results may be discarded; unobserved child errors still fail owner exit.
```

```sev
def nested(value: int) -> int:
    child = async double(value) with local, thread
    return await child + 1

test "completion includes the whole descendant tree":
    parent = async nested(20) with local, thread
    assert(await parent == 41)

test with compiler "a task result has one consumer":
    task = async double(21) with local, thread
    assert(await task == 42)
    reject:
        again = await task

test with compiler "local task handles cannot escape their owner":
    reject:
        def escape():
            return async double(21) with local, thread
```

### Explicit owners and cancellation

```sev
import concurrency

def serve_request(request: Request) -> Reply | Error:
    with concurrency.scope() as request_work:
        auth = async authenticate(request.credentials) with request_work, virtual
        data = async load_data(request.key) with request_work, virtual
        return Reply(await auth, await data)
    # Scope completes before request-owned buffers can be released.

# Scope API contract:
# scope(deadline?) -> Scope
# scope.cancel() -> request cancellation of registered descendants
# task.cancel() -> request cancellation; safe to repeat
# task.completed() -> terminal-state observation; does not consume result
# await task -> consume result/error exactly once
# scope.join() -> finish registration, join all children, propagate failures
# joining/closed scope + spawn -> ScopeClosed, no task is started
# custom `with request` -> request must implement the ScopeOwner contract
# `self`/`module` may retain only captures living at least as long as that owner
```

```text
open -> cancelling -> joining -> closed
open -> joining -> closed                 # Normal completion does not cancel.
child fails -> cancel siblings immediately; parent sees failure at await/exit
body fails -> preserve body failure as primary; retain child failures as secondary
multiple child failures -> stable spawn-order diagnostics, not scheduler order
explicit await handled a child error -> owner must not report that same error again
cancellation races return -> first committed terminal state wins
cancel after completion -> no-op, retain result
cancel blocked accept/read/channel receive -> wake wait, finish native cancellation
cancel noncooperative foreign call -> keep captures alive and wait for actual return
scope deadline -> request cancellation; it cannot safely kill a running OS thread
process grace expires -> force termination, reap, then release IPC buffers
```

### Captures, synchronization, and process isolation

```sev
def first(values: view tensor[int]) -> int:
    return values[0]

test with compiler "live task loans exclude mutation":
    reject:
        values = tensor[int](1)
        child = async first(view values) with local, thread
        values[0] = 42
        await child
    accept:
        values = tensor[int](1)
        child = async first(view values) with local, thread
        await child
        values[0] = 42

test with compiler "process captures cannot transport a borrow":
    reject:
        values = tensor[int](1)
        child = async first(view values) with local, process

# Ordinary by-value arguments -> snapshot/copy unless their contract moves them.
# Explicit view -> loan retained through completion; no hidden copy of a view.
# Mutable capture -> exclusive loan or explicit shared synchronization.
# `lock` protects the captured receiver, not every task in the program.
# Parent cannot access locked shared state until its loan ends at await/join.
# Await/block while holding a receiver lock -> rejected when it can depend on
# another task requiring that lock; a lock is not a substitute for ownership.
```

### Channels and bounded work

```sev
test "bounded fan-in retains both results":
    messages = channel[int,2]
    first = async messages.send(20) with local, thread
    second = async messages.send(22) with local, thread
    assert(messages.receive() + messages.receive() == 42)
    await first
    await second

# channel[T,N] -> at most N queued elements; N is Number of elements, not bytes.
# send when full -> suspend with deadline/cancellation, never overwrite.
# close sender -> queued values drain, then receiver observes Closed.
# close receiver -> blocked senders wake with Closed; unsent value stays owned.
# virtual wait -> park task; thread wait -> condition/readiness wait.
# process channel -> explicit serialized IPC endpoint, not shared-memory channel.
# Select chooses one ready arm; losing registrations are removed before return.
```

### Implementation and acceptance boundary

```text
Current bootstrap -> pthread-backed tasks, default local cleanup, consuming await,
                     channels, snapshot captures, and legacy lock support.
Target contracts -> actual virtual scheduler, process task executor, explicit
                    ScopeOwner, cooperative cancellation, aggregate failures.
Do not label an OS-thread-backed implementation as virtual scheduling.
Do not label a subprocess command API as the process task executor.

Required tests (keep adjacent to the future runtime implementation):
  normal return / early return / throw -> each child joined once
  discarded task handle -> result cleaned and unobserved error propagated
  failed spawn -> no registered phantom task, captures released once
  failure while siblings block -> cancel, wake, join all siblings
  parent deadline -> child operations share original deadline
  moved capture -> parent access rejected, task releases it once
  borrowed capture -> parent drop rejected until await/join
  process worker crash -> typed failure, no zombie or orphaned pipe
  virtual socket wait -> another virtual task progresses on one worker
  saturated queue -> bounded storage and deterministic admission policy
```

The chat lab in `docs/labs/17-chat` uses a bounded readiness loop today. That
loop demonstrates multiplexed IO; it does not claim to implement the proposed
virtual task scheduler. Networking operations and framing are in SIP-0041.
