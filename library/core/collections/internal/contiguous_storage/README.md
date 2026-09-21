# Contiguous storage compatibility package

This package re-exports core.storage. The single ContiguousStorage implementation
lives there, below collections and above core.memory. New dependencies should
use core.storage directly.
