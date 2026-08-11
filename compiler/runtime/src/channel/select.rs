use super::{Receiver, TryRecvError};
use std::{thread, time::Duration};

pub enum SelectCase<'a, T> {
    Receive(&'a Receiver<T>),
    Default,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectResult<T> {
    Received { case_index: usize, value: T },
    Closed { case_index: usize },
    Default { case_index: usize },
}

/// Simple fair-ish polling select.
///
/// This is deliberately independent of channel internals. A later optimized
/// runtime can replace this with registered waiters while preserving the API.
pub fn select<T>(cases: &[SelectCase<'_, T>], poll_interval: Duration) -> SelectResult<T> {
    let mut start = 0usize;

    loop {
        let len = cases.len();

        for offset in 0..len {
            let index = (start + offset) % len;

            match &cases[index] {
                SelectCase::Receive(receiver) => match receiver.try_recv() {
                    Ok(value) => {
                        return SelectResult::Received {
                            case_index: index,
                            value,
                        };
                    }
                    Err(TryRecvError::Closed) => {
                        return SelectResult::Closed { case_index: index };
                    }
                    Err(TryRecvError::Empty) => {}
                },
                SelectCase::Default => {}
            }
        }

        if let Some(index) = cases
            .iter()
            .position(|case| matches!(case, SelectCase::Default))
        {
            return SelectResult::Default { case_index: index };
        }

        start = start.wrapping_add(1);
        thread::sleep(poll_interval);
    }
}
