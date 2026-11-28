//! The host's record of a `forward::Runtime`'s device-side position and token, checked before every
//! launch. Host only (no CUDA), so `cargo test` covers it.
//!
//! The device advances its own position (`argmax_advance` increments it), so the host copy is only
//! trustworthy while every enqueued step has finished cleanly. A step that fails after its launches
//! began (a failed readback, a failed graph launch) may or may not have advanced the device, so the
//! cursor goes stale and refuses every step until the host sets the position and token again.
use anyhow::{ensure, Result};

#[derive(Debug, Clone)]
pub struct Cursor {
    capacity: usize,
    pos: usize,
    /// The next step's token was set by the host (and checked against the embedding rows), not
    /// written by the previous step's `argmax_advance`.
    host_token: bool,
    /// A step began and did not finish cleanly: the device position is unknown.
    stale: bool,
}

impl Cursor {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, pos: 0, host_token: false, stale: false }
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// The position the next step decodes at.
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn stale(&self) -> bool {
        self.stale
    }

    /// Checks `pos` and `token` before the host writes them to the device.
    pub fn check_set(&self, pos: usize, token: usize, embed_rows: usize) -> Result<()> {
        ensure!(pos < self.capacity, "position {pos} is past the cache capacity {}", self.capacity);
        ensure!(token < embed_rows, "token {token} is past the {embed_rows} embedding rows");
        Ok(())
    }
    /// The host wrote the position and token to the device.
    pub fn set(&mut self, pos: usize) {
        self.pos = pos;
        self.host_token = true;
        self.stale = false;
    }

    /// Checks a step can run: the last step finished, the position is within the cache, and, when
    /// the token comes from the previous step's `argmax_advance` (any vocabulary index), every token
    /// has an embedding row.
    pub fn check_step(&self, embed_rows: usize, vocab: usize) -> Result<()> {
        ensure!(!self.stale, "a previous step or prefill failed after it began, so the device position is unknown; set the position and token first");
        ensure!(self.pos < self.capacity, "position {} is past the cache capacity {}", self.pos, self.capacity);
        ensure!(self.host_token || embed_rows >= vocab,
            "a step after a step looks up a greedy token from the whole vocabulary, but only {embed_rows} of {vocab} embedding rows are on the GPU; set the token first");
        Ok(())
    }

    /// Launches that change the device position are about to begin.
    pub fn begin(&mut self) {
        self.stale = true;
    }
    /// They finished, leaving the device at `pos` with a token from `argmax_advance`.
    pub fn finish_at(&mut self, pos: usize) {
        self.pos = pos;
        self.host_token = false;
        self.stale = false;
    }
}
