//! sparkforge: take a model config, derive every tensor shape, and generate kernels
//! specialized for those shapes on DGX Spark (GB10, compute capability 12.1).
//!
//! Pipeline:  config.json -> ModelSpec -> Plan (shapes + tile/launch choices) -> CUDA source
//!            -> NVRTC (sm_121a) -> cubin cache -> runtime.

pub mod codegen;
pub mod config;
pub mod modules;
pub mod plan;
pub mod target;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
pub mod verify;
#[cfg(feature = "cuda")]
pub mod device_weights;
