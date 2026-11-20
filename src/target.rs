//! GB10 / SM121 hardware model. Values are what the planner is allowed to rely on.
//! Sources: NVIDIA DGX Spark spec page; community SM121 measurements (see README).

#[derive(Debug, Clone, serde::Serialize)]
pub struct Target {
    pub name: &'static str,
    /// NVRTC/nvcc arch. The `a` suffix is required for block-scaled FP4 (`kind::mxf4nvf4`).
    pub arch: &'static str,
    pub sm_count: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_sm: u32,
    pub regs_per_sm: u32,
    /// Max dynamic shared memory per block after opt-in (bytes).
    pub smem_per_block_optin: u32,
    pub smem_per_sm: u32,
    pub l2_bytes: u64,
    /// Nominal DRAM bandwidth (GB/s). Never reached in practice.
    pub dram_bw_nominal_gbs: f64,
    /// Planning bandwidth: what a well-written streaming kernel actually gets (GB/s).
    pub dram_bw_effective_gbs: f64,
    /// Unified memory the GPU can realistically use (GiB), shared with the OS.
    pub usable_mem_gib: f64,
    /// Tensor-core instruction family available. No tcgen05 / TMEM / WGMMA on SM12x.
    pub mma_family: &'static str,
}

pub const GB10: Target = Target {
    name: "NVIDIA GB10 (DGX Spark)",
    arch: "sm_121a",
    sm_count: 48,
    warp_size: 32,
    max_threads_per_block: 1024,
    max_threads_per_sm: 1536,
    regs_per_sm: 65536,
    smem_per_block_optin: 101_376,
    smem_per_sm: 102_400,
    l2_bytes: 24 * 1024 * 1024,
    dram_bw_nominal_gbs: 273.0,
    dram_bw_effective_gbs: 235.0,
    usable_mem_gib: 114.0,
    mma_family: "mma.sync (bf16 m16n8k16, fp8 m16n8k32, nvfp4 kind::mxf4nvf4 m16n8k64)",
};
