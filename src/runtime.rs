//! GB10 runtime: compile generated source with NVRTC for sm_121a and load the entry points.
//! Feature-gated (`--features cuda`) so the planner/codegen build anywhere.
use crate::codegen::Generated;
use anyhow::{Context, Result};
use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::Arc;

/// Dynamic shared memory a kernel may use without opting in.
const DEFAULT_DYN_SMEM: u32 = 48 * 1024;

pub struct Engine {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub funcs: HashMap<String, CudaFunction>,
}

impl Engine {
    pub fn load(g: &Generated, cuda_include: &str) -> Result<Self> {
        let ctx = CudaContext::new(0).context("opening GPU 0")?;
        let (maj, min) = ctx.compute_capability()?;
        anyhow::ensure!((maj, min) == (12, 1), "expected GB10 (cc 12.1), found {maj}.{min}");
        let src = &g.files.iter().find(|(n, _)| n == "model.cu").context("model.cu missing")?.1;
        let opts = CompileOptions {
            arch: Some("sm_121a"),
            include_paths: vec![cuda_include.to_string()],
            options: vec!["--std=c++17".into(), "--use_fast_math".into()],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(src, opts).context("NVRTC compile")?;
        let module = ctx.load_module(ptx)?;
        let mut funcs = HashMap::new();
        for k in g.kernels.iter().filter(|k| k.status == "generated") {
            let f = module.load_function(&k.entry)?;
            // Above 48 KB the launch fails with CUDA_ERROR_INVALID_VALUE unless the function opts in.
            if k.dyn_smem > DEFAULT_DYN_SMEM {
                f.set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    k.dyn_smem as i32,
                )
                .with_context(|| format!("opting {} into {} B dynamic smem", k.entry, k.dyn_smem))?;
            }
            funcs.insert(k.entry.clone(), f);
        }
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, module, funcs })
    }
}
