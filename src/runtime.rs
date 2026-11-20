//! GB10 runtime: compile generated source with NVRTC for sm_121a and load the entry points.
//! Feature-gated (`--features cuda`) so the planner/codegen build anywhere.
use crate::codegen::Generated;
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Engine {
    pub ctx: Arc<CudaContext>,
    pub module: Arc<CudaModule>,
    pub funcs: HashMap<String, CudaFunction>,
}

impl Engine {
    pub fn load(g: &Generated, cuda_include: &str) -> Result<Self> {
        let ctx = CudaContext::new(0).context("opening GPU 0")?;
        let (maj, min) = ctx.compute_capability()?;
        anyhow::ensure!((maj, min) == (12, 1), "expected GB10 (cc 12.1), found {maj}.{min}");
        let src = &g.files.iter().find(|(n, _)| n == "model.cu").unwrap().1;
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
            funcs.insert(k.entry.clone(), module.load_function(&k.entry)?);
        }
        Ok(Self { ctx, module, funcs })
    }
}
