//! EML Core in Rust: eml(x,y) = exp(x) - ln(y), fused & parallel.
//!
//! This is the Rust mirror of eml_core.py.  Every function computes the
//! identical mathematical result but:
//!   1. Operates on Complex<f64> natively (no Python overhead).
//!   2. Fuses the EML tree algebraically — e.g. eml_mul(a,b) goes from
//!      ~15 primitive calls to 4 transcendentals.
//!   3. Uses Rayon to parallelise element-wise and matmul operations.
//!
//! The Python reference stays the source of truth; verify.py checks that
//! Rust and Python agree to within 1e-9.

mod eml_ops;
pub mod eml_optimizer;
pub mod emilio;
pub mod gguf;
pub mod model;
pub mod tokenizer;
pub mod autoeml_kernel;
pub mod autoeml_reference;
#[cfg(feature = "python")]
mod python;

pub use eml_ops::*;
pub use eml_optimizer::*;
