//! Rust 图像处理算法库: 参考 Photoshop 的调色、滤镜、变换、扭曲、边缘/轮廓等算法, 零第三方依赖。

pub mod binary;
pub mod color;
pub mod distort;
pub mod edge;
pub mod filter;
pub mod image;
pub mod transform;

pub use image::{Border, Gray, Image};
pub use transform::Interp;
