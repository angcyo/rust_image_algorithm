//! 图层混合模式 (对应 Photoshop 图层混合) 与蒙版合成。

use crate::image::{clamp_u8, Image};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    /// 正片叠底 a*b
    Multiply,
    /// 滤色 1-(1-a)(1-b)
    Screen,
    /// 叠加 (按底层亮度选 multiply/screen)
    Overlay,
    /// 柔光 (W3C 公式)
    SoftLight,
    /// 强光 (按上层亮度, 等于上下层互换的 Overlay)
    HardLight,
    /// 颜色减淡 b/(1-a)
    ColorDodge,
    /// 颜色加深 1-(1-b)/a
    ColorBurn,
    Darken,
    Lighten,
    Difference,
    Exclusion,
    /// 线性减淡 (添加)
    LinearDodge,
    /// 线性加深
    LinearBurn,
}

fn blend_channel(mode: BlendMode, b: f32, s: f32) -> f32 {
    // b: 底层 (0..1), s: 上层 (0..1)
    match mode {
        BlendMode::Normal => s,
        BlendMode::Multiply => b * s,
        BlendMode::Screen => 1.0 - (1.0 - b) * (1.0 - s),
        BlendMode::Overlay => blend_channel(BlendMode::HardLight, s, b),
        BlendMode::HardLight => if s <= 0.5 { 2.0 * b * s } else { 1.0 - 2.0 * (1.0 - b) * (1.0 - s) },
        BlendMode::SoftLight => {
            if s <= 0.5 {
                b - (1.0 - 2.0 * s) * b * (1.0 - b)
            } else {
                let d = if b <= 0.25 { ((16.0 * b - 12.0) * b + 4.0) * b } else { b.sqrt() };
                b + (2.0 * s - 1.0) * (d - b)
            }
        }
        BlendMode::ColorDodge => if s >= 1.0 { 1.0 } else { (b / (1.0 - s)).min(1.0) },
        BlendMode::ColorBurn => if s <= 0.0 { 0.0 } else { 1.0 - ((1.0 - b) / s).min(1.0) },
        BlendMode::Darken => b.min(s),
        BlendMode::Lighten => b.max(s),
        BlendMode::Difference => (b - s).abs(),
        BlendMode::Exclusion => b + s - 2.0 * b * s,
        BlendMode::LinearDodge => (b + s).min(1.0),
        BlendMode::LinearBurn => (b + s - 1.0).max(0.0),
    }
}

/// 把 `top` 以混合模式 `mode` 与 `opacity`(0..1) 叠加到 `base` 上; 两图尺寸须相同。
/// 可选 `mask` (灰度图尺寸相同, 255=完全应用) 作为图层蒙版; 考虑 top 的 alpha。
pub fn blend(base: &Image, top: &Image, mode: BlendMode, opacity: f32, mask: Option<&crate::Gray>) -> Image {
    assert_eq!((base.width, base.height), (top.width, top.height));
    let mut out = base.clone();
    for i in 0..base.width * base.height {
        let m = mask.map_or(1.0, |m| m.data[i] / 255.0);
        let a = top.data[i * 4 + 3] as f32 / 255.0 * opacity * m;
        for c in 0..3 {
            let (b, s) = (base.data[i * 4 + c] as f32 / 255.0, top.data[i * 4 + c] as f32 / 255.0);
            let r = blend_channel(mode, b, s);
            out.data[i * 4 + c] = clamp_u8((b * (1.0 - a) + r * a) * 255.0);
        }
    }
    out
}

/// 羽化蒙版: 将二值/灰度蒙版高斯模糊 `radius` 像素。
pub fn feather_mask(mask: &crate::Gray, radius: f32) -> crate::Gray {
    crate::filter::gaussian_blur_gray(mask, radius)
}
