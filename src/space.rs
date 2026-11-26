//! 色彩空间转换 (HSV / CIE Lab / YCbCr) 及基于 Lab 的调整。

use crate::image::{clamp_u8, Image};

/// RGB(0..255) -> HSV (h:0..360, s,v:0..1)。
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let mx = r.max(g).max(b);
    let d = mx - r.min(g).min(b);
    let h = if d < 1e-6 {
        0.0
    } else if mx == r {
        ((g - b) / d).rem_euclid(6.0) * 60.0
    } else if mx == g {
        ((b - r) / d + 2.0) * 60.0
    } else {
        ((r - g) / d + 4.0) * 60.0
    };
    (h, if mx <= 0.0 { 0.0 } else { d / mx }, mx)
}

/// HSV -> RGB。
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let hp = h.rem_euclid(360.0) / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (clamp_u8((r + m) * 255.0), clamp_u8((g + m) * 255.0), clamp_u8((b + m) * 255.0))
}

fn srgb_to_lin(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}
fn lin_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// sRGB -> CIE Lab (D65), L:0..100。
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (srgb_to_lin(r as f32 / 255.0), srgb_to_lin(g as f32 / 255.0), srgb_to_lin(b as f32 / 255.0));
    let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// CIE Lab -> sRGB。
pub fn lab_to_rgb(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let (fx, fz) = (fy + a / 500.0, fy - b / 200.0);
    let inv = |t: f32| if t.powi(3) > 0.008856 { t.powi(3) } else { (t - 16.0 / 116.0) / 7.787 };
    let (x, y, z) = (inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883);
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    let f = |v: f32| clamp_u8(lin_to_srgb(v.clamp(0.0, 1.0)) * 255.0);
    (f(r), f(g), f(bl))
}

/// RGB -> YCbCr (JPEG, 全范围)。
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    (0.299 * r + 0.587 * g + 0.114 * b, 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b, 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b)
}

/// 对每个像素在 Lab 空间中应用 `f(L,a,b) -> (L,a,b)`。
pub fn map_lab(img: &Image, f: impl Fn(f32, f32, f32) -> (f32, f32, f32)) -> Image {
    img.map_pixels(|p| {
        let (l, a, b) = rgb_to_lab(p[0], p[1], p[2]);
        let (l, a, b) = f(l, a, b);
        let (r, g, bl) = lab_to_rgb(l, a, b);
        [r, g, bl, p[3]]
    })
}

/// 仅调整 Lab 明度 L 的对比度 (不改变色彩, 避免 RGB 对比度带来的色偏)。`k>1` 增强。
pub fn lab_contrast(img: &Image, k: f32) -> Image {
    map_lab(img, |l, a, b| (((l - 50.0) * k + 50.0).clamp(0.0, 100.0), a, b))
}

/// Lab 色彩增强: 放大 a/b 通道 `k` 倍 (类似 PS Lab 模式下加强饱和度)。
pub fn lab_saturation(img: &Image, k: f32) -> Image {
    map_lab(img, |l, a, b| (l, (a * k).clamp(-128.0, 127.0), (b * k).clamp(-128.0, 127.0)))
}

/// 提取单个通道: Lab 的 L/a/b 或 HSV 的 H/S/V, 归一化为灰度图可视化。`kind`: "L","a","b","H","S","V","Y","Cb","Cr"。
pub fn channel_view(img: &Image, kind: &str) -> Image {
    let mut g = crate::Gray::new(img.width, img.height, 0.0);
    for i in 0..img.width * img.height {
        let (r, gg, b) = (img.data[i * 4], img.data[i * 4 + 1], img.data[i * 4 + 2]);
        g.data[i] = match kind {
            "L" => rgb_to_lab(r, gg, b).0 * 2.55,
            "a" => rgb_to_lab(r, gg, b).1 + 128.0,
            "b" => rgb_to_lab(r, gg, b).2 + 128.0,
            "H" => rgb_to_hsv(r, gg, b).0 / 360.0 * 255.0,
            "S" => rgb_to_hsv(r, gg, b).1 * 255.0,
            "V" => rgb_to_hsv(r, gg, b).2 * 255.0,
            "Cb" => rgb_to_ycbcr(r, gg, b).1,
            "Cr" => rgb_to_ycbcr(r, gg, b).2,
            _ => rgb_to_ycbcr(r, gg, b).0,
        };
    }
    g.to_image()
}
