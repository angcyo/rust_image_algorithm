//! 扭曲与变形 (对应 Photoshop 「滤镜 > 扭曲」「液化」「变形」)。
//!
//! 均采用反向映射 + 插值, 通过 [`crate::transform::remap`] 实现。
//! 参数中的坐标 `center` 为像素坐标; `radius` 为像素半径。

use crate::image::{Border, Image};
use crate::transform::{remap, Interp};
use std::f32::consts::{PI, TAU};

/// 旋转扭曲 (Twirl): 圆内像素按距离衰减旋转, `angle` 为中心处旋转角度(度)。
pub fn swirl(img: &Image, center: (f32, f32), radius: f32, angle: f32, interp: Interp) -> Image {
    let a = angle.to_radians();
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        let (dx, dy) = (x - center.0, y - center.1);
        let d = (dx * dx + dy * dy).sqrt();
        if d >= radius {
            return Some((x, y));
        }
        let t = 1.0 - d / radius;
        let th = a * t * t; // 二次衰减, 边缘平滑
        let (s, c) = th.sin_cos();
        Some((center.0 + dx * c - dy * s, center.1 + dx * s + dy * c))
    })
}

/// 挤压/膨胀 (Pinch / Spherize): `amount` -1..1, 正值膨胀(球面化), 负值挤压。
pub fn pinch(img: &Image, center: (f32, f32), radius: f32, amount: f32, interp: Interp) -> Image {
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        let (dx, dy) = (x - center.0, y - center.1);
        let d = (dx * dx + dy * dy).sqrt();
        if d >= radius || d < 1e-6 {
            return Some((x, y));
        }
        let r = d / radius;
        let nr = if amount >= 0.0 { r.powf(1.0 + amount.min(0.99) * 3.0) } else { r.powf(1.0 / (1.0 - amount.max(-1.0) * 0.75)) };
        let k = nr * radius / d;
        Some((center.0 + dx * k, center.1 + dy * k))
    })
}

/// 波纹 (Ripple): 沿 x、y 方向的正弦位移。`amplitude` 振幅(px), `wavelength` 波长(px)。
pub fn ripple(img: &Image, amplitude: f32, wavelength: f32, interp: Interp) -> Image {
    let k = TAU / wavelength.max(1.0);
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| Some((x + amplitude * (k * y).sin(), y + amplitude * (k * x).sin())))
}

/// 波浪 (Wave): 水平方向正弦位移; `phase` 相位(弧度) 可用于动画。`vertical=true` 时沿垂直方向位移。
pub fn wave(img: &Image, amplitude: f32, wavelength: f32, phase: f32, vertical: bool, interp: Interp) -> Image {
    let k = TAU / wavelength.max(1.0);
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        if vertical { Some((x, y + amplitude * (k * x + phase).sin())) } else { Some((x + amplitude * (k * y + phase).sin(), y)) }
    })
}

/// 径向透镜畸变 (Brown–Conrady 模型): `x_u = x_d (1 + k1 r² + k2 r⁴)`。
/// `k1>0` 桶形畸变 (类似 PS「镜头校正」负值方向), `k1<0` 枕形畸变。`r` 以半对角线归一化。
pub fn lens_distortion(img: &Image, k1: f32, k2: f32, interp: Interp) -> Image {
    let (cx, cy) = ((img.width as f32 - 1.0) / 2.0, (img.height as f32 - 1.0) / 2.0);
    let norm = (cx * cx + cy * cy).sqrt();
    remap(img, img.width, img.height, interp, Border::Zero, [0, 0, 0, 255], |x, y| {
        let (dx, dy) = ((x - cx) / norm, (y - cy) / norm);
        let r2 = dx * dx + dy * dy;
        let f = 1.0 + k1 * r2 + k2 * r2 * r2;
        Some((cx + dx * f * norm, cy + dy * f * norm))
    })
}

/// 鱼眼 (等距投影近似) `strength` > 0。
pub fn fisheye(img: &Image, strength: f32, interp: Interp) -> Image {
    let (cx, cy) = ((img.width as f32 - 1.0) / 2.0, (img.height as f32 - 1.0) / 2.0);
    let rmax = cx.min(cy).max(1.0);
    remap(img, img.width, img.height, interp, Border::Zero, [0, 0, 0, 255], |x, y| {
        let (dx, dy) = ((x - cx) / rmax, (y - cy) / rmax);
        let r = (dx * dx + dy * dy).sqrt();
        if r >= 1.0 || r < 1e-6 {
            return Some((x, y));
        }
        let nr = r.powf(1.0 + strength.max(0.0)).min(1.0); // 中心放大, 边缘压缩
        let k = nr / r;
        Some((cx + dx * k * rmax, cy + dy * k * rmax))
    })
}

/// 极坐标: 直角坐标 -> 极坐标 (`to_polar=true`, 把矩形卷成圆) 或 极坐标 -> 直角坐标。
pub fn polar(img: &Image, to_polar: bool, interp: Interp) -> Image {
    let (w, h) = (img.width as f32, img.height as f32);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let rmax = cx.min(cy);
    remap(img, img.width, img.height, interp, Border::Wrap, [0; 4], |x, y| {
        if to_polar {
            let (dx, dy) = (x - cx, y - cy);
            let r = (dx * dx + dy * dy).sqrt();
            let th = dy.atan2(dx).rem_euclid(TAU);
            Some((th / TAU * w, r / rmax * h))
        } else {
            let (th, r) = (x / w * TAU, y / h * rmax);
            Some((cx + r * th.cos(), cy + r * th.sin()))
        }
    })
}

/// 切变 (Shear) 滤镜: 按给定控制曲线 `curve` (自上而下的若干 x 位移, 线性插值) 水平移位。
pub fn shear_curve(img: &Image, curve: &[f32], interp: Interp) -> Image {
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        let t = y / (img.height as f32 - 1.0).max(1.0) * (curve.len() as f32 - 1.0);
        let i = (t.floor() as usize).min(curve.len().saturating_sub(2));
        let f = t - i as f32;
        let dx = if curve.len() < 2 { curve.first().copied().unwrap_or(0.0) } else { curve[i] * (1.0 - f) + curve[i + 1] * f };
        Some((x - dx, y))
    })
}

/// 球面映射 (类似 PS「球面化」, 包含整图内切圆): 内切圆内按球面折射映射, 圆外保持不变。
pub fn spherize(img: &Image, center: (f32, f32), radius: f32, interp: Interp) -> Image {
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        let (dx, dy) = (x - center.0, y - center.1);
        let d = (dx * dx + dy * dy).sqrt();
        if d >= radius || d < 1e-6 {
            return Some((x, y));
        }
        let r = d / radius;
        let z = (1.0 - r * r).sqrt();
        let nr = 1.0 - z;
        let k = nr * radius / d;
        Some((center.0 + dx * k, center.1 + dy * k))
    })
}

/// 圆柱面投影 (水平): `strength` 0..1, 0 不变, 1 为半圆柱, 常用于全景/卷轴效果。
pub fn cylinder(img: &Image, strength: f32, interp: Interp) -> Image {
    let w = img.width as f32;
    let cx = (w - 1.0) / 2.0;
    let s = strength.clamp(0.0, 0.999) * PI / 2.0;
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        if s < 1e-4 {
            return Some((x, y));
        }
        let u = (x - cx) / cx; // -1..1
        let th = u * s;
        Some((cx + th.sin() / s.sin() * cx, y))
    })
}

/// 网格变形 (Mesh Warp): 把图像划分为 `(cols x rows)` 个单元, `points` 为 `(cols+1)*(rows+1)` 个**目标**控制点 (行优先)。
/// 采用逐单元双线性反解 (牛顿迭代) 求源坐标, 适合 PS「变形」网格。
pub fn mesh_warp(img: &Image, cols: usize, rows: usize, points: &[(f32, f32)], interp: Interp) -> Image {
    assert_eq!(points.len(), (cols + 1) * (rows + 1));
    let (cw, rh) = (img.width as f32 / cols as f32, img.height as f32 / rows as f32);
    let mut out = Image::new(img.width, img.height, [0; 4]);
    for r in 0..rows {
        for c in 0..cols {
            let q = [points[r * (cols + 1) + c], points[r * (cols + 1) + c + 1], points[(r + 1) * (cols + 1) + c + 1], points[(r + 1) * (cols + 1) + c]];
            let (minx, maxx) = q.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
            let (miny, maxy) = q.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
            for y in (miny.floor().max(0.0) as usize)..=(maxy.ceil().min(img.height as f32 - 1.0).max(0.0) as usize) {
                for x in (minx.floor().max(0.0) as usize)..=(maxx.ceil().min(img.width as f32 - 1.0).max(0.0) as usize) {
                    if let Some((u, v)) = invert_bilinear(&q, x as f32, y as f32) {
                        if (-0.001..=1.001).contains(&u) && (-0.001..=1.001).contains(&v) {
                            let (sx, sy) = ((c as f32 + u) * cw, (r as f32 + v) * rh);
                            out.set(x, y, crate::transform::sample(img, sx - 0.0, sy - 0.0, interp, Border::Clamp));
                        }
                    }
                }
            }
        }
    }
    out
}

/// 求双线性四边形 (TL,TR,BR,BL) 内点 `(px,py)` 的参数 `(u,v)`, Newton 迭代。
fn invert_bilinear(q: &[(f32, f32); 4], px: f32, py: f32) -> Option<(f32, f32)> {
    let (mut u, mut v) = (0.5f32, 0.5f32);
    for _ in 0..12 {
        let f = |i: usize| {
            let (a, b, c, d) = (q[0], q[1], q[3], q[2]);
            let _ = i;
            (
                (1.0 - u) * (1.0 - v) * a.0 + u * (1.0 - v) * b.0 + (1.0 - u) * v * c.0 + u * v * d.0,
                (1.0 - u) * (1.0 - v) * a.1 + u * (1.0 - v) * b.1 + (1.0 - u) * v * c.1 + u * v * d.1,
            )
        };
        let (fx, fy) = f(0);
        let (ex, ey) = (fx - px, fy - py);
        let (a, b, c, d) = (q[0], q[1], q[3], q[2]);
        let dxu = (1.0 - v) * (b.0 - a.0) + v * (d.0 - c.0);
        let dxv = (1.0 - u) * (c.0 - a.0) + u * (d.0 - b.0);
        let dyu = (1.0 - v) * (b.1 - a.1) + v * (d.1 - c.1);
        let dyv = (1.0 - u) * (c.1 - a.1) + u * (d.1 - b.1);
        let det = dxu * dyv - dxv * dyu;
        if det.abs() < 1e-9 {
            return None;
        }
        u -= (dyv * ex - dxv * ey) / det;
        v -= (-dyu * ex + dxu * ey) / det;
    }
    Some((u, v))
}

/// 液化-向前变形 (Forward Warp): 在 `center` 半径内沿 `(dx,dy)` 推动像素, 强度按高斯/平滑衰减。
pub fn liquify_push(img: &Image, center: (f32, f32), radius: f32, dx: f32, dy: f32, interp: Interp) -> Image {
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        let d2 = (x - center.0).powi(2) + (y - center.1).powi(2);
        if d2 >= radius * radius {
            return Some((x, y));
        }
        let t = 1.0 - d2 / (radius * radius);
        let w = t * t;
        Some((x - dx * w, y - dy * w))
    })
}

/// 通用位移图置换 (Displace): `map_x`/`map_y` 为位移图, 取其亮度 (128 为中性) 乘 `scale` 作为位移。
pub fn displace(img: &Image, map: &Image, scale_x: f32, scale_y: f32, interp: Interp) -> Image {
    let g = map.to_gray();
    remap(img, img.width, img.height, interp, Border::Clamp, [0; 4], |x, y| {
        let v = g.get((x as usize).min(g.width - 1), (y as usize).min(g.height - 1));
        let d = (v - 128.0) / 128.0;
        Some((x + d * scale_x, y + d * scale_y))
    })
}
