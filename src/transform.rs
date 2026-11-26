//! 几何变换: 缩放、旋转、翻转、裁剪、仿射、透视 (对应 Photoshop 「编辑 > 变换」)。
//!
//! 所有变换采用**反向映射**: 遍历目标像素, 求其在源图中的位置再插值, 避免空洞。

use crate::image::{clamp_u8, map_coord, Border, Gray, Image};

/// 插值方式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    /// 最近邻: 最快, 有锯齿, 适合像素画。
    Nearest,
    /// 双线性: 平滑, 适合一般缩放。
    Bilinear,
    /// 双三次 (Catmull-Rom, a=-0.5): 更锐利, PS 默认缩放算法的近似。
    Bicubic,
}

/// 双线性采样, 返回 f32 RGBA。
pub fn sample_bilinear(img: &Image, x: f32, y: f32, b: Border) -> [f32; 4] {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p = [img.get_border(x0, y0, b), img.get_border(x0 + 1, y0, b), img.get_border(x0, y0 + 1, b), img.get_border(x0 + 1, y0 + 1, b)];
    let mut o = [0.0; 4];
    for c in 0..4 {
        o[c] = (p[0][c] as f32 * (1.0 - fx) + p[1][c] as f32 * fx) * (1.0 - fy) + (p[2][c] as f32 * (1.0 - fx) + p[3][c] as f32 * fx) * fy;
    }
    o
}

fn cubic_w(t: f32) -> f32 {
    let a = -0.5;
    let t = t.abs();
    if t <= 1.0 {
        (a + 2.0) * t.powi(3) - (a + 3.0) * t * t + 1.0
    } else if t < 2.0 {
        a * t.powi(3) - 5.0 * a * t * t + 8.0 * a * t - 4.0 * a
    } else {
        0.0
    }
}

/// 双三次采样。
pub fn sample_bicubic(img: &Image, x: f32, y: f32, b: Border) -> [f32; 4] {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let mut o = [0.0f32; 4];
    for j in -1..=2isize {
        let wy = cubic_w(j as f32 - fy);
        for i in -1..=2isize {
            let w = wy * cubic_w(i as f32 - fx);
            let p = img.get_border(x0 as isize + i, y0 as isize + j, b);
            for c in 0..4 {
                o[c] += w * p[c] as f32;
            }
        }
    }
    o
}

/// 统一采样入口。
pub fn sample(img: &Image, x: f32, y: f32, interp: Interp, b: Border) -> [u8; 4] {
    let v = match interp {
        Interp::Nearest => {
            let p = img.get_border(x.round() as isize, y.round() as isize, b);
            return p;
        }
        Interp::Bilinear => sample_bilinear(img, x, y, b),
        Interp::Bicubic => sample_bicubic(img, x, y, b),
    };
    [clamp_u8(v[0]), clamp_u8(v[1]), clamp_u8(v[2]), clamp_u8(v[3])]
}

/// 灰度双线性采样 (边缘 Clamp)。
pub fn sample_gray_bilinear(g: &Gray, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let b = Border::Clamp;
    (g.get_border(x0, y0, b) * (1.0 - fx) + g.get_border(x0 + 1, y0, b) * fx) * (1.0 - fy)
        + (g.get_border(x0, y0 + 1, b) * (1.0 - fx) + g.get_border(x0 + 1, y0 + 1, b) * fx) * fy
}

/// 通用重映射: 对目标尺寸 `(w,h)` 的每个像素 `(x,y)`, 由 `map` 给出源坐标。`None` 表示填充 `fill`。
pub fn remap(img: &Image, w: usize, h: usize, interp: Interp, border: Border, fill: [u8; 4], map: impl Fn(f32, f32) -> Option<(f32, f32)>) -> Image {
    let mut out = Image::new(w, h, fill);
    for y in 0..h {
        for x in 0..w {
            if let Some((sx, sy)) = map(x as f32, y as f32) {
                out.set(x, y, sample(img, sx, sy, interp, border));
            }
        }
    }
    out
}

/// 缩放到指定尺寸 (像素中心对齐)。缩小倍率大于 2 时先做高斯预滤波以抑制混叠。
pub fn resize(img: &Image, new_w: usize, new_h: usize, interp: Interp) -> Image {
    let (sx, sy) = (img.width as f32 / new_w as f32, img.height as f32 / new_h as f32);
    let src = if (sx > 2.0 || sy > 2.0) && interp != Interp::Nearest {
        crate::filter::gaussian_blur(img, (sx.max(sy) * 0.5).max(0.5))
    } else {
        img.clone()
    };
    remap(&src, new_w, new_h, interp, Border::Clamp, [0; 4], |x, y| Some(((x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5)))
}

/// 等比缩放, 倍率 `scale`。
pub fn scale(img: &Image, scale: f32, interp: Interp) -> Image {
    resize(img, ((img.width as f32 * scale).round() as usize).max(1), ((img.height as f32 * scale).round() as usize).max(1), interp)
}

/// 裁剪矩形区域 (越界部分自动截断)。
pub fn crop(img: &Image, x: usize, y: usize, w: usize, h: usize) -> Image {
    let x1 = (x + w).min(img.width);
    let y1 = (y + h).min(img.height);
    let (x, y) = (x.min(x1), y.min(y1));
    Image::from_fn(x1 - x, y1 - y, |i, j| img.get(x + i, y + j))
}

/// 水平翻转。
pub fn flip_horizontal(img: &Image) -> Image {
    Image::from_fn(img.width, img.height, |x, y| img.get(img.width - 1 - x, y))
}

/// 垂直翻转。
pub fn flip_vertical(img: &Image) -> Image {
    Image::from_fn(img.width, img.height, |x, y| img.get(x, img.height - 1 - y))
}

/// 顺时针旋转 90° * `times`。
pub fn rotate90(img: &Image, times: u32) -> Image {
    match times % 4 {
        0 => img.clone(),
        1 => Image::from_fn(img.height, img.width, |x, y| img.get(y, img.height - 1 - x)),
        2 => Image::from_fn(img.width, img.height, |x, y| img.get(img.width - 1 - x, img.height - 1 - y)),
        _ => Image::from_fn(img.height, img.width, |x, y| img.get(img.width - 1 - y, x)),
    }
}

/// 任意角度旋转 (顺时针为正, 度)。`expand=true` 时扩展画布容纳全部内容, 否则保持原尺寸。
pub fn rotate(img: &Image, angle: f32, expand: bool, interp: Interp, fill: [u8; 4]) -> Image {
    let (s, c) = angle.to_radians().sin_cos();
    let (w, h) = (img.width as f32, img.height as f32);
    let (nw, nh) = if expand { ((w * c.abs() + h * s.abs()).ceil() as usize, (w * s.abs() + h * c.abs()).ceil() as usize) } else { (img.width, img.height) };
    let (cx, cy) = ((w - 1.0) / 2.0, (h - 1.0) / 2.0);
    let (ncx, ncy) = ((nw as f32 - 1.0) / 2.0, (nh as f32 - 1.0) / 2.0);
    remap(img, nw, nh, interp, Border::Zero, fill, |x, y| {
        let (dx, dy) = (x - ncx, y - ncy);
        // 逆旋转
        let (sx, sy) = (cx + dx * c + dy * s, cy - dx * s + dy * c);
        if sx < -0.5 || sy < -0.5 || sx > w - 0.5 || sy > h - 0.5 { None } else { Some((sx, sy)) }
    })
}

/// 2x3 仿射矩阵 `[[a,b,tx],[c,d,ty]]`: `x' = a x + b y + tx`, `y' = c x + d y + ty` (源 -> 目标)。
pub type Affine = [[f32; 3]; 2];

/// 求仿射逆矩阵。奇异时返回 None。
pub fn affine_invert(m: &Affine) -> Option<Affine> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-9 {
        return None;
    }
    let (a, b, c, d) = (m[1][1] / det, -m[0][1] / det, -m[1][0] / det, m[0][0] / det);
    Some([[a, b, -(a * m[0][2] + b * m[1][2])], [c, d, -(c * m[0][2] + d * m[1][2])]])
}

/// 应用仿射变换 (平移/缩放/旋转/斜切的任意组合), 输出尺寸 `(w,h)`。
pub fn warp_affine(img: &Image, m: &Affine, w: usize, h: usize, interp: Interp, fill: [u8; 4]) -> Option<Image> {
    let inv = affine_invert(m)?;
    let (iw, ih) = (img.width as f32, img.height as f32);
    Some(remap(img, w, h, interp, Border::Clamp, fill, |x, y| {
        let (sx, sy) = (inv[0][0] * x + inv[0][1] * y + inv[0][2], inv[1][0] * x + inv[1][1] * y + inv[1][2]);
        if sx < -0.5 || sy < -0.5 || sx > iw - 0.5 || sy > ih - 0.5 { None } else { Some((sx, sy)) }
    }))
}

/// 斜切 (Skew): `sx`/`sy` 为水平/垂直斜切角度(度)。输出画布自动扩展。
pub fn skew(img: &Image, sx_deg: f32, sy_deg: f32, interp: Interp, fill: [u8; 4]) -> Image {
    let (tx, ty) = (sx_deg.to_radians().tan(), sy_deg.to_radians().tan());
    let (w, h) = (img.width as f32, img.height as f32);
    let nw = (w + h * tx.abs()).ceil() as usize;
    let nh = (h + w * ty.abs()).ceil() as usize;
    let ox = if tx < 0.0 { -tx * h } else { 0.0 };
    let oy = if ty < 0.0 { -ty * w } else { 0.0 };
    let m = [[1.0, tx, ox], [ty, 1.0, oy]];
    warp_affine(img, &m, nw, nh, interp, fill).unwrap()
}

/// 3x3 单应矩阵 (行优先), 将点 `(x,y)` 映射为 `(x',y') = ((h0x+h1y+h2)/w, (h3x+h4y+h5)/w)`, `w = h6x+h7y+h8`。
pub type Homography = [f32; 9];

/// 由 4 组点对求透视变换 (DLT, 固定 h8=1, 高斯消元)。
/// `src[i]` 映射到 `dst[i]`。点共线/退化时返回 None。
pub fn homography_from_points(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Option<Homography> {
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (src[i].0 as f64, src[i].1 as f64);
        let (u, v) = (dst[i].0 as f64, dst[i].1 as f64);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for col in 0..8 {
        let piv = (col..8).max_by(|&p, &q| a[p][col].abs().partial_cmp(&a[q][col].abs()).unwrap())?;
        if a[piv][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, piv);
        for r in 0..8 {
            if r != col {
                let f = a[r][col] / a[col][col];
                for k in col..9 {
                    a[r][k] -= f * a[col][k];
                }
            }
        }
    }
    let mut h = [0.0f32; 9];
    for i in 0..8 {
        h[i] = (a[i][8] / a[i][i]) as f32;
    }
    h[8] = 1.0;
    Some(h)
}

/// 3x3 矩阵求逆。
pub fn homography_invert(h: &Homography) -> Option<Homography> {
    let m = h.map(|v| v as f64);
    let c = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    let det = m[0] * c[0] + m[1] * c[3] + m[2] * c[6];
    if det.abs() < 1e-12 {
        return None;
    }
    Some(c.map(|v| (v / det) as f32))
}

/// 应用单应矩阵到点。
pub fn apply_homography(h: &Homography, x: f32, y: f32) -> (f32, f32) {
    let w = h[6] * x + h[7] * y + h[8];
    ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
}

/// 透视变换: 已知单应矩阵 (源->目标), 输出 `(w,h)`。
pub fn warp_perspective(img: &Image, h: &Homography, w: usize, ht: usize, interp: Interp, fill: [u8; 4]) -> Option<Image> {
    let inv = homography_invert(h)?;
    let (iw, ih) = (img.width as f32, img.height as f32);
    Some(remap(img, w, ht, interp, Border::Clamp, fill, |x, y| {
        let den = inv[6] * x + inv[7] * y + inv[8];
        if den.abs() < 1e-9 {
            return None;
        }
        let (sx, sy) = apply_homography(&inv, x, y);
        if sx < -0.5 || sy < -0.5 || sx > iw - 0.5 || sy > ih - 0.5 { None } else { Some((sx, sy)) }
    }))
}

/// 透视变换: 把源图上四边形 `quad` (左上,右上,右下,左下) 拉正为 `out_w x out_h` 矩形 (文档矫正/透视裁剪)。
pub fn perspective_rectify(img: &Image, quad: &[(f32, f32); 4], out_w: usize, out_h: usize, interp: Interp) -> Option<Image> {
    let (w, h) = (out_w as f32 - 1.0, out_h as f32 - 1.0);
    let dst = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let hm = homography_from_points(quad, &dst)?;
    warp_perspective(img, &hm, out_w, out_h, interp, [0, 0, 0, 0])
}

/// 透视变换: 把整张图映射到目标四边形 `quad` (左上,右上,右下,左下), 画布尺寸 `(w,h)` (类似 PS「透视」/「扭曲」)。
pub fn perspective_distort(img: &Image, quad: &[(f32, f32); 4], w: usize, h: usize, interp: Interp, fill: [u8; 4]) -> Option<Image> {
    let (iw, ih) = (img.width as f32 - 1.0, img.height as f32 - 1.0);
    let src = [(0.0, 0.0), (iw, 0.0), (iw, ih), (0.0, ih)];
    let hm = homography_from_points(&src, quad)?;
    warp_perspective(img, &hm, w, h, interp, fill)
}

/// 自由变换后的坐标映射辅助: 保留未使用的边界检查工具。
#[allow(dead_code)]
pub(crate) fn in_bounds(img: &Image, x: isize, y: isize) -> bool {
    map_coord(x, img.width, Border::Zero).is_some() && map_coord(y, img.height, Border::Zero).is_some()
}
