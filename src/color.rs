//! 色彩调整 (对应 Photoshop 「图像 > 调整」菜单)。

use crate::image::{clamp_u8, Image};

/// RGB(0..=255) -> HSL (h:0..360, s,l:0..1)。
pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let l = (mx + mn) / 2.0;
    let d = mx - mn;
    if d < 1e-6 {
        return (0.0, 0.0, l);
    }
    let s = d / (1.0 - (2.0 * l - 1.0).abs());
    let h = if mx == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if mx == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    } * 60.0;
    (h, s, l)
}

/// HSL -> RGB。
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
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
    let m = l - c / 2.0;
    (clamp_u8((r + m) * 255.0), clamp_u8((g + m) * 255.0), clamp_u8((b + m) * 255.0))
}

/// 灰度化方式。
#[derive(Clone, Copy, Debug)]
pub enum GrayMode {
    /// Rec.601 加权 (0.299/0.587/0.114)。
    Luma601,
    /// Rec.709 加权 (0.2126/0.7152/0.0722)。
    Luma709,
    /// 三通道平均。
    Average,
    /// (max+min)/2, 去饱和度 (Desaturate)。
    Lightness,
}

/// 去色/灰度化。
pub fn grayscale(img: &Image, mode: GrayMode) -> Image {
    img.map_pixels(|p| {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let v = match mode {
            GrayMode::Luma601 => 0.299 * r + 0.587 * g + 0.114 * b,
            GrayMode::Luma709 => 0.2126 * r + 0.7152 * g + 0.0722 * b,
            GrayMode::Average => (r + g + b) / 3.0,
            GrayMode::Lightness => (r.max(g).max(b) + r.min(g).min(b)) / 2.0,
        };
        let c = clamp_u8(v);
        [c, c, c, p[3]]
    })
}

/// 亮度/对比度。`brightness`: -255..255 加法偏移; `contrast`: -100..100 (百分比, 0 不变)。
/// 对比度以 128 为中心做线性缩放: `(v-128)*k+128`, `k=(100+contrast)/100` 的平方曲线, 与 PS 旧版"传统"模式近似。
pub fn brightness_contrast(img: &Image, brightness: f32, contrast: f32) -> Image {
    let k = ((100.0 + contrast.clamp(-100.0, 100.0)) / 100.0).powi(2);
    img.map_pixels(|p| {
        let f = |v: u8| clamp_u8((v as f32 - 128.0) * k + 128.0 + brightness);
        [f(p[0]), f(p[1]), f(p[2]), p[3]]
    })
}

/// 反相。
pub fn invert(img: &Image) -> Image {
    img.map_pixels(|p| [255 - p[0], 255 - p[1], 255 - p[2], p[3]])
}

/// Gamma 校正 (`gamma>1` 提亮中间调)。
pub fn gamma(img: &Image, gamma: f32) -> Image {
    let lut = build_lut(|v| 255.0 * (v / 255.0).powf(1.0 / gamma.max(1e-3)));
    apply_lut(img, &lut, &lut, &lut)
}

/// 色阶 (Levels): 输入黑/白场、中间调 gamma、输出黑/白场。
pub fn levels(img: &Image, in_black: f32, in_white: f32, mid_gamma: f32, out_black: f32, out_white: f32) -> Image {
    let lut = build_lut(|v| {
        let t = ((v - in_black) / (in_white - in_black).max(1e-3)).clamp(0.0, 1.0);
        out_black + t.powf(1.0 / mid_gamma.max(1e-3)) * (out_white - out_black)
    });
    apply_lut(img, &lut, &lut, &lut)
}

/// 曲线 (Curves): 控制点 `(输入,输出)` 0..=255, 使用单调三次插值 (Fritsch–Carlson) 避免过冲。
/// 分别指定 R、G、B 曲线; 传入同一组即为 RGB 复合曲线。
pub fn curves(img: &Image, r: &[(f32, f32)], g: &[(f32, f32)], b: &[(f32, f32)]) -> Image {
    apply_lut(img, &curve_lut(r), &curve_lut(g), &curve_lut(b))
}

fn curve_lut(pts: &[(f32, f32)]) -> [u8; 256] {
    let mut p: Vec<(f32, f32)> = pts.to_vec();
    p.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if p.is_empty() {
        p = vec![(0.0, 0.0), (255.0, 255.0)];
    }
    let n = p.len();
    let mut d = vec![0.0f32; n];
    if n > 1 {
        let sl: Vec<f32> = (0..n - 1).map(|i| (p[i + 1].1 - p[i].1) / (p[i + 1].0 - p[i].0).max(1e-6)).collect();
        d[0] = sl[0];
        d[n - 1] = sl[n - 2];
        for i in 1..n - 1 {
            d[i] = if sl[i - 1] * sl[i] <= 0.0 { 0.0 } else { (sl[i - 1] + sl[i]) / 2.0 };
        }
        for i in 0..n - 1 {
            if sl[i].abs() < 1e-9 {
                d[i] = 0.0;
                d[i + 1] = 0.0;
            } else {
                let (a, b) = (d[i] / sl[i], d[i + 1] / sl[i]);
                let s = a * a + b * b;
                if s > 9.0 {
                    let t = 3.0 / s.sqrt();
                    d[i] = t * a * sl[i];
                    d[i + 1] = t * b * sl[i];
                }
            }
        }
    }
    let mut lut = [0u8; 256];
    for (x, o) in lut.iter_mut().enumerate() {
        let x = x as f32;
        let y = if n == 1 {
            p[0].1
        } else if x <= p[0].0 {
            p[0].1
        } else if x >= p[n - 1].0 {
            p[n - 1].1
        } else {
            let i = p.windows(2).position(|w| x >= w[0].0 && x <= w[1].0).unwrap();
            let h = p[i + 1].0 - p[i].0;
            let t = (x - p[i].0) / h;
            let (t2, t3) = (t * t, t * t * t);
            (2.0 * t3 - 3.0 * t2 + 1.0) * p[i].1
                + (t3 - 2.0 * t2 + t) * h * d[i]
                + (-2.0 * t3 + 3.0 * t2) * p[i + 1].1
                + (t3 - t2) * h * d[i + 1]
        };
        *o = clamp_u8(y);
    }
    lut
}

fn build_lut(f: impl Fn(f32) -> f32) -> [u8; 256] {
    let mut l = [0u8; 256];
    for (i, o) in l.iter_mut().enumerate() {
        *o = clamp_u8(f(i as f32));
    }
    l
}

fn apply_lut(img: &Image, r: &[u8; 256], g: &[u8; 256], b: &[u8; 256]) -> Image {
    img.map_pixels(|p| [r[p[0] as usize], g[p[1] as usize], b[p[2] as usize], p[3]])
}

/// 色相/饱和度/明度。`hue`: -180..180 度; `saturation`: -100..100; `lightness`: -100..100。
pub fn hue_saturation(img: &Image, hue: f32, saturation: f32, lightness: f32) -> Image {
    img.map_pixels(|p| {
        let (h, s, l) = rgb_to_hsl(p[0], p[1], p[2]);
        let s = if saturation >= 0.0 { s + (1.0 - s) * saturation / 100.0 } else { s * (1.0 + saturation / 100.0) };
        let l = if lightness >= 0.0 { l + (1.0 - l) * lightness / 100.0 } else { l * (1.0 + lightness / 100.0) };
        let (r, g, b) = hsl_to_rgb(h + hue, s.clamp(0.0, 1.0), l.clamp(0.0, 1.0));
        [r, g, b, p[3]]
    })
}

/// 自然饱和度 (Vibrance): 对低饱和像素增强更多, 保护已饱和颜色。`amount`: -1..1。
pub fn vibrance(img: &Image, amount: f32) -> Image {
    img.map_pixels(|p| {
        let (h, s, l) = rgb_to_hsl(p[0], p[1], p[2]);
        let s2 = (s + amount * (1.0 - s) * (1.0 - s)).clamp(0.0, 1.0);
        let (r, g, b) = hsl_to_rgb(h, s2, l);
        [r, g, b, p[3]]
    })
}

/// 通道混合器: 3x3 矩阵 (每行对应输出 R,G,B, 每行系数作用于输入 R,G,B) 加常量偏移。
pub fn channel_mixer(img: &Image, m: [[f32; 3]; 3], offset: [f32; 3]) -> Image {
    img.map_pixels(|p| {
        let v = [p[0] as f32, p[1] as f32, p[2] as f32];
        let o = |i: usize| clamp_u8(m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + offset[i]);
        [o(0), o(1), o(2), p[3]]
    })
}

/// 棕褐色 (Sepia) 老照片效果。
pub fn sepia(img: &Image) -> Image {
    channel_mixer(img, [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], [0.0; 3])
}

/// 色调分离 (Posterize): 每通道保留 `levels` (>=2) 个色阶。
pub fn posterize(img: &Image, levels: u32) -> Image {
    let n = levels.max(2) as f32 - 1.0;
    img.map_pixels(|p| {
        let f = |v: u8| clamp_u8((v as f32 / 255.0 * n).round() / n * 255.0);
        [f(p[0]), f(p[1]), f(p[2]), p[3]]
    })
}

/// 阈值: 亮度 >= threshold 为白, 否则为黑。
pub fn threshold(img: &Image, threshold: u8) -> Image {
    img.map_pixels(|p| {
        let l = clamp_u8(0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32);
        let v = if l >= threshold { 255 } else { 0 };
        [v, v, v, p[3]]
    })
}

/// 曝光度: `exposure` 以 EV 为单位 (线性光 * 2^ev), `offset` 位移, `gamma` 校正。
pub fn exposure(img: &Image, exposure: f32, offset: f32, gamma_c: f32) -> Image {
    let lut = build_lut(|v| {
        let lin = (v / 255.0).powf(2.2) * 2f32.powf(exposure) + offset;
        255.0 * lin.max(0.0).powf(1.0 / 2.2).powf(1.0 / gamma_c.max(1e-3))
    });
    apply_lut(img, &lut, &lut, &lut)
}

/// 色彩平衡: 对阴影/中间调/高光分别加 RGB 偏移 (各 -100..100)。
pub fn color_balance(img: &Image, shadows: [f32; 3], midtones: [f32; 3], highlights: [f32; 3]) -> Image {
    img.map_pixels(|p| {
        let l = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0;
        let ws = (1.0 - l * 4.0).clamp(0.0, 1.0);
        let wh = (l * 4.0 - 3.0).clamp(0.0, 1.0);
        let wm = 1.0 - ws - wh;
        let f = |c: usize| clamp_u8(p[c] as f32 + (shadows[c] * ws + midtones[c] * wm + highlights[c] * wh) * 0.64);
        [f(0), f(1), f(2), p[3]]
    })
}

/// 照片滤镜: 以 `color` 叠加 `density`(0..1) 的色温滤镜, 可选保持明度。
pub fn photo_filter(img: &Image, color: [u8; 3], density: f32, preserve_luminosity: bool) -> Image {
    img.map_pixels(|p| {
        let l0 = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        let mut v = [0.0f32; 3];
        for c in 0..3 {
            v[c] = p[c] as f32 * (1.0 - density) + p[c] as f32 * color[c] as f32 / 255.0 * density;
        }
        if preserve_luminosity {
            let l1 = 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2];
            if l1 > 1e-3 {
                for c in v.iter_mut() {
                    *c *= l0 / l1;
                }
            }
        }
        [clamp_u8(v[0]), clamp_u8(v[1]), clamp_u8(v[2]), p[3]]
    })
}

/// 256 阶直方图, 返回 [R,G,B,亮度]。
pub fn histogram(img: &Image) -> [[u32; 256]; 4] {
    let mut h = [[0u32; 256]; 4];
    for p in img.data.chunks_exact(4) {
        h[0][p[0] as usize] += 1;
        h[1][p[1] as usize] += 1;
        h[2][p[2] as usize] += 1;
        h[3][clamp_u8(0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as usize] += 1;
    }
    h
}

/// 自动色阶: 每通道按百分位 `clip`(如 0.005) 裁剪后线性拉伸 (Auto Levels)。
pub fn auto_levels(img: &Image, clip: f32) -> Image {
    let h = histogram(img);
    let total = (img.width * img.height) as f32;
    let mut luts = [[0u8; 256]; 3];
    for c in 0..3 {
        let (mut lo, mut hi) = (0usize, 255usize);
        let mut acc = 0.0;
        while lo < 255 && acc + (h[c][lo] as f32) < total * clip {
            acc += h[c][lo] as f32;
            lo += 1;
        }
        acc = 0.0;
        while hi > 0 && acc + (h[c][hi] as f32) < total * clip {
            acc += h[c][hi] as f32;
            hi -= 1;
        }
        let (lo, hi) = (lo as f32, (hi as f32).max(lo as f32 + 1.0));
        luts[c] = build_lut(|v| (v - lo) / (hi - lo) * 255.0);
    }
    apply_lut(img, &luts[0], &luts[1], &luts[2])
}

/// 直方图均衡化 (仅作用于 HSL 的明度, 保持色相)。
pub fn equalize(img: &Image) -> Image {
    let h = histogram(img)[3];
    let total: u32 = h.iter().sum();
    let mut cdf = [0u32; 256];
    let mut a = 0;
    for i in 0..256 {
        a += h[i];
        cdf[i] = a;
    }
    let cmin = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
    img.map_pixels(|p| {
        let (hh, s, l) = rgb_to_hsl(p[0], p[1], p[2]);
        let li = clamp_u8(0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as usize;
        let nl = if total > cmin { (cdf[li] - cmin) as f32 / (total - cmin) as f32 } else { l };
        let (r, g, b) = hsl_to_rgb(hh, s, nl);
        [r, g, b, p[3]]
    })
}

/// 暗角 (Vignette): `strength` 0..1, `radius` 0..1 (相对半对角线起始位置), `feather` 过渡宽度。
pub fn vignette(img: &Image, strength: f32, radius: f32, feather: f32) -> Image {
    let (cx, cy) = (img.width as f32 / 2.0, img.height as f32 / 2.0);
    let maxd = (cx * cx + cy * cy).sqrt();
    let mut out = img.clone();
    for y in 0..img.height {
        for x in 0..img.width {
            let d = (((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / maxd - radius) / feather.max(1e-3);
            let t = d.clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            let k = 1.0 - strength * t;
            let p = img.get(x, y);
            out.set(x, y, [clamp_u8(p[0] as f32 * k), clamp_u8(p[1] as f32 * k), clamp_u8(p[2] as f32 * k), p[3]]);
        }
    }
    out
}

/// 渐变映射: 按亮度从 `stops` (位置0..1, RGB) 线性插值着色。
pub fn gradient_map(img: &Image, stops: &[(f32, [u8; 3])]) -> Image {
    let mut s = stops.to_vec();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    img.map_pixels(|p| {
        let l = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0;
        let c = if l <= s[0].0 {
            s[0].1
        } else if l >= s[s.len() - 1].0 {
            s[s.len() - 1].1
        } else {
            let i = s.windows(2).position(|w| l >= w[0].0 && l <= w[1].0).unwrap();
            let t = (l - s[i].0) / (s[i + 1].0 - s[i].0).max(1e-6);
            let m = |k: usize| clamp_u8(s[i].1[k] as f32 * (1.0 - t) + s[i + 1].1[k] as f32 * t);
            [m(0), m(1), m(2)]
        };
        [c[0], c[1], c[2], p[3]]
    })
}
