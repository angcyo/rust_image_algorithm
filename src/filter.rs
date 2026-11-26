//! 滤镜: 卷积、模糊、锐化、降噪、风格化 (对应 Photoshop 「滤镜」菜单)。

use crate::image::{clamp_u8, Border, Gray, Image};

/// 简单可复现的 xorshift 伪随机数生成器 (避免第三方依赖)。
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    /// 0..1 均匀分布。
    pub fn uniform(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    /// 标准正态 (Box-Muller)。
    pub fn gauss(&mut self) -> f32 {
        let (u1, u2) = (self.uniform().max(1e-7), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// 通用二维卷积 (灰度)。`kernel` 为 `kw*kh` 行优先, 奇数尺寸; 做相关运算 (不翻转核)。
/// `divisor` 为归一化除数, `bias` 为结果偏移。
pub fn convolve_gray(g: &Gray, kernel: &[f32], kw: usize, kh: usize, divisor: f32, bias: f32, border: Border) -> Gray {
    assert!(kw % 2 == 1 && kh % 2 == 1 && kernel.len() == kw * kh);
    let (rx, ry) = ((kw / 2) as isize, (kh / 2) as isize);
    let mut out = Gray::new(g.width, g.height, 0.0);
    for y in 0..g.height {
        for x in 0..g.width {
            let mut s = 0.0;
            for j in 0..kh {
                for i in 0..kw {
                    s += kernel[j * kw + i] * g.get_border(x as isize + i as isize - rx, y as isize + j as isize - ry, border);
                }
            }
            out.set(x, y, s / divisor + bias);
        }
    }
    out
}

/// 对 RGB 三通道做卷积, alpha 保持不变。
pub fn convolve(img: &Image, kernel: &[f32], kw: usize, kh: usize, divisor: f32, bias: f32, border: Border) -> Image {
    let ch: Vec<Gray> = (0..3).map(|c| convolve_gray(&img.channel(c), kernel, kw, kh, divisor, bias, border)).collect();
    merge(img, &ch)
}

fn merge(src: &Image, ch: &[Gray]) -> Image {
    let mut out = src.clone();
    for i in 0..src.width * src.height {
        for c in 0..3 {
            out.data[i * 4 + c] = clamp_u8(ch[c].data[i]);
        }
    }
    out
}

/// 一维高斯核, 半径 `r = ceil(3σ)`。
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil().max(1.0) as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// 可分离卷积 (先水平后垂直)。
pub fn separable_gray(g: &Gray, kx: &[f32], ky: &[f32], border: Border) -> Gray {
    let (rx, ry) = ((kx.len() / 2) as isize, (ky.len() / 2) as isize);
    let mut tmp = Gray::new(g.width, g.height, 0.0);
    for y in 0..g.height {
        for x in 0..g.width {
            let s: f32 = kx.iter().enumerate().map(|(i, k)| k * g.get_border(x as isize + i as isize - rx, y as isize, border)).sum();
            tmp.set(x, y, s);
        }
    }
    let mut out = Gray::new(g.width, g.height, 0.0);
    for y in 0..g.height {
        for x in 0..g.width {
            let s: f32 = ky.iter().enumerate().map(|(j, k)| k * tmp.get_border(x as isize, y as isize + j as isize - ry, border)).sum();
            out.set(x, y, s);
        }
    }
    out
}

/// 高斯模糊 (灰度)。
pub fn gaussian_blur_gray(g: &Gray, sigma: f32) -> Gray {
    if sigma <= 0.0 {
        return g.clone();
    }
    let k = gaussian_kernel(sigma);
    separable_gray(g, &k, &k, Border::Clamp)
}

/// 高斯模糊: `sigma` 为标准差 (像素), PS 中"半径"约等于 σ。
pub fn gaussian_blur(img: &Image, sigma: f32) -> Image {
    let ch: Vec<Gray> = (0..3).map(|c| gaussian_blur_gray(&img.channel(c), sigma)).collect();
    merge(img, &ch)
}

/// 方框模糊 (均值模糊), `radius` 为半径。
pub fn box_blur(img: &Image, radius: usize) -> Image {
    let n = 2 * radius + 1;
    let k = vec![1.0 / n as f32; n];
    let ch: Vec<Gray> = (0..3).map(|c| separable_gray(&img.channel(c), &k, &k, Border::Clamp)).collect();
    merge(img, &ch)
}

/// 动感模糊: 沿 `angle`(度) 方向平均 `distance` 像素。
pub fn motion_blur(img: &Image, angle: f32, distance: usize) -> Image {
    let (dx, dy) = (angle.to_radians().cos(), -angle.to_radians().sin());
    let n = distance.max(1);
    let mut out = img.clone();
    for y in 0..img.height {
        for x in 0..img.width {
            let mut s = [0.0f32; 3];
            for i in 0..n {
                let t = i as f32 - (n as f32 - 1.0) / 2.0;
                let p = img.get_border((x as f32 + dx * t).round() as isize, (y as f32 + dy * t).round() as isize, Border::Clamp);
                for c in 0..3 {
                    s[c] += p[c] as f32;
                }
            }
            let a = img.get(x, y)[3];
            out.set(x, y, [clamp_u8(s[0] / n as f32), clamp_u8(s[1] / n as f32), clamp_u8(s[2] / n as f32), a]);
        }
    }
    out
}

/// 径向模糊。`spin=true` 为旋转模糊 (amount 为总角度,度), 否则为缩放模糊 (amount 为缩放百分比)。
pub fn radial_blur(img: &Image, center: (f32, f32), amount: f32, spin: bool, samples: usize) -> Image {
    let n = samples.max(2);
    let mut out = img.clone();
    for y in 0..img.height {
        for x in 0..img.width {
            let (dx, dy) = (x as f32 - center.0, y as f32 - center.1);
            let mut s = [0.0f32; 3];
            for i in 0..n {
                let t = i as f32 / (n - 1) as f32 - 0.5;
                let (sx, sy) = if spin {
                    let a = (amount * t).to_radians();
                    (center.0 + dx * a.cos() - dy * a.sin(), center.1 + dx * a.sin() + dy * a.cos())
                } else {
                    let k = 1.0 + amount / 100.0 * t;
                    (center.0 + dx * k, center.1 + dy * k)
                };
                let p = crate::transform::sample_bilinear(img, sx, sy, Border::Clamp);
                for c in 0..3 {
                    s[c] += p[c];
                }
            }
            let a = img.get(x, y)[3];
            out.set(x, y, [clamp_u8(s[0] / n as f32), clamp_u8(s[1] / n as f32), clamp_u8(s[2] / n as f32), a]);
        }
    }
    out
}

/// 锐化 (3x3 拉普拉斯锐化核), `amount` 为强度。
pub fn sharpen(img: &Image, amount: f32) -> Image {
    let k = [0.0, -amount, 0.0, -amount, 1.0 + 4.0 * amount, -amount, 0.0, -amount, 0.0];
    convolve(img, &k, 3, 3, 1.0, 0.0, Border::Clamp)
}

/// USM 锐化: `out = src + amount*(src-blur)`, 仅当差值绝对值 >= `threshold` 时生效。
pub fn unsharp_mask(img: &Image, amount: f32, radius: f32, threshold: f32) -> Image {
    let blur = gaussian_blur(img, radius);
    let mut out = img.clone();
    for i in 0..img.width * img.height {
        for c in 0..3 {
            let (s, b) = (img.data[i * 4 + c] as f32, blur.data[i * 4 + c] as f32);
            let d = s - b;
            out.data[i * 4 + c] = if d.abs() >= threshold { clamp_u8(s + amount * d) } else { img.data[i * 4 + c] };
        }
    }
    out
}

/// 高反差保留: `src - blur + 128`。
pub fn high_pass(img: &Image, radius: f32) -> Image {
    let blur = gaussian_blur(img, radius);
    let mut out = img.clone();
    for i in 0..img.width * img.height {
        for c in 0..3 {
            out.data[i * 4 + c] = clamp_u8(img.data[i * 4 + c] as f32 - blur.data[i * 4 + c] as f32 + 128.0);
        }
    }
    out
}

/// 中值滤波 (去除椒盐噪声), `radius` 为窗口半径。
pub fn median(img: &Image, radius: usize) -> Image {
    let r = radius as isize;
    let mut out = img.clone();
    let mut buf: Vec<u8> = Vec::new();
    for y in 0..img.height {
        for x in 0..img.width {
            let mut res = img.get(x, y);
            for c in 0..3 {
                buf.clear();
                for j in -r..=r {
                    for i in -r..=r {
                        buf.push(img.get_border(x as isize + i, y as isize + j, Border::Clamp)[c]);
                    }
                }
                buf.sort_unstable();
                res[c] = buf[buf.len() / 2];
            }
            out.set(x, y, res);
        }
    }
    out
}

/// 双边滤波 (保边平滑, 类似 PS 「表面模糊」)。`sigma_space` 空间σ, `sigma_color` 颜色σ(0..255)。
pub fn bilateral(img: &Image, sigma_space: f32, sigma_color: f32) -> Image {
    let r = (sigma_space * 2.0).ceil().max(1.0) as isize;
    let mut out = img.clone();
    for y in 0..img.height {
        for x in 0..img.width {
            let c0 = img.get(x, y);
            let (mut s, mut ws) = ([0.0f32; 3], 0.0f32);
            for j in -r..=r {
                for i in -r..=r {
                    let p = img.get_border(x as isize + i, y as isize + j, Border::Clamp);
                    let dc: f32 = (0..3).map(|c| (p[c] as f32 - c0[c] as f32).powi(2)).sum();
                    let w = (-((i * i + j * j) as f32) / (2.0 * sigma_space * sigma_space) - dc / (2.0 * sigma_color * sigma_color * 3.0)).exp();
                    for c in 0..3 {
                        s[c] += w * p[c] as f32;
                    }
                    ws += w;
                }
            }
            out.set(x, y, [clamp_u8(s[0] / ws), clamp_u8(s[1] / ws), clamp_u8(s[2] / ws), c0[3]]);
        }
    }
    out
}

/// 浮雕: `angle` 光照方向(度), `height` 浮雕强度。结果以 128 为中性灰。
pub fn emboss(img: &Image, angle: f32, height: f32) -> Image {
    let g = img.to_gray();
    let (dx, dy) = (angle.to_radians().cos(), -angle.to_radians().sin());
    let mut out = Gray::new(g.width, g.height, 0.0);
    for y in 0..g.height as isize {
        for x in 0..g.width as isize {
            let a = crate::transform::sample_gray_bilinear(&g, x as f32 + dx, y as f32 + dy);
            let b = crate::transform::sample_gray_bilinear(&g, x as f32 - dx, y as f32 - dy);
            out.set(x as usize, y as usize, 128.0 + (a - b) * height);
        }
    }
    let mut res = out.to_image();
    for i in 0..img.width * img.height {
        res.data[i * 4 + 3] = img.data[i * 4 + 3];
    }
    res
}

/// 添加噪点。`gaussian=true` 为高斯分布, 否则均匀分布; `mono` 为单色噪点。`amount` 为标准差 (0..255)。
pub fn add_noise(img: &Image, amount: f32, gaussian: bool, mono: bool, seed: u64) -> Image {
    let mut rng = Rng::new(seed);
    let draw = |rng: &mut Rng| if gaussian { rng.gauss() * amount } else { (rng.uniform() * 2.0 - 1.0) * amount * 1.732 };
    let mut out = img.clone();
    for p in out.data.chunks_exact_mut(4) {
        let n0 = draw(&mut rng);
        for c in 0..3 {
            let n = if mono { n0 } else { draw(&mut rng) };
            p[c] = clamp_u8(p[c] as f32 + n);
        }
    }
    out
}

/// 马赛克/像素化: 块边长 `size`。
pub fn pixelate(img: &Image, size: usize) -> Image {
    let s = size.max(1);
    let mut out = img.clone();
    for by in (0..img.height).step_by(s) {
        for bx in (0..img.width).step_by(s) {
            let (x1, y1) = ((bx + s).min(img.width), (by + s).min(img.height));
            let mut sum = [0u32; 4];
            for y in by..y1 {
                for x in bx..x1 {
                    let p = img.get(x, y);
                    for c in 0..4 {
                        sum[c] += p[c] as u32;
                    }
                }
            }
            let n = ((x1 - bx) * (y1 - by)) as u32;
            let avg = [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8, (sum[3] / n) as u8];
            for y in by..y1 {
                for x in bx..x1 {
                    out.set(x, y, avg);
                }
            }
        }
    }
    out
}

/// 油画效果 (Kuwahara 之外的强度直方图法): `radius` 半径, `levels` 强度分级数。
pub fn oil_paint(img: &Image, radius: usize, levels: usize) -> Image {
    let r = radius as isize;
    let lv = levels.clamp(2, 256);
    let mut out = img.clone();
    for y in 0..img.height {
        for x in 0..img.width {
            let mut cnt = vec![0u32; lv];
            let mut sum = vec![[0u32; 3]; lv];
            for j in -r..=r {
                for i in -r..=r {
                    let p = img.get_border(x as isize + i, y as isize + j, Border::Clamp);
                    let l = ((p[0] as u32 + p[1] as u32 + p[2] as u32) / 3) as usize * lv / 256;
                    cnt[l] += 1;
                    for c in 0..3 {
                        sum[l][c] += p[c] as u32;
                    }
                }
            }
            let best = (0..lv).max_by_key(|&i| cnt[i]).unwrap();
            let n = cnt[best].max(1);
            out.set(x, y, [(sum[best][0] / n) as u8, (sum[best][1] / n) as u8, (sum[best][2] / n) as u8, img.get(x, y)[3]]);
        }
    }
    out
}

/// 彩色半调/分色: 对灰度做 `cell` 大小的网点, 网点半径随亮度变化。输出黑底白点反相为白底黑点。
pub fn halftone(img: &Image, cell: usize) -> Image {
    let g = img.to_gray();
    let c = cell.max(2) as f32;
    let mut out = Image::new(img.width, img.height, [255, 255, 255, 255]);
    for y in 0..img.height {
        for x in 0..img.width {
            let (cx, cy) = (((x as f32 / c).floor() + 0.5) * c, ((y as f32 / c).floor() + 0.5) * c);
            let v = g.get((cx as usize).min(img.width - 1), (cy as usize).min(img.height - 1)) / 255.0;
            let rad = (1.0 - v).sqrt() * c * 0.7071;
            if ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() <= rad {
                out.set(x, y, [0, 0, 0, 255]);
            }
        }
    }
    out
}

/// 查找边缘风格化 / 素描: 边缘暗线 + 白底 (彩色铅笔效果)。`strength` 控制线条深浅。
pub fn sketch(img: &Image, sigma: f32, strength: f32) -> Image {
    let g = img.to_gray();
    let inv = Gray { width: g.width, height: g.height, data: g.data.iter().map(|v| 255.0 - v).collect() };
    let blur = gaussian_blur_gray(&inv, sigma);
    let mut out = g.clone();
    for i in 0..out.data.len() {
        // 颜色减淡混合: a / (1 - b)
        let a = g.data[i];
        let b = blur.data[i] / 255.0;
        out.data[i] = (a / (1.0 - b * strength).max(0.01)).min(255.0);
    }
    out.to_image()
}

/// 去斑 (轻量降噪): 3x3 高斯平滑。
pub fn despeckle(img: &Image) -> Image {
    let k = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    convolve(img, &k, 3, 3, 16.0, 0.0, Border::Clamp)
}

/// 灰度方框模糊 (半径 `radius`)。
pub fn box_blur_gray(g: &Gray, radius: usize) -> Gray {
    let n = 2 * radius + 1;
    let k = vec![1.0 / n as f32; n];
    separable_gray(g, &k, &k, Border::Clamp)
}
