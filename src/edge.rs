//! 边缘检测 (对应 Photoshop「查找边缘」「照亮边缘」及专业视觉库的经典算子)。

use crate::filter::{convolve_gray, gaussian_blur_gray};
use crate::image::{Border, Gray, Image};

/// 梯度算子。
#[derive(Clone, Copy, Debug)]
pub enum GradientOperator {
    /// 2x2 Roberts 交叉 (噪声敏感, 边缘细)。
    Roberts,
    /// 3x3 Prewitt (均值平滑)。
    Prewitt,
    /// 3x3 Sobel (高斯加权平滑, 最常用)。
    Sobel,
    /// 3x3 Scharr (旋转对称性更好, 角度更精确)。
    Scharr,
}

/// 梯度结果。
pub struct Gradient {
    pub gx: Gray,
    pub gy: Gray,
    /// 幅值 `sqrt(gx²+gy²)`。
    pub magnitude: Gray,
    /// 方向 (弧度, -π..π), 指向灰度上升方向。
    pub direction: Gray,
}

/// 计算梯度 (灰度输入)。
pub fn gradient(g: &Gray, op: GradientOperator) -> Gradient {
    let (kx, ky, k): (Vec<f32>, Vec<f32>, usize) = match op {
        GradientOperator::Roberts => (vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
        GradientOperator::Prewitt => (vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3),
        GradientOperator::Sobel => (vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], 3),
        GradientOperator::Scharr => (vec![-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0], vec![-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0], 3),
    };
    let gx = convolve_gray(g, &kx, k, k, 1.0, 0.0, Border::Clamp);
    let gy = convolve_gray(g, &ky, k, k, 1.0, 0.0, Border::Clamp);
    let mut mag = Gray::new(g.width, g.height, 0.0);
    let mut dir = Gray::new(g.width, g.height, 0.0);
    for i in 0..g.data.len() {
        mag.data[i] = (gx.data[i].powi(2) + gy.data[i].powi(2)).sqrt();
        dir.data[i] = gy.data[i].atan2(gx.data[i]);
    }
    Gradient { gx, gy, magnitude: mag, direction: dir }
}

/// 梯度边缘图: 幅值归一化到 0..255, 超过 `threshold`(0..255, 0 表示不二值化) 则二值化。
pub fn gradient_edges(img: &Image, op: GradientOperator, blur_sigma: f32, threshold: f32) -> Gray {
    let g = gaussian_blur_gray(&img.to_gray(), blur_sigma);
    let m = gradient(&g, op).magnitude.normalize();
    if threshold > 0.0 {
        Gray { width: m.width, height: m.height, data: m.data.iter().map(|&v| if v >= threshold { 255.0 } else { 0.0 }).collect() }
    } else {
        m
    }
}

/// 拉普拉斯算子 (二阶导), `eight_neighbor` 为 8 邻域核。输出取绝对值并归一化。
pub fn laplacian(img: &Image, eight_neighbor: bool) -> Gray {
    let k: [f32; 9] = if eight_neighbor { [1.0, 1.0, 1.0, 1.0, -8.0, 1.0, 1.0, 1.0, 1.0] } else { [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0] };
    let r = convolve_gray(&img.to_gray(), &k, 3, 3, 1.0, 0.0, Border::Clamp);
    Gray { width: r.width, height: r.height, data: r.data.iter().map(|v| v.abs()).collect() }.normalize()
}

/// 高斯-拉普拉斯 (LoG / Marr–Hildreth): 先高斯平滑再取拉普拉斯, 返回零交叉边缘图 (0/255)。
/// `min_response` 过滤弱零交叉。
pub fn laplacian_of_gaussian(img: &Image, sigma: f32, min_response: f32) -> Gray {
    let g = gaussian_blur_gray(&img.to_gray(), sigma);
    let l = convolve_gray(&g, &[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3, 1.0, 0.0, Border::Clamp);
    let mut out = Gray::new(l.width, l.height, 0.0);
    for y in 0..l.height.saturating_sub(1) {
        for x in 0..l.width.saturating_sub(1) {
            let v = l.get(x, y);
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                let n = l.get(nx, ny);
                if v * n < 0.0 && (v - n).abs() >= min_response {
                    out.set(x, y, 255.0);
                }
            }
        }
    }
    out
}

/// 高斯差分 (DoG) 边缘/带通: `blur(σ1) - blur(σ2)`, 结果以 128 为中心。常用于 XDoG 素描风格与斑点检测。
pub fn difference_of_gaussians(img: &Image, sigma1: f32, sigma2: f32) -> Gray {
    let g = img.to_gray();
    let (a, b) = (gaussian_blur_gray(&g, sigma1), gaussian_blur_gray(&g, sigma2));
    Gray { width: g.width, height: g.height, data: a.data.iter().zip(&b.data).map(|(x, y)| (x - y) * 4.0 + 128.0).collect() }
}

/// Canny 参数。
#[derive(Clone, Copy, Debug)]
pub struct CannyParams {
    /// 预平滑高斯 σ (0 表示不平滑)。
    pub sigma: f32,
    /// 低阈值 (梯度幅值, Sobel 未归一)。
    pub low: f32,
    /// 高阈值。
    pub high: f32,
    /// 梯度算子。
    pub operator: GradientOperator,
}

impl Default for CannyParams {
    fn default() -> Self {
        CannyParams { sigma: 1.4, low: 40.0, high: 100.0, operator: GradientOperator::Sobel }
    }
}

/// Canny 边缘检测: 高斯平滑 → 梯度 → 非极大值抑制 → 双阈值 + 滞后连接。返回 0/255 图。
pub fn canny(img: &Image, p: CannyParams) -> Gray {
    let g = gaussian_blur_gray(&img.to_gray(), p.sigma);
    let gr = gradient(&g, p.operator);
    let (w, h) = (g.width, g.height);
    let mut nms = Gray::new(w, h, 0.0);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let m = gr.magnitude.get(x, y);
            let mut a = gr.direction.get(x, y).to_degrees();
            if a < 0.0 {
                a += 180.0;
            }
            // 量化梯度方向到 4 个方向, 比较梯度方向两侧邻居
            let (dx, dy): (isize, isize) = if !(22.5..157.5).contains(&a) { (1, 0) } else if a < 67.5 { (1, 1) } else if a < 112.5 { (0, 1) } else { (-1, 1) };
            let n1 = gr.magnitude.get((x as isize + dx) as usize, (y as isize + dy) as usize);
            let n2 = gr.magnitude.get((x as isize - dx) as usize, (y as isize - dy) as usize);
            if m >= n1 && m >= n2 {
                nms.set(x, y, m);
            }
        }
    }
    let mut out = Gray::new(w, h, 0.0);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if nms.get(x, y) >= p.high {
                out.set(x, y, 255.0);
                stack.push((x, y));
            }
        }
    }
    while let Some((x, y)) = stack.pop() {
        for j in -1..=1isize {
            for i in -1..=1isize {
                let (nx, ny) = (x as isize + i, y as isize + j);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if out.get(nx, ny) == 0.0 && nms.get(nx, ny) >= p.low {
                    out.set(nx, ny, 255.0);
                    stack.push((nx, ny));
                }
            }
        }
    }
    out
}

/// PS「查找边缘」风格: 白底, 边缘以彩色深线表示 (逐通道 Sobel 幅值反相)。
pub fn find_edges_color(img: &Image) -> Image {
    let mut out = img.clone();
    for c in 0..3 {
        let m = gradient(&img.channel(c), GradientOperator::Sobel).magnitude;
        for i in 0..m.data.len() {
            out.data[i * 4 + c] = (255.0 - m.data[i]).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// 霍夫直线检测 (标准 Hough): 对二值边缘图投票, 返回 `(rho, theta_rad, votes)` 按票数降序。
/// 直线方程 `x cosθ + y sinθ = ρ`。`min_votes` 为最小票数, `theta_steps` 为角度分辨率 (如 180)。
pub fn hough_lines(edges: &Gray, theta_steps: usize, min_votes: u32) -> Vec<(f32, f32, u32)> {
    let (w, h) = (edges.width as f32, edges.height as f32);
    let diag = (w * w + h * h).sqrt().ceil() as isize;
    let nr = (2 * diag + 1) as usize;
    let mut acc = vec![0u32; nr * theta_steps];
    let trig: Vec<(f32, f32)> = (0..theta_steps).map(|t| (t as f32 * std::f32::consts::PI / theta_steps as f32).sin_cos()).collect();
    for y in 0..edges.height {
        for x in 0..edges.width {
            if edges.get(x, y) > 127.0 {
                for (t, (s, c)) in trig.iter().enumerate() {
                    let r = (x as f32 * c + y as f32 * s).round() as isize + diag;
                    acc[r as usize * theta_steps + t] += 1;
                }
            }
        }
    }
    let mut res = Vec::new();
    for r in 0..nr {
        for t in 0..theta_steps {
            let v = acc[r * theta_steps + t];
            if v >= min_votes {
                // 3x3 局部极大
                let mut is_max = true;
                'o: for dr in -1..=1isize {
                    for dt in -1..=1isize {
                        let (rr, tt) = (r as isize + dr, t as isize + dt);
                        if (dr, dt) != (0, 0) && rr >= 0 && rr < nr as isize && tt >= 0 && tt < theta_steps as isize && acc[rr as usize * theta_steps + tt as usize] > v {
                            is_max = false;
                            break 'o;
                        }
                    }
                }
                if is_max {
                    res.push(((r as isize - diag) as f32, t as f32 * std::f32::consts::PI / theta_steps as f32, v));
                }
            }
        }
    }
    res.sort_by(|a, b| b.2.cmp(&a.2));
    res
}
