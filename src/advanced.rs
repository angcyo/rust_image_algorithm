//! 进阶算法: Kuwahara、非局部均值降噪、镜头模糊、辉光、去雾、卡通化、Harris 角点。

use crate::filter::{gaussian_blur, gaussian_blur_gray};
use crate::image::{clamp_u8, Border, Gray, Image};

/// Kuwahara 滤波 (保边油画/水彩): 把窗口分成 4 个 (r+1)² 象限, 取方差最小象限的均值。`radius` >= 1。
pub fn kuwahara(img: &Image, radius: usize) -> Image {
    let r = radius.max(1) as isize;
    let gray = img.to_gray();
    let mut out = img.clone();
    for y in 0..img.height as isize {
        for x in 0..img.width as isize {
            let mut best = (f32::MAX, [0.0f32; 3]);
            for (ox, oy) in [(-r, -r), (0, -r), (-r, 0), (0, 0)] {
                let (mut s, mut s2, mut n) = (0.0f32, 0.0f32, 0.0f32);
                let mut col = [0.0f32; 3];
                for j in 0..=r {
                    for i in 0..=r {
                        let (px, py) = (x + ox + i, y + oy + j);
                        let v = gray.get_border(px, py, Border::Clamp);
                        let p = img.get_border(px, py, Border::Clamp);
                        s += v;
                        s2 += v * v;
                        n += 1.0;
                        for c in 0..3 {
                            col[c] += p[c] as f32;
                        }
                    }
                }
                let var = s2 / n - (s / n).powi(2);
                if var < best.0 {
                    best = (var, [col[0] / n, col[1] / n, col[2] / n]);
                }
            }
            let a = img.get(x as usize, y as usize)[3];
            out.set(x as usize, y as usize, [clamp_u8(best.1[0]), clamp_u8(best.1[1]), clamp_u8(best.1[2]), a]);
        }
    }
    out
}

/// 非局部均值降噪 (NLM): 对比 `patch_radius` 大小的邻域相似度, 在 `search_radius` 内加权平均。
/// `h` 为滤波强度 (约等于噪声标准差 0.6~1 倍)。复杂度 O(N·S²·P²), 适合小图。
pub fn non_local_means(img: &Image, patch_radius: usize, search_radius: usize, h: f32) -> Image {
    let (pr, sr) = (patch_radius as isize, search_radius as isize);
    let mut out = img.clone();
    let np = ((2 * pr + 1) * (2 * pr + 1) * 3) as f32;
    for y in 0..img.height as isize {
        for x in 0..img.width as isize {
            let (mut acc, mut wsum) = ([0.0f32; 3], 0.0f32);
            for sy in -sr..=sr {
                for sx in -sr..=sr {
                    let mut d = 0.0;
                    for j in -pr..=pr {
                        for i in -pr..=pr {
                            let a = img.get_border(x + i, y + j, Border::Reflect);
                            let b = img.get_border(x + sx + i, y + sy + j, Border::Reflect);
                            for c in 0..3 {
                                d += (a[c] as f32 - b[c] as f32).powi(2);
                            }
                        }
                    }
                    let w = (-(d / np) / (h * h)).exp();
                    let p = img.get_border(x + sx, y + sy, Border::Reflect);
                    for c in 0..3 {
                        acc[c] += w * p[c] as f32;
                    }
                    wsum += w;
                }
            }
            let a = img.get(x as usize, y as usize)[3];
            out.set(x as usize, y as usize, [clamp_u8(acc[0] / wsum), clamp_u8(acc[1] / wsum), clamp_u8(acc[2] / wsum), a]);
        }
    }
    out
}

/// 镜头模糊 (散景): 以圆盘核 `radius` 卷积。`highlight_boost`>0 时先增强高光 (gamma) 以得到明显光斑。
pub fn lens_blur(img: &Image, radius: usize, highlight_boost: f32) -> Image {
    let r = radius.max(1) as isize;
    let w = |p: [u8; 4]| 1.0 + highlight_boost * (p[0] as f32 + p[1] as f32 + p[2] as f32) / 765.0 * 8.0;
    let mut out = img.clone();
    for y in 0..img.height as isize {
        for x in 0..img.width as isize {
            let (mut acc, mut ws) = ([0.0f32; 3], 0.0f32);
            for j in -r..=r {
                for i in -r..=r {
                    if i * i + j * j <= r * r {
                        let p = img.get_border(x + i, y + j, Border::Clamp);
                        let k = w(p);
                        for c in 0..3 {
                            acc[c] += k * p[c] as f32;
                        }
                        ws += k;
                    }
                }
            }
            let a = img.get(x as usize, y as usize)[3];
            out.set(x as usize, y as usize, [clamp_u8(acc[0] / ws), clamp_u8(acc[1] / ws), clamp_u8(acc[2] / ws), a]);
        }
    }
    out
}

/// 辉光 / Bloom: 提取亮度高于 `threshold` 的部分, 高斯模糊后以滤色叠加, `intensity` 为强度。
pub fn bloom(img: &Image, threshold: f32, sigma: f32, intensity: f32) -> Image {
    let bright = img.map_pixels(|p| {
        let l = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        if l >= threshold { p } else { [0, 0, 0, p[3]] }
    });
    let blur = gaussian_blur(&bright, sigma);
    let mut out = img.clone();
    for i in 0..img.width * img.height {
        for c in 0..3 {
            let (b, s) = (img.data[i * 4 + c] as f32 / 255.0, blur.data[i * 4 + c] as f32 / 255.0 * intensity);
            out.data[i * 4 + c] = clamp_u8((1.0 - (1.0 - b) * (1.0 - s.min(1.0))) * 255.0);
        }
    }
    out
}

/// 暗通道先验去雾 (He et al.): `patch` 暗通道窗口半径, `omega`(~0.95) 保留少量雾以保持景深, `t0` 透射率下限。
pub fn dehaze(img: &Image, patch: usize, omega: f32, t0: f32) -> Image {
    let (w, h) = (img.width, img.height);
    let r = patch as isize;
    let min_rgb: Vec<f32> = img.data.chunks_exact(4).map(|p| p[0].min(p[1]).min(p[2]) as f32).collect();
    let dark_of = |v: &[f32]| {
        let mut d = vec![0.0f32; w * h];
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut m = f32::MAX;
                for j in -r..=r {
                    for i in -r..=r {
                        let (xx, yy) = ((x + i).clamp(0, w as isize - 1), (y + j).clamp(0, h as isize - 1));
                        m = m.min(v[yy as usize * w + xx as usize]);
                    }
                }
                d[y as usize * w + x as usize] = m;
            }
        }
        d
    };
    let dark = dark_of(&min_rgb);
    // 取暗通道最亮 0.1% 像素中原图最亮者作为大气光
    let mut idx: Vec<usize> = (0..w * h).collect();
    idx.sort_by(|&a, &b| dark[b].partial_cmp(&dark[a]).unwrap());
    let top = &idx[..(w * h / 1000).max(1)];
    let &best = top.iter().max_by(|&&a, &&b| min_rgb_sum(img, a).partial_cmp(&min_rgb_sum(img, b)).unwrap()).unwrap();
    let a_light = [img.data[best * 4] as f32, img.data[best * 4 + 1] as f32, img.data[best * 4 + 2] as f32];
    let norm: Vec<f32> = (0..w * h).map(|i| (0..3).map(|c| img.data[i * 4 + c] as f32 / a_light[c].max(1.0)).fold(f32::MAX, f32::min)).collect();
    let t: Vec<f32> = dark_of(&norm).iter().map(|d| (1.0 - omega * d).max(t0)).collect();
    let mut out = img.clone();
    for i in 0..w * h {
        for c in 0..3 {
            out.data[i * 4 + c] = clamp_u8((img.data[i * 4 + c] as f32 - a_light[c]) / t[i] + a_light[c]);
        }
    }
    out
}

fn min_rgb_sum(img: &Image, i: usize) -> f32 {
    img.data[i * 4] as f32 + img.data[i * 4 + 1] as f32 + img.data[i * 4 + 2] as f32
}

/// 卡通化: 双边平滑 + 色调分离 + Canny 黑色描边。
pub fn cartoon(img: &Image, levels: u32, edge_low: f32, edge_high: f32) -> Image {
    let smooth = crate::filter::bilateral(img, 3.0, 40.0);
    let mut out = crate::color::posterize(&smooth, levels);
    let e = crate::edge::canny(img, crate::edge::CannyParams { sigma: 1.2, low: edge_low, high: edge_high, ..Default::default() });
    for i in 0..e.data.len() {
        if e.data[i] > 0.0 {
            out.data[i * 4..i * 4 + 3].copy_from_slice(&[0, 0, 0]);
        }
    }
    out
}

/// Harris 角点检测: 响应 `R = det(M) - k·trace(M)²`, 窗口高斯加权 `sigma`, 返回 (x, y, 响应) 按响应降序, 做 3x3 非极大抑制。
/// `rel_threshold` 为相对最大响应的比例 (如 0.01)。
pub fn harris_corners(img: &Image, sigma: f32, k: f32, rel_threshold: f32) -> Vec<(usize, usize, f32)> {
    let g = gaussian_blur_gray(&img.to_gray(), 1.0);
    let gr = crate::edge::gradient(&g, crate::edge::GradientOperator::Sobel);
    let mut ixx = Gray::new(g.width, g.height, 0.0);
    let mut iyy = ixx.clone();
    let mut ixy = ixx.clone();
    for i in 0..g.data.len() {
        ixx.data[i] = gr.gx.data[i] * gr.gx.data[i];
        iyy.data[i] = gr.gy.data[i] * gr.gy.data[i];
        ixy.data[i] = gr.gx.data[i] * gr.gy.data[i];
    }
    let (ixx, iyy, ixy) = (gaussian_blur_gray(&ixx, sigma), gaussian_blur_gray(&iyy, sigma), gaussian_blur_gray(&ixy, sigma));
    let resp: Vec<f32> = (0..g.data.len()).map(|i| ixx.data[i] * iyy.data[i] - ixy.data[i].powi(2) - k * (ixx.data[i] + iyy.data[i]).powi(2)).collect();
    let mx = resp.iter().cloned().fold(0.0f32, f32::max);
    let mut pts = Vec::new();
    for y in 1..g.height.saturating_sub(1) {
        for x in 1..g.width.saturating_sub(1) {
            let v = resp[y * g.width + x];
            if v > mx * rel_threshold && (-1..=1isize).all(|j| (-1..=1isize).all(|i| resp[(y as isize + j) as usize * g.width + (x as isize + i) as usize] <= v)) {
                pts.push((x, y, v));
            }
        }
    }
    pts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    pts
}

/// 在图像上绘制十字/方块标记点。
pub fn draw_points(img: &mut Image, pts: &[(usize, usize)], color: [u8; 4], radius: usize) {
    let r = radius as isize;
    for &(x, y) in pts {
        for j in -r..=r {
            for i in -r..=r {
                let (px, py) = (x as isize + i, y as isize + j);
                if px >= 0 && py >= 0 && (px as usize) < img.width && (py as usize) < img.height && (i.abs() == r || j.abs() == r) {
                    img.set(px as usize, py as usize, color);
                }
            }
        }
    }
}

/// 霍夫圆检测 (梯度方向投票): 在 `[r_min,r_max]` 内返回 `(cx, cy, r, 票数)`。`edges` 为 Canny 结果, `gray` 用于梯度方向。
pub fn hough_circles(edges: &Gray, r_min: usize, r_max: usize, min_votes: u32) -> Vec<(usize, usize, usize, u32)> {
    let (w, h) = (edges.width, edges.height);
    let nr = r_max - r_min + 1;
    let mut acc = vec![0u32; w * h * nr];
    let steps = 90;
    for y in 0..h {
        for x in 0..w {
            if edges.get(x, y) < 128.0 {
                continue;
            }
            for ri in 0..nr {
                let r = (r_min + ri) as f32;
                for t in 0..steps {
                    let a = t as f32 / steps as f32 * std::f32::consts::TAU;
                    let (cx, cy) = ((x as f32 + r * a.cos()).round() as isize, (y as f32 + r * a.sin()).round() as isize);
                    if cx >= 0 && cy >= 0 && (cx as usize) < w && (cy as usize) < h {
                        acc[(cy as usize * w + cx as usize) * nr + ri] += 1;
                    }
                }
            }
        }
    }
    let mut res = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for ri in 0..nr {
                let v = acc[(y * w + x) * nr + ri];
                if v >= min_votes {
                    let local_max = (-1..=1isize).all(|j| (-1..=1isize).all(|i| {
                        let (xx, yy) = (x as isize + i, y as isize + j);
                        xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize || acc[(yy as usize * w + xx as usize) * nr + ri] <= v
                    }));
                    if local_max {
                        res.push((x, y, r_min + ri, v));
                    }
                }
            }
        }
    }
    res.sort_by(|a, b| b.3.cmp(&a.3));
    res
}
