//! 二值化、形态学、连通域与轮廓提取。
//!
//! 二值图用 [`Gray`] 表示: 前景 255, 背景 0。

use crate::filter::{box_blur_gray, gaussian_blur_gray};
use crate::image::{Border, Gray, Image};

/// 固定阈值二值化。`invert` 为反相。
pub fn binarize(g: &Gray, threshold: f32, invert: bool) -> Gray {
    Gray { width: g.width, height: g.height, data: g.data.iter().map(|&v| if (v >= threshold) != invert { 255.0 } else { 0.0 }).collect() }
}

/// Otsu 自动阈值 (最大化类间方差), 返回 0..=255 的阈值。
pub fn otsu_threshold(g: &Gray) -> f32 {
    let mut h = [0f64; 256];
    for &v in &g.data {
        h[v.round().clamp(0.0, 255.0) as usize] += 1.0;
    }
    let total: f64 = h.iter().sum();
    let sum_all: f64 = h.iter().enumerate().map(|(i, c)| i as f64 * c).sum();
    let (mut wb, mut sb, mut best, mut t) = (0.0, 0.0, -1.0, 0);
    for i in 0..256 {
        wb += h[i];
        if wb == 0.0 {
            continue;
        }
        let wf = total - wb;
        if wf == 0.0 {
            break;
        }
        sb += i as f64 * h[i];
        let (mb, mf) = (sb / wb, (sum_all - sb) / wf);
        let var = wb * wf * (mb - mf).powi(2);
        if var > best {
            best = var;
            t = i;
        }
    }
    t as f32 + 0.5
}

/// 三角法阈值: 适合单峰直方图。
pub fn triangle_threshold(g: &Gray) -> f32 {
    let mut h = [0f32; 256];
    for &v in &g.data {
        h[v.round().clamp(0.0, 255.0) as usize] += 1.0;
    }
    let (mut mn, mut mx) = (0usize, 0usize);
    for i in 0..256 {
        if h[i] > 0.0 {
            mn = i;
            break;
        }
    }
    for i in (0..256).rev() {
        if h[i] > 0.0 {
            mx = i;
            break;
        }
    }
    let peak = (0..256).max_by(|&a, &b| h[a].partial_cmp(&h[b]).unwrap()).unwrap();
    // 峰值远离的一端作为线段另一端点
    let end = if peak - mn > mx - peak { mn } else { mx };
    let (x1, y1, x2, y2) = (peak as f32, h[peak], end as f32, h[end]);
    let (lo, hi) = (peak.min(end), peak.max(end));
    let mut best = (0.0, peak);
    for i in lo..=hi {
        let d = ((y2 - y1) * i as f32 - (x2 - x1) * h[i] + x2 * y1 - y2 * x1).abs() / ((y2 - y1).powi(2) + (x2 - x1).powi(2)).sqrt().max(1e-6);
        if d > best.0 {
            best = (d, i);
        }
    }
    best.1 as f32
}

/// 自适应阈值 (局部阈值): `mean` 窗口均值或高斯加权; `block_radius` 窗口半径, `c` 为从局部均值减去的常数。
pub fn adaptive_threshold(g: &Gray, block_radius: usize, c: f32, gaussian: bool) -> Gray {
    let local = if gaussian { gaussian_blur_gray(g, block_radius as f32 / 2.0 + 0.5) } else { box_blur_gray(g, block_radius) };
    Gray { width: g.width, height: g.height, data: g.data.iter().zip(&local.data).map(|(v, m)| if *v > m - c { 255.0 } else { 0.0 }).collect() }
}

/// 结构元素形状。
#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    /// (2r+1)² 方形。
    Rect(usize),
    /// 十字形。
    Cross(usize),
    /// 椭圆/圆形。
    Ellipse(usize),
}

fn offsets(k: Kernel) -> Vec<(isize, isize)> {
    let r = match k {
        Kernel::Rect(r) | Kernel::Cross(r) | Kernel::Ellipse(r) => r as isize,
    };
    let mut v = Vec::new();
    for j in -r..=r {
        for i in -r..=r {
            let ok = match k {
                Kernel::Rect(_) => true,
                Kernel::Cross(_) => i == 0 || j == 0,
                Kernel::Ellipse(_) => i * i + j * j <= r * r,
            };
            if ok {
                v.push((i, j));
            }
        }
    }
    v
}

fn morph(g: &Gray, k: Kernel, max: bool) -> Gray {
    let offs = offsets(k);
    let mut out = Gray::new(g.width, g.height, 0.0);
    for y in 0..g.height as isize {
        for x in 0..g.width as isize {
            let mut v = if max { f32::MIN } else { f32::MAX };
            for &(i, j) in &offs {
                let p = g.get_border(x + i, y + j, Border::Clamp);
                v = if max { v.max(p) } else { v.min(p) };
            }
            out.set(x as usize, y as usize, v);
        }
    }
    out
}

/// 膨胀 (局部最大值)。灰度/二值通用。
pub fn dilate(g: &Gray, k: Kernel) -> Gray {
    morph(g, k, true)
}
/// 腐蚀 (局部最小值)。
pub fn erode(g: &Gray, k: Kernel) -> Gray {
    morph(g, k, false)
}
/// 开运算 (先腐蚀后膨胀): 去除小亮点、断开细连接。
pub fn open(g: &Gray, k: Kernel) -> Gray {
    dilate(&erode(g, k), k)
}
/// 闭运算 (先膨胀后腐蚀): 填充小孔洞、连接断裂。
pub fn close(g: &Gray, k: Kernel) -> Gray {
    erode(&dilate(g, k), k)
}
/// 形态学梯度 (膨胀 - 腐蚀): 提取边界。
pub fn morph_gradient(g: &Gray, k: Kernel) -> Gray {
    sub(&dilate(g, k), &erode(g, k))
}
/// 顶帽 (原图 - 开运算): 提取比结构元素小的亮细节, 用于背景不均匀校正。
pub fn top_hat(g: &Gray, k: Kernel) -> Gray {
    sub(g, &open(g, k))
}
/// 黑帽 (闭运算 - 原图): 提取暗细节。
pub fn black_hat(g: &Gray, k: Kernel) -> Gray {
    sub(&close(g, k), g)
}

fn sub(a: &Gray, b: &Gray) -> Gray {
    Gray { width: a.width, height: a.height, data: a.data.iter().zip(&b.data).map(|(x, y)| x - y).collect() }
}

/// 连通域标记 (8 邻域, 两遍扫描 + 并查集)。返回 `(标签图, 区域数)`, 标签从 1 开始, 背景 0。
pub fn connected_components(bin: &Gray) -> (Vec<u32>, u32) {
    let (w, h) = (bin.width, bin.height);
    let mut lab = vec![0u32; w * h];
    let mut parent: Vec<u32> = vec![0];
    fn find(p: &mut Vec<u32>, mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for y in 0..h {
        for x in 0..w {
            if bin.get(x, y) < 128.0 {
                continue;
            }
            let mut nb = Vec::with_capacity(4);
            for (dx, dy) in [(-1isize, 0isize), (-1, -1), (0, -1), (1, -1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < w && lab[ny as usize * w + nx as usize] > 0 {
                    nb.push(lab[ny as usize * w + nx as usize]);
                }
            }
            if nb.is_empty() {
                let id = parent.len() as u32;
                parent.push(id);
                lab[y * w + x] = id;
            } else {
                let m = *nb.iter().min().unwrap();
                lab[y * w + x] = m;
                for &n in &nb {
                    let (a, b) = (find(&mut parent, n), find(&mut parent, m));
                    if a != b {
                        parent[a.max(b) as usize] = a.min(b);
                    }
                }
            }
        }
    }
    let mut remap = vec![0u32; parent.len()];
    let mut count = 0;
    for v in lab.iter_mut() {
        if *v > 0 {
            let r = find(&mut parent, *v) as usize;
            if remap[r] == 0 {
                count += 1;
                remap[r] = count;
            }
            *v = remap[r];
        }
    }
    (lab, count)
}

/// 去除面积小于 `min_area` 的连通域 (去噪点)。
pub fn remove_small_components(bin: &Gray, min_area: usize) -> Gray {
    let (lab, n) = connected_components(bin);
    let mut area = vec![0usize; n as usize + 1];
    for &l in &lab {
        area[l as usize] += 1;
    }
    Gray { width: bin.width, height: bin.height, data: lab.iter().map(|&l| if l > 0 && area[l as usize] >= min_area { 255.0 } else { 0.0 }).collect() }
}

/// 填充孔洞: 从边界泛洪背景, 未被到达的背景即孔洞。
pub fn fill_holes(bin: &Gray) -> Gray {
    let (w, h) = (bin.width, bin.height);
    let mut seen = vec![false; w * h];
    let mut st = Vec::new();
    for x in 0..w {
        st.push((x, 0));
        st.push((x, h - 1));
    }
    for y in 0..h {
        st.push((0, y));
        st.push((w - 1, y));
    }
    while let Some((x, y)) = st.pop() {
        if seen[y * w + x] || bin.get(x, y) >= 128.0 {
            continue;
        }
        seen[y * w + x] = true;
        if x > 0 { st.push((x - 1, y)); }
        if y > 0 { st.push((x, y - 1)); }
        if x + 1 < w { st.push((x + 1, y)); }
        if y + 1 < h { st.push((x, y + 1)); }
    }
    Gray { width: w, height: h, data: (0..w * h).map(|i| if seen[i] { 0.0 } else { 255.0 }).collect() }
}

/// 欧氏距离变换 (到最近背景像素的距离, 两遍 Felzenszwalb 精确算法)。
pub fn distance_transform(bin: &Gray) -> Gray {
    let (w, h) = (bin.width, bin.height);
    let inf = (w * w + h * h) as f32 + 1.0;
    let mut d: Vec<f32> = bin.data.iter().map(|&v| if v >= 128.0 { inf } else { 0.0 }).collect();
    fn dt1(f: &[f32]) -> Vec<f32> {
        let n = f.len();
        let (mut v, mut z) = (vec![0usize; n], vec![0f32; n + 1]);
        let mut k = 0;
        z[0] = f32::MIN;
        z[1] = f32::MAX;
        for q in 1..n {
            loop {
                let p = v[k];
                let s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * q as f32 - 2.0 * p as f32);
                if s <= z[k] && k > 0 {
                    k -= 1;
                } else {
                    if s <= z[k] {
                        v[0] = q;
                        z[1] = f32::MAX;
                    } else {
                        k += 1;
                        v[k] = q;
                        z[k] = s;
                        z[k + 1] = f32::MAX;
                    }
                    break;
                }
            }
        }
        let mut out = vec![0f32; n];
        k = 0;
        for q in 0..n {
            while z[k + 1] < q as f32 {
                k += 1;
            }
            out[q] = (q as f32 - v[k] as f32).powi(2) + f[v[k]];
        }
        out
    }
    for x in 0..w {
        let col: Vec<f32> = (0..h).map(|y| d[y * w + x]).collect();
        for (y, v) in dt1(&col).into_iter().enumerate() {
            d[y * w + x] = v;
        }
    }
    for y in 0..h {
        let r = dt1(&d[y * w..(y + 1) * w]);
        d[y * w..(y + 1) * w].copy_from_slice(&r);
    }
    Gray { width: w, height: h, data: d.into_iter().map(|v| v.sqrt().min(1e5)).collect() }
}

/// 轮廓 (像素点序列, 顺时针)。
pub type Contour = Vec<(usize, usize)>;

const NB: [(isize, isize); 8] = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)];

/// 提取所有连通域的**外轮廓** (Moore 邻域边界跟踪 + Jacob 终止条件, 8 连通)。
/// `min_length` 过滤过短轮廓。孔洞内轮廓不提取 (可对反相图再调用一次得到孔洞)。
pub fn find_contours(bin: &Gray, min_length: usize) -> Vec<Contour> {
    let (lab, n) = connected_components(bin);
    let (w, h) = (bin.width, bin.height);
    let mut started = vec![false; n as usize + 1];
    let mut res = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let l = lab[y * w + x] as usize;
            if l == 0 || started[l] {
                continue;
            }
            started[l] = true;
            let c = trace(&lab, w, h, l as u32, (x, y));
            if c.len() >= min_length {
                res.push(c);
            }
        }
    }
    res
}

fn trace(lab: &[u32], w: usize, h: usize, id: u32, s: (usize, usize)) -> Contour {
    let fg = |x: isize, y: isize| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && lab[y as usize * w + x as usize] == id;
    let mut contour = vec![s];
    let (mut cur, mut d) = (s, 0usize); // d: 回溯点相对当前点的方向索引, 起点从西侧(背景)进入
    let cap = w * h * 8 + 16;
    for _ in 0..cap {
        let mut found = None;
        for k in 1..=8 {
            let idx = (d + k) % 8;
            let (nx, ny) = (cur.0 as isize + NB[idx].0, cur.1 as isize + NB[idx].1);
            if fg(nx, ny) {
                found = Some((idx, (d + k - 1) % 8, (nx as usize, ny as usize)));
                break;
            }
        }
        let Some((_, prev, next)) = found else { break }; // 孤立像素
        // 回溯点位置 = cur + NB[prev]; 求其相对 next 的方向
        let bp = (cur.0 as isize + NB[prev].0 - next.0 as isize, cur.1 as isize + NB[prev].1 - next.1 as isize);
        d = NB.iter().position(|&o| o == bp).unwrap_or(0);
        cur = next;
        if cur == s && d == 0 {
            break;
        }
        contour.push(cur);
    }
    contour
}

/// Douglas–Peucker 折线简化, `epsilon` 为最大允许偏差 (像素)。`closed` 时按闭合轮廓处理。
pub fn simplify_contour(pts: &[(usize, usize)], epsilon: f32, closed: bool) -> Vec<(usize, usize)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    fn dp(p: &[(usize, usize)], eps: f32, out: &mut Vec<(usize, usize)>) {
        let (a, b) = (p[0], p[p.len() - 1]);
        let (dx, dy) = (b.0 as f32 - a.0 as f32, b.1 as f32 - a.1 as f32);
        let len = (dx * dx + dy * dy).sqrt();
        let (mut md, mut mi) = (0.0, 0);
        for (i, q) in p.iter().enumerate().take(p.len() - 1).skip(1) {
            let (qx, qy) = (q.0 as f32 - a.0 as f32, q.1 as f32 - a.1 as f32);
            let d = if len < 1e-6 { (qx * qx + qy * qy).sqrt() } else { (qx * dy - qy * dx).abs() / len };
            if d > md {
                md = d;
                mi = i;
            }
        }
        if md > eps {
            dp(&p[..=mi], eps, out);
            out.pop();
            dp(&p[mi..], eps, out);
        } else {
            out.push(a);
            out.push(b);
        }
    }
    let mut v = pts.to_vec();
    if closed {
        v.push(pts[0]);
    }
    let mut out = Vec::new();
    dp(&v, epsilon, &mut out);
    out.dedup();
    if closed && out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    out
}

/// 轮廓周长 (闭合折线长度)。
pub fn contour_perimeter(c: &[(usize, usize)]) -> f32 {
    (0..c.len()).map(|i| {
        let (a, b) = (c[i], c[(i + 1) % c.len()]);
        ((a.0 as f32 - b.0 as f32).powi(2) + (a.1 as f32 - b.1 as f32).powi(2)).sqrt()
    }).sum()
}

/// 轮廓面积 (鞋带公式)。
pub fn contour_area(c: &[(usize, usize)]) -> f32 {
    let mut s = 0.0;
    for i in 0..c.len() {
        let (a, b) = (c[i], c[(i + 1) % c.len()]);
        s += a.0 as f32 * b.1 as f32 - b.0 as f32 * a.1 as f32;
    }
    s.abs() / 2.0
}

/// 轮廓包围盒 `(x, y, w, h)`。
pub fn bounding_rect(c: &[(usize, usize)]) -> (usize, usize, usize, usize) {
    let (x0, x1) = c.iter().fold((usize::MAX, 0), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (y0, y1) = c.iter().fold((usize::MAX, 0), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
}

/// 在图像上绘制轮廓折线 (Bresenham), `thickness` 以邻域扩展实现。
pub fn draw_contours(img: &mut Image, contours: &[Contour], color: [u8; 4], thickness: usize) {
    for c in contours {
        let pts: Vec<(isize, isize)> = c.iter().map(|p| (p.0 as isize, p.1 as isize)).collect();
        draw_polyline(img, &pts, color, thickness);
    }
}

/// 绘制闭合折线 (坐标可越界, 越界部分被裁剪)。
pub fn draw_polyline(img: &mut Image, pts: &[(isize, isize)], color: [u8; 4], thickness: usize) {
    let r = thickness.saturating_sub(1) as isize / 2;
    let mut plot = |x: isize, y: isize| {
        for j in -r..=r {
            for i in -r..=r {
                let (px, py) = (x + i, y + j);
                if px >= 0 && py >= 0 && (px as usize) < img.width && (py as usize) < img.height {
                    img.set(px as usize, py as usize, color);
                }
            }
        }
    };
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        let (mut x0, mut y0, x1, y1) = (a.0, a.1, b.0, b.1);
        let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
        let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
        let mut err = dx + dy;
        loop {
            plot(x0, y0);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x0 += sx; }
            if e2 <= dx { err += dx; y0 += sy; }
        }
    }
}
/// 骨架化 (Zhang-Suen 细化), 输入二值图 (255 前景), 输出 1 像素宽骨架。
pub fn skeletonize(bin: &Gray) -> Gray {
    let (w, h) = (bin.width, bin.height);
    let mut a: Vec<u8> = bin.data.iter().map(|&v| (v > 127.0) as u8).collect();
    let at = |a: &[u8], x: isize, y: isize| -> u8 {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize { 0 } else { a[y as usize * w + x as usize] }
    };
    loop {
        let mut changed = false;
        for step in 0..2 {
            let mut del = Vec::new();
            for y in 0..h as isize {
                for x in 0..w as isize {
                    if a[y as usize * w + x as usize] == 0 {
                        continue;
                    }
                    // P2..P9 顺时针, 从正上方开始
                    let p = [at(&a, x, y - 1), at(&a, x + 1, y - 1), at(&a, x + 1, y), at(&a, x + 1, y + 1),
                             at(&a, x, y + 1), at(&a, x - 1, y + 1), at(&a, x - 1, y), at(&a, x - 1, y - 1)];
                    let b: u8 = p.iter().sum();
                    let t = (0..8).filter(|&i| p[i] == 0 && p[(i + 1) % 8] == 1).count();
                    let (c1, c2) = if step == 0 {
                        (p[0] * p[2] * p[4], p[2] * p[4] * p[6])
                    } else {
                        (p[0] * p[2] * p[6], p[0] * p[4] * p[6])
                    };
                    if (2..=6).contains(&b) && t == 1 && c1 == 0 && c2 == 0 {
                        del.push(y as usize * w + x as usize);
                    }
                }
            }
            if !del.is_empty() {
                changed = true;
                for i in del {
                    a[i] = 0;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Gray { width: w, height: h, data: a.into_iter().map(|v| v as f32 * 255.0).collect() }
}