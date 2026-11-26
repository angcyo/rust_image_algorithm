//! 高精度轮廓提取 (类似 Photoshop "魔棒 → 建立工作路径" / Illustrator "图像描摹")。
//!
//! # 原理
//! 对灰度场 `f(x,y)` 用 **Marching Squares (移动方格)** 提取阈值等值线:
//! 每个 2x2 像素格按四角是否 `>= threshold` 查表得到 0/1/2 条线段, 线段端点在格边上按
//! **线性插值** 求得 (亚像素精度), 再把共享端点的线段首尾相连成闭合环。
//! 所有环方向统一为 "前景在行进方向左侧" (图像坐标 y 向下), 因此:
//! - 外轮廓与孔洞天然由环的有向面积符号区分, 无需额外的填充/洪水算法;
//! - 图像边缘视为背景, 贴边的前景也能得到闭合轮廓;
//! - 鞍点 (对角两角同号) 用格中心均值消歧, 避免拓扑错误。
//!
//! 坐标约定: 像素 `(x, y)` 的**中心**位于浮点坐标 `(x, y)`。
//!
//! # 处理流程 (`extract_contours`)
//! 预模糊 → 等值线提取 → 面积/长度过滤 → 孔洞过滤 → 层级关系 → Douglas–Peucker 简化 →
//! Chaikin 平滑 → 偏移 (拐角类型) → 平移。

use crate::filter::gaussian_blur_gray;
use crate::{Gray, Image};
use std::collections::HashMap;
use std::fmt::Write;

/// 偏移时拐角 (连接点) 的处理方式, 含义同 Photoshop/Illustrator 描边的 "连接"。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JoinType {
    /// 尖角: 延长两条偏移边直到相交; 超过 `miter_limit` 时退化为斜角。
    Miter,
    /// 圆角: 以原顶点为圆心、偏移距离为半径的圆弧。
    Round,
    /// 斜角: 直接连接两条偏移边的端点, 切掉尖角。
    Bevel,
}

/// 轮廓提取参数。`Default` 为: 阈值 128、保留孔洞、不偏移、亚像素、不简化。
#[derive(Clone, Debug)]
pub struct ContourParams {
    /// 二值化阈值 (灰度 0~255), `value >= threshold` 视为前景。
    pub threshold: f32,
    /// 反相: 取暗部为前景 (如白底黑字)。
    pub invert: bool,
    /// 预处理高斯模糊 σ (像素), 0 不模糊。抑制噪点与锯齿, 使等值线更平滑。
    pub smooth_sigma: f32,
    /// 是否用线性插值求亚像素位置; `false` 时端点固定在格边中点 (速度略快, 精度低)。
    pub subpixel: bool,
    /// 是否保留孔洞轮廓 (内轮廓)。`false` 只输出外轮廓。
    pub keep_holes: bool,
    /// 最小面积 (像素²), 小于则丢弃 (去噪点)。在偏移前按原始轮廓判断。
    pub min_area: f32,
    /// 最大面积 (像素²), `<= 0` 表示不限制 (可用于去掉整幅背景框)。
    pub max_area: f32,
    /// 最小周长 (像素), 小于则丢弃。
    pub min_length: f32,
    /// Douglas–Peucker 简化容差 (像素), 0 不简化。
    pub simplify_epsilon: f32,
    /// Chaikin 切角平滑迭代次数 (0 不平滑)。每次迭代点数翻倍, 轮廓更圆滑 (会略微内缩)。
    pub smooth_iterations: u32,
    /// 沿法线方向的偏移距离 (像素): `>0` 外扩 (前景变大, 孔洞变小), `<0` 内缩。
    pub offset: f32,
    /// 偏移时的拐角类型。
    pub join: JoinType,
    /// 尖角限制 (斜接长度 / 偏移距离), 仅 `Miter` 使用, 典型值 2~10。
    pub miter_limit: f32,
    /// 圆角每段弧的最大角度 (度), 越小越圆滑, 仅 `Round` 使用。
    pub arc_step_deg: f32,
    /// 整体平移 `(dx, dy)` (像素), 在偏移之后应用。
    pub translate: (f32, f32),
}

impl Default for ContourParams {
    fn default() -> Self {
        ContourParams {
            threshold: 128.0,
            invert: false,
            smooth_sigma: 0.0,
            subpixel: true,
            keep_holes: true,
            min_area: 0.0,
            max_area: 0.0,
            min_length: 0.0,
            simplify_epsilon: 0.0,
            smooth_iterations: 0,
            offset: 0.0,
            join: JoinType::Miter,
            miter_limit: 4.0,
            arc_step_deg: 15.0,
            translate: (0.0, 0.0),
        }
    }
}

/// 一条闭合的浮点轮廓路径。
#[derive(Clone, Debug)]
pub struct ContourPath {
    /// 顶点序列 (首尾隐式相连)。
    pub points: Vec<(f32, f32)>,
    /// 是否为孔洞 (内轮廓)。
    pub is_hole: bool,
    /// 面积 (像素², 取绝对值, 为偏移/平移之前的原始轮廓面积)。
    pub area: f32,
    /// 直接包含它的轮廓在结果数组中的下标 (顶层为 `None`)。
    pub parent: Option<usize>,
}

/// 有向面积 (鞋带公式, 图像坐标系)。本模块中外轮廓为正, 孔洞为负。
pub fn signed_area(p: &[(f32, f32)]) -> f32 {
    let mut s = 0.0f64;
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        s += a.0 as f64 * b.1 as f64 - b.0 as f64 * a.1 as f64;
    }
    (s / 2.0) as f32
}

/// 闭合折线周长。
pub fn path_length(p: &[(f32, f32)]) -> f32 {
    (0..p.len()).map(|i| {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    }).sum()
}

/// 点是否在多边形内 (射线法, 偶奇规则)。
pub fn point_in_polygon(pt: (f32, f32), poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.1 > pt.1) != (b.1 > pt.1) && pt.0 < (b.0 - a.0) * (pt.1 - a.1) / (b.1 - a.1) + a.0 {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// 从灰度图提取轮廓, 参数见 [`ContourParams`]。
///
/// 返回的路径按提取顺序排列, `parent` 下标指向同一数组。
pub fn extract_contours(gray: &Gray, p: &ContourParams) -> Vec<ContourPath> {
    let src = if p.smooth_sigma > 0.0 { gaussian_blur_gray(gray, p.smooth_sigma) } else { gray.clone() };
    let mut paths: Vec<ContourPath> = marching_squares(&src, p.threshold, p.invert, p.subpixel)
        .into_iter()
        .filter_map(|pts| {
            let sa = signed_area(&pts);
            let area = sa.abs();
            let ok = area >= p.min_area
                && (p.max_area <= 0.0 || area <= p.max_area)
                && path_length(&pts) >= p.min_length
                && (p.keep_holes || sa > 0.0);
            ok.then(|| ContourPath { points: pts, is_hole: sa < 0.0, area, parent: None })
        })
        .collect();

    // 层级: 父轮廓 = 包含其首点且面积更大的最小轮廓
    for i in 0..paths.len() {
        let q = paths[i].points[0];
        let mut best: Option<usize> = None;
        for j in 0..paths.len() {
            if j != i && paths[j].area > paths[i].area && point_in_polygon(q, &paths[j].points)
                && best.is_none_or(|b| paths[j].area < paths[b].area)
            {
                best = Some(j);
            }
        }
        paths[i].parent = best;
    }

    for c in paths.iter_mut() {
        if p.simplify_epsilon > 0.0 {
            c.points = simplify_path(&c.points, p.simplify_epsilon);
        }
        for _ in 0..p.smooth_iterations {
            c.points = chaikin(&c.points);
        }
        if p.offset != 0.0 {
            c.points = offset_path(&c.points, p.offset, p.join, p.miter_limit, p.arc_step_deg);
        }
        if p.translate != (0.0, 0.0) {
            for q in c.points.iter_mut() {
                q.0 += p.translate.0;
                q.1 += p.translate.1;
            }
        }
    }
    // 内缩到消失 (面积变号/过小) 的轮廓丢弃, 同时修正 parent 下标
    let keep: Vec<bool> = paths.iter().map(|c| {
        c.points.len() >= 3 && (signed_area(&c.points) > 0.0) == !c.is_hole && signed_area(&c.points).abs() > 1e-3
    }).collect();
    if keep.iter().all(|&k| k) {
        return paths;
    }
    let mut remap = vec![None; paths.len()];
    let mut n = 0;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            remap[i] = Some(n);
            n += 1;
        }
    }
    paths.into_iter().enumerate().filter(|(i, _)| keep[*i]).map(|(_, mut c)| {
        c.parent = c.parent.and_then(|q| remap[q]);
        c
    }).collect()
}

/// 对彩色/灰度图的便捷封装: 先转灰度再 [`extract_contours`]。
pub fn extract_contours_image(img: &Image, p: &ContourParams) -> Vec<ContourPath> {
    extract_contours(&img.to_gray(), p)
}

/// Marching Squares。输出的环: 前景在行进方向左侧, 外轮廓有向面积 (本模块定义) 为正。
fn marching_squares(g: &Gray, threshold: f32, invert: bool, subpixel: bool) -> Vec<Vec<(f32, f32)>> {
    // 四周补一圈背景, 保证所有轮廓闭合
    let (gw, gh) = (g.width + 2, g.height + 2);
    let bg = -1.0f32;
    let mut f = vec![bg; gw * gh];
    for y in 0..g.height {
        for x in 0..g.width {
            let v = g.data[y * g.width + x];
            f[(y + 1) * gw + x + 1] = if invert { threshold - v } else { v - threshold };
            // 恰好等于阈值时 invert 会得到 0 (算前景), 与非反相的 ">=" 保持对称
        }
    }
    let inside = |v: f32| v >= 0.0;
    // 边的唯一标识: 水平边 2*(y*gw+x), 竖直边 2*(y*gw+x)+1 (均以左/上端点为基准)
    struct Seg {
        a: u64,
        b: u64,
        pa: (f32, f32),
    }
    let mut segs: Vec<Seg> = Vec::new();
    let off = 1.0f32; // 补边后的坐标偏移
    for y in 0..gh - 1 {
        for x in 0..gw - 1 {
            let c = [f[y * gw + x], f[y * gw + x + 1], f[(y + 1) * gw + x + 1], f[(y + 1) * gw + x]]; // TL TR BR BL
            let corner = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)];
            let flags = [inside(c[0]), inside(c[1]), inside(c[2]), inside(c[3])];
            let n_in = flags.iter().filter(|&&b| b).count();
            if n_in == 0 || n_in == 4 {
                continue;
            }
            // 边 i 连接角 i 与角 (i+1)%4: 0 上, 1 右, 2 下, 3 左
            let edge_id = |i: usize| -> u64 {
                let (cx, cy) = match i {
                    0 => (x, y),
                    1 => (x + 1, y),
                    2 => (x, y + 1),
                    _ => (x, y),
                };
                let base = (cy * gw + cx) as u64 * 2;
                if i == 0 || i == 2 { base } else { base + 1 }
            };
            let crossing = |i: usize| -> (f32, f32) {
                let (j, k) = (i, (i + 1) % 4);
                let t = if subpixel { c[j] / (c[j] - c[k]) } else { 0.5 };
                let (pj, pk) = (corner[j], corner[k]);
                (
                    pj.0 as f32 + (pk.0 as f32 - pj.0 as f32) * t - off,
                    pj.1 as f32 + (pk.1 as f32 - pj.1 as f32) * t - off,
                )
            };
            let cross: Vec<usize> = (0..4).filter(|&i| flags[i] != flags[(i + 1) % 4]).collect();
            // (边a, 边b, 已知位于"前景侧"的角, 已知位于"背景侧"的角)
            let mut pairs: Vec<(usize, usize, Vec<usize>, Vec<usize>)> = Vec::new();
            if cross.len() == 2 {
                let ins: Vec<usize> = (0..4).filter(|&i| flags[i]).collect();
                let outs: Vec<usize> = (0..4).filter(|&i| !flags[i]).collect();
                pairs.push((cross[0], cross[1], ins, outs));
            } else {
                let center_in = inside((c[0] + c[1] + c[2] + c[3]) / 4.0);
                // 每个线段切下一个角: 角 k 由其两侧的边 (k+3)%4 与 k 夹成
                let cut: Vec<usize> = if center_in {
                    (0..4).filter(|&k| !flags[k]).collect()
                } else {
                    (0..4).filter(|&k| flags[k]).collect()
                };
                for k in cut {
                    let (e1, e2) = ((k + 3) % 4, k);
                    if flags[k] {
                        pairs.push((e1, e2, vec![k], vec![]));
                    } else {
                        pairs.push((e1, e2, vec![], vec![k]));
                    }
                }
            }
            for (ea, eb, ins, outs) in pairs {
                let (pa, pb) = (crossing(ea), crossing(eb));
                let m = ((pa.0 + pb.0) / 2.0, (pa.1 + pb.1) / 2.0);
                let d = (pb.0 - pa.0, pb.1 - pa.1);
                let n = (-d.1, d.0); // 行进方向的左法线 (坐标系 y 向下)
                let side = |k: usize| {
                    let q = (corner[k].0 as f32 - off - m.0, corner[k].1 as f32 - off - m.1);
                    q.0 * n.0 + q.1 * n.1
                };
                let score: f32 = ins.iter().map(|&k| side(k)).sum::<f32>() - outs.iter().map(|&k| side(k)).sum::<f32>();
                // 约定: 前景在行进方向的左侧; 屏幕坐标 (y 向下) 中 "左" 取 (-dy, dx) 的反向
                if score > 0.0 {
                    segs.push(Seg { a: edge_id(ea), b: edge_id(eb), pa });
                } else {
                    segs.push(Seg { a: edge_id(eb), b: edge_id(ea), pa: pb });
                }
            }
        }
    }

    let mut start: HashMap<u64, usize> = HashMap::with_capacity(segs.len());
    for (i, s) in segs.iter().enumerate() {
        start.insert(s.a, i);
    }
    let mut used = vec![false; segs.len()];
    let mut loops = Vec::new();
    for i in 0..segs.len() {
        if used[i] {
            continue;
        }
        let mut pts = Vec::new();
        let mut cur = i;
        loop {
            used[cur] = true;
            pts.push(segs[cur].pa);
            match start.get(&segs[cur].b) {
                Some(&nx) if !used[nx] => cur = nx,
                _ => break,
            }
        }
        if pts.len() >= 3 {
            loops.push(pts);
        }
    }
    loops
}

/// Douglas–Peucker 简化闭合浮点折线。
pub fn simplify_path(pts: &[(f32, f32)], epsilon: f32) -> Vec<(f32, f32)> {
    if pts.len() < 4 {
        return pts.to_vec();
    }
    // 以相距最远的两点为锚点把环拆成两段分别简化, 避免起点偏置
    let a = 0;
    let b = (0..pts.len()).max_by(|&i, &j| {
        let di = (pts[i].0 - pts[a].0).powi(2) + (pts[i].1 - pts[a].1).powi(2);
        let dj = (pts[j].0 - pts[a].0).powi(2) + (pts[j].1 - pts[a].1).powi(2);
        di.partial_cmp(&dj).unwrap()
    }).unwrap();
    fn dp(p: &[(f32, f32)], eps: f32, out: &mut Vec<(f32, f32)>) {
        let (s, e) = (p[0], p[p.len() - 1]);
        let (dx, dy) = (e.0 - s.0, e.1 - s.1);
        let len = (dx * dx + dy * dy).sqrt();
        let (mut md, mut mi) = (0.0, 0);
        for (i, q) in p.iter().enumerate().take(p.len() - 1).skip(1) {
            let (qx, qy) = (q.0 - s.0, q.1 - s.1);
            let d = if len < 1e-9 { (qx * qx + qy * qy).sqrt() } else { (qx * dy - qy * dx).abs() / len };
            if d > md {
                md = d;
                mi = i;
            }
        }
        if md > eps {
            dp(&p[..=mi], eps, out);
            dp(&p[mi..], eps, out);
        } else {
            out.push(s);
        }
    }
    let mut seg1: Vec<_> = pts[a..=b].to_vec();
    let mut seg2: Vec<_> = pts[b..].to_vec();
    seg2.push(pts[a]);
    let mut out = Vec::new();
    dp(&seg1, epsilon, &mut out);
    dp(&seg2, epsilon, &mut out);
    seg1.clear();
    out
}

/// Chaikin 切角平滑 (一次迭代: 每条边取 1/4、3/4 处两点)。
pub fn chaikin(pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = pts.len();
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        out.push((0.75 * a.0 + 0.25 * b.0, 0.75 * a.1 + 0.25 * b.1));
        out.push((0.25 * a.0 + 0.75 * b.0, 0.25 * a.1 + 0.75 * b.1));
    }
    out
}

/// 闭合折线沿法线偏移 `d` 像素 (`d>0` 向前景外侧)。
///
/// 每个顶点根据相邻两条边的转向判定:
/// - **内侧** (偏移边相交): 取交点, 即 `p + (n1+n2)/(1+n1·n2)·d`;
/// - **外侧** (偏移边分离): 按 `join` 补连接 (尖角 / 圆角 / 斜角)。
///
/// 不做自相交清理: 偏移量远大于局部曲率半径时内侧可能出现小回环。
pub fn offset_path(pts: &[(f32, f32)], d: f32, join: JoinType, miter_limit: f32, arc_step_deg: f32) -> Vec<(f32, f32)> {
    let n = pts.len();
    if n < 3 || d == 0.0 {
        return pts.to_vec();
    }
    // 去掉重复点
    let mut p: Vec<(f32, f32)> = Vec::with_capacity(n);
    for &q in pts {
        if p.last().is_none_or(|l| (l.0 - q.0).abs() > 1e-6 || (l.1 - q.1).abs() > 1e-6) {
            p.push(q);
        }
    }
    if p.len() > 1 && p[0] == p[p.len() - 1] {
        p.pop();
    }
    let n = p.len();
    if n < 3 {
        return p;
    }
    // 单位方向; 环方向为 y 向下坐标系中的顺时针 (前景在行进方向右侧), 外侧法线 = (dy, -dx)
    let dir: Vec<(f32, f32)> = (0..n).map(|i| {
        let (a, b) = (p[i], p[(i + 1) % n]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l = (dx * dx + dy * dy).sqrt().max(1e-12);
        (dx / l, dy / l)
    }).collect();
    let nor = |v: (f32, f32)| (v.1, -v.0);
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let (d1, d2) = (dir[(i + n - 1) % n], dir[i]);
        let (n1, n2) = (nor(d1), nor(d2));
        let v = p[i];
        let cross = d1.0 * d2.1 - d1.1 * d2.0;
        let dot = n1.0 * n2.0 + n1.1 * n2.1;
                let sgn = 1.0f32;
        let (o1, o2) = ((v.0 + sgn * n1.0 * d, v.1 + sgn * n1.1 * d), (v.0 + sgn * n2.0 * d, v.1 + sgn * n2.1 * d));
        if dot > 0.9999 {
            out.push(o1);
            continue;
        }
        // 此环方向下 cross > 0 为前景的凸顶点
        let gap = cross * d > 0.0 && cross.abs() > 1e-9;
        let miter_pt = || {
            let k = 1.0 + dot;
            if k < 1e-6 {
                None
            } else {
                Some((v.0 + sgn * (n1.0 + n2.0) / k * d, v.1 + sgn * (n1.1 + n2.1) / k * d))
            }
        };
        if !gap {
            match miter_pt() {
                Some(m) => out.push(m),
                None => {
                    out.push(o1);
                    out.push(o2);
                }
            }
            continue;
        }
        match join {
            JoinType::Bevel => {
                out.push(o1);
                out.push(o2);
            }
            JoinType::Miter => {
                let ratio = (2.0 / (1.0 + dot)).sqrt();
                match miter_pt() {
                    Some(m) if ratio <= miter_limit => out.push(m),
                    _ => {
                        out.push(o1);
                        out.push(o2);
                    }
                }
            }
            JoinType::Round => {
                let a1 = (sgn * n1.1).atan2(sgn * n1.0);
                let mut da = (sgn * n2.1).atan2(sgn * n2.0) - a1;
                // 沿较短弧旋转
                while da > std::f32::consts::PI {
                    da -= 2.0 * std::f32::consts::PI;
                }
                while da < -std::f32::consts::PI {
                    da += 2.0 * std::f32::consts::PI;
                }
                let steps = (da.abs() / arc_step_deg.max(1.0).to_radians()).ceil().max(1.0) as usize;
                for s in 0..=steps {
                    let a = a1 + da * s as f32 / steps as f32;
                    out.push((v.0 + a.cos() * d.abs() * d.signum(), v.1 + a.sin() * d.abs() * d.signum()));
                }
            }
        }
    }
    out
}

/// 在图像上绘制浮点轮廓 (Bresenham 折线, 坐标四舍五入), 可超出图像范围。
pub fn draw_paths(img: &mut Image, paths: &[ContourPath], color: [u8; 4], thickness: usize) {
    for c in paths {
        let pts: Vec<(isize, isize)> = c.points.iter().map(|p| (p.0.round() as isize, p.1.round() as isize)).collect();
        crate::binary::draw_polyline(img, &pts, color, thickness);
    }
}

/// 导出为 SVG (每条轮廓一个 `<path>`, 孔洞通过 `fill-rule="evenodd"` 合并成一个路径)。
pub fn paths_to_svg(paths: &[ContourPath], width: usize, height: usize) -> String {
    let mut d = String::new();
    for c in paths {
        for (i, p) in c.points.iter().enumerate() {
            let _ = write!(d, "{}{:.2},{:.2} ", if i == 0 { "M" } else { "L" }, p.0, p.1);
        }
        d.push_str("Z ");
    }
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n<path fill-rule=\"evenodd\" d=\"{}\"/>\n</svg>\n",
        d.trim_end()
    )
}
