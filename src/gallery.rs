//! 示例库: 对一张图片逐个运行全部算法, 产出 (分类, 名称, 说明, 代码, 结果图)。
//! `examples/gallery.rs` 用它生成 `docs/images/*.png` 与 `docs/GALLERY.md`; `tests/gallery.rs` 用它做回归测试。

use crate::advanced::*;
use crate::binary::*;
use crate::blend::*;
use crate::color::*;
use crate::distort::*;
use crate::edge::*;
use crate::filter::*;
use crate::space::*;
use crate::transform::*;
use crate::{Gray, Image, Interp};

pub struct Case {
    pub category: &'static str,
    pub name: &'static str,
    pub title: &'static str,
    /// 调用代码 (由 `stringify!` 生成)
    pub code: String,
    pub image: Image,
}

macro_rules! case {
    ($v:ident, $cat:expr, $name:expr, $title:expr, $e:expr) => {
        $v.push(Case { category: $cat, name: $name, title: $title, code: stringify!($e).to_string(), image: $e });
    };
}

fn gray_img(g: &Gray) -> Image {
    g.to_image()
}

/// 生成全部示例。`src` 为输入图 (建议 200~400 像素)。
pub fn cases(src: &Image) -> Vec<Case> {
    let mut v: Vec<Case> = Vec::new();
    let (w, h) = (src.width as f32, src.height as f32);
    let c = (w / 2.0, h / 2.0);
    let r = w.min(h) / 2.0;
    let b = Interp::Bilinear;
    let gray = src.to_gray();
    let otsu = otsu_threshold(&gray);
    let bin = binarize(&gray, otsu, false);
    let overlay = Image::from_fn(src.width, src.height, |x, y| {
        let t = x as f32 / w;
        [(255.0 * t) as u8, (120.0 * (1.0 - t)) as u8, (255.0 * (1.0 - t)) as u8, 255 - (y as f32 / h * 80.0) as u8]
    });
    let id = [(0.0, 0.0), (255.0, 255.0)];

    let cat = "调色";
    case!(v, cat, "color_grayscale", "灰度化 (Rec.709)", grayscale(src, GrayMode::Luma709));
    case!(v, cat, "color_brightness_contrast", "亮度/对比度", brightness_contrast(src, 20.0, 30.0));
    case!(v, cat, "color_invert", "反相", invert(src));
    case!(v, cat, "color_gamma", "Gamma", gamma(src, 1.8));
    case!(v, cat, "color_levels", "色阶", levels(src, 30.0, 220.0, 1.2, 0.0, 255.0));
    case!(v, cat, "color_curves", "曲线 (S 曲线)", curves(src, &[(0.0, 0.0), (64.0, 40.0), (192.0, 220.0), (255.0, 255.0)], &id, &id));
    case!(v, cat, "color_hue_saturation", "色相/饱和度", hue_saturation(src, 60.0, 30.0, 0.0));
    case!(v, cat, "color_vibrance", "自然饱和度", vibrance(src, 0.7));
    case!(v, cat, "color_channel_mixer", "通道混合 (R/B 互换)", channel_mixer(src, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [0.0; 3]));
    case!(v, cat, "color_sepia", "棕褐色", sepia(src));
    case!(v, cat, "color_posterize", "色调分离", posterize(src, 4));
    case!(v, cat, "color_threshold", "阈值", threshold(src, 128));
    case!(v, cat, "color_exposure", "曝光度", exposure(src, 0.8, 0.0, 1.0));
    case!(v, cat, "color_balance", "色彩平衡", color_balance(src, [0.0, 0.0, 30.0], [30.0, 0.0, -20.0], [40.0, 10.0, -30.0]));
    case!(v, cat, "color_photo_filter", "照片滤镜 (暖色)", photo_filter(src, [236, 138, 0], 0.5, true));
    case!(v, cat, "color_auto_levels", "自动色阶 (对低对比度图)", auto_levels(&brightness_contrast(src, 0.0, -60.0), 0.01));
    case!(v, cat, "color_equalize", "直方图均衡", equalize(src));
    case!(v, cat, "color_vignette", "暗角", vignette(src, 0.8, 0.4, 0.6));
    case!(v, cat, "color_gradient_map", "渐变映射", gradient_map(src, &[(0.0, [20, 0, 80]), (0.5, [220, 60, 90]), (1.0, [255, 240, 150])]));
    case!(v, cat, "space_lab_contrast", "Lab 明度对比度", lab_contrast(src, 1.5));
    case!(v, cat, "space_lab_saturation", "Lab 色彩增强", lab_saturation(src, 1.8));
    case!(v, cat, "space_channel_L", "Lab 的 L 通道", channel_view(src, "L"));
    case!(v, cat, "space_channel_S", "HSV 的 S 通道", channel_view(src, "S"));
    case!(v, cat, "space_channel_Cr", "YCbCr 的 Cr 通道", channel_view(src, "Cr"));

    let cat = "混合";
    for (name, title, m) in [
        ("blend_multiply", "混合: 正片叠底", BlendMode::Multiply),
        ("blend_screen", "混合: 滤色", BlendMode::Screen),
        ("blend_overlay", "混合: 叠加", BlendMode::Overlay),
        ("blend_soft_light", "混合: 柔光", BlendMode::SoftLight),
        ("blend_difference", "混合: 差值", BlendMode::Difference),
    ] {
        v.push(Case { category: cat, name, title, code: format!("blend(src, &overlay, BlendMode::{m:?}, 1.0, None)"), image: blend(src, &overlay, m, 1.0, None) });
    }
    let radial = Gray { width: src.width, height: src.height, data: (0..src.width * src.height).map(|i| { let (x, y) = ((i % src.width) as f32 - c.0, (i / src.width) as f32 - c.1); if (x * x + y * y).sqrt() < r * 0.6 { 255.0 } else { 0.0 } }).collect() };
    case!(v, cat, "blend_masked", "羽化蒙版合成", blend(src, &overlay, BlendMode::Normal, 1.0, Some(&feather_mask(&radial, 12.0))));

    let cat = "滤镜";
    case!(v, cat, "filter_gaussian", "高斯模糊 σ=3", gaussian_blur(src, 3.0));
    case!(v, cat, "filter_box", "方框模糊", box_blur(src, 3));
    case!(v, cat, "filter_motion", "动感模糊", motion_blur(src, 30.0, 15));
    case!(v, cat, "filter_radial_spin", "径向模糊(旋转)", radial_blur(src, c, 20.0, true, 16));
    case!(v, cat, "filter_radial_zoom", "径向模糊(缩放)", radial_blur(src, c, 30.0, false, 16));
    case!(v, cat, "filter_bilateral", "双边滤波(保边)", bilateral(src, 3.0, 40.0));
    case!(v, cat, "filter_median", "中值滤波", median(src, 2));
    case!(v, cat, "filter_sharpen", "锐化", sharpen(src, 0.8));
    case!(v, cat, "filter_unsharp", "USM 锐化", unsharp_mask(src, 1.5, 2.0, 0.0));
    case!(v, cat, "filter_high_pass", "高反差保留", high_pass(src, 3.0));
    case!(v, cat, "filter_emboss", "浮雕", emboss(src, 135.0, 1.0));
    case!(v, cat, "filter_noise", "添加噪点", add_noise(src, 25.0, true, false, 42));
    case!(v, cat, "filter_pixelate", "马赛克", pixelate(src, 10));
    case!(v, cat, "filter_oil", "油画", oil_paint(src, 3, 20));
    case!(v, cat, "filter_halftone", "半调网点", halftone(src, 8));
    case!(v, cat, "filter_sketch", "素描", sketch(src, 5.0, 1.0));
    case!(v, cat, "filter_despeckle", "去斑", despeckle(src));
    case!(v, cat, "adv_kuwahara", "Kuwahara 保边绘画", kuwahara(src, 4));
    case!(v, cat, "adv_nlm", "非局部均值降噪(对噪点图)", non_local_means(&add_noise(src, 20.0, true, false, 1), 1, 4, 20.0));
    case!(v, cat, "adv_lens_blur", "镜头模糊(散景)", lens_blur(src, 5, 1.0));
    case!(v, cat, "adv_bloom", "辉光", bloom(src, 180.0, 6.0, 1.0));
    case!(v, cat, "adv_dehaze", "暗通道去雾", dehaze(src, 3, 0.95, 0.2));
    case!(v, cat, "adv_cartoon", "卡通化", cartoon(src, 5, 40.0, 100.0));

    let cat = "边缘";
    case!(v, cat, "edge_roberts", "Roberts", gray_img(&gradient_edges(src, GradientOperator::Roberts, 1.0, 0.0)));
    case!(v, cat, "edge_prewitt", "Prewitt", gray_img(&gradient_edges(src, GradientOperator::Prewitt, 1.0, 0.0)));
    case!(v, cat, "edge_sobel", "Sobel", gray_img(&gradient_edges(src, GradientOperator::Sobel, 1.0, 0.0)));
    case!(v, cat, "edge_scharr", "Scharr", gray_img(&gradient_edges(src, GradientOperator::Scharr, 1.0, 0.0)));
    case!(v, cat, "edge_laplacian", "拉普拉斯", gray_img(&laplacian(src, true)));
    case!(v, cat, "edge_log", "LoG 零交叉", gray_img(&laplacian_of_gaussian(src, 2.0, 4.0)));
    case!(v, cat, "edge_dog", "高斯差分 DoG", gray_img(&difference_of_gaussians(src, 1.0, 2.0)));
    case!(v, cat, "edge_canny", "Canny", gray_img(&canny(src, CannyParams::default())));
    case!(v, cat, "edge_find_edges_color", "查找边缘(彩色)", find_edges_color(src));
    case!(v, cat, "edge_hough_lines", "霍夫直线", {
        let e = canny(src, CannyParams::default());
        let mut out = src.clone();
        for (rho, th, _) in hough_lines(&e, 180, (w * 0.3) as u32).into_iter().take(8) {
            let (s, co) = th.sin_cos();
            for t in -(w + h) as i32..(w + h) as i32 {
                let (x, y) = (rho * co - t as f32 * s, rho * s + t as f32 * co);
                if x >= 0.0 && y >= 0.0 && x < w && y < h {
                    out.set(x as usize, y as usize, [255, 0, 0, 255]);
                }
            }
        }
        out
    });
    case!(v, cat, "edge_hough_circles", "霍夫圆 (取最强 3 个)", {
        let e = canny(src, CannyParams::default());
        let mut out = src.clone();
        for (cx, cy, rr, _) in hough_circles(&e, (r * 0.2) as usize, (r * 0.5) as usize, 40).into_iter().take(3) {
            let pts: Vec<(usize, usize)> = (0..360).map(|a| { let a = (a as f32).to_radians(); (((cx as f32 + rr as f32 * a.cos()).max(0.0) as usize).min(src.width - 1), ((cy as f32 + rr as f32 * a.sin()).max(0.0) as usize).min(src.height - 1)) }).collect();
            draw_points(&mut out, &pts, [0, 255, 0, 255], 0);
        }
        out
    });
    case!(v, cat, "feature_harris", "Harris 角点 (前 60 个)", {
        let pts: Vec<(usize, usize)> = harris_corners(src, 2.0, 0.04, 0.01).into_iter().take(60).map(|p| (p.0, p.1)).collect();
        let mut out = src.clone();
        draw_points(&mut out, &pts, [255, 0, 0, 255], 3);
        out
    });

    let cat = "二值化/形态学/轮廓";
    case!(v, cat, "bin_otsu", "Otsu 阈值", gray_img(&bin));
    case!(v, cat, "bin_triangle", "三角法阈值", gray_img(&binarize(&gray, triangle_threshold(&gray), false)));
    case!(v, cat, "bin_adaptive", "自适应阈值", gray_img(&adaptive_threshold(&gray, 7, 5.0, true)));
    case!(v, cat, "morph_erode", "腐蚀", gray_img(&erode(&bin, Kernel::Ellipse(2))));
    case!(v, cat, "morph_dilate", "膨胀", gray_img(&dilate(&bin, Kernel::Ellipse(2))));
    case!(v, cat, "morph_open", "开运算", gray_img(&open(&bin, Kernel::Ellipse(2))));
    case!(v, cat, "morph_close", "闭运算", gray_img(&close(&bin, Kernel::Ellipse(2))));
    case!(v, cat, "morph_gradient", "形态学梯度", gray_img(&morph_gradient(&bin, Kernel::Rect(1))));
    case!(v, cat, "morph_top_hat", "顶帽(灰度)", gray_img(&top_hat(&gray, Kernel::Ellipse(5)).normalize()));
    case!(v, cat, "morph_skeleton", "骨架化", gray_img(&skeletonize(&bin)));
    case!(v, cat, "bin_remove_small", "去除小连通域 (<30)", gray_img(&remove_small_components(&bin, 30)));
    case!(v, cat, "bin_fill_holes", "填充孔洞", gray_img(&fill_holes(&bin)));
    case!(v, cat, "bin_distance", "距离变换", gray_img(&distance_transform(&bin).normalize()));
    case!(v, cat, "contour_all", "轮廓提取 (外轮廓, 洋红)", {
        let mut out = src.clone();
        draw_contours(&mut out, &find_contours(&bin, 20), [255, 0, 255, 255], 1);
        out
    });
    case!(v, cat, "contour_simplified", "轮廓多边形简化 ε=3 (绿)", {
        let mut out = src.clone();
        let cs: Vec<_> = find_contours(&bin, 20).iter().map(|c| simplify_contour(c, 3.0, true)).collect();
        draw_contours(&mut out, &cs, [0, 255, 0, 255], 1);
        out
    });
    case!(v, cat, "contour_canny_on_edges", "对 Canny 边缘做轮廓", {
        let e = dilate(&canny(src, CannyParams::default()), Kernel::Rect(1));
        let mut out = Image::new(src.width, src.height, [0, 0, 0, 255]);
        draw_contours(&mut out, &find_contours(&e, 30), [255, 255, 0, 255], 1);
        out
    });

    let cat = "几何变换";
    case!(v, cat, "tf_resize_nearest", "缩放: 最近邻 (缩小到 1/4 再放大)", resize(&resize(src, src.width / 4, src.height / 4, Interp::Nearest), src.width, src.height, Interp::Nearest));
    case!(v, cat, "tf_resize_bilinear", "缩放: 双线性 (缩小到 1/4 再放大)", resize(&resize(src, src.width / 4, src.height / 4, Interp::Bilinear), src.width, src.height, Interp::Bilinear));
    case!(v, cat, "tf_resize_bicubic", "缩放: 双三次 (缩小到 1/4 再放大)", resize(&resize(src, src.width / 4, src.height / 4, Interp::Bicubic), src.width, src.height, Interp::Bicubic));
    case!(v, cat, "tf_crop", "裁剪", crop(src, src.width / 4, src.height / 4, src.width / 2, src.height / 2));
    case!(v, cat, "tf_flip_h", "水平翻转", flip_horizontal(src));
    case!(v, cat, "tf_flip_v", "垂直翻转", flip_vertical(src));
    case!(v, cat, "tf_rotate90", "旋转 90°", rotate90(src, 1));
    case!(v, cat, "tf_rotate30", "任意角度旋转 30° (扩展画布)", rotate(src, 30.0, true, Interp::Bicubic, [30, 30, 30, 255]));
    case!(v, cat, "tf_affine", "仿射 (缩放+旋转+平移)", {
        let (s, co) = (0.35f32.sin() * 0.8, 0.35f32.cos() * 0.8);
        warp_affine(src, &[[co, -s, 30.0], [s, co, 10.0]], src.width, src.height, b, [30, 30, 30, 255]).unwrap()
    });
    case!(v, cat, "tf_skew", "斜切", skew(src, 20.0, 0.0, b, [30, 30, 30, 255]));
    case!(v, cat, "tf_perspective_distort", "透视: 映射到梯形", perspective_distort(src, &[(w * 0.2, h * 0.1), (w * 0.8, 0.0), (w - 1.0, h - 1.0), (0.0, h * 0.9)], src.width, src.height, b, [30, 30, 30, 255]).unwrap());
    case!(v, cat, "tf_perspective_rectify", "透视矫正: 把中间四边形拉正", perspective_rectify(src, &[(w * 0.2, h * 0.15), (w * 0.85, h * 0.1), (w * 0.9, h * 0.9), (w * 0.1, h * 0.8)], src.width, src.height, b).unwrap());

    let cat = "扭曲变形";
    case!(v, cat, "dist_swirl", "旋转扭曲", swirl(src, c, r, 120.0, b));
    case!(v, cat, "dist_pinch_bulge", "膨胀", pinch(src, c, r, 0.6, b));
    case!(v, cat, "dist_pinch_squeeze", "挤压", pinch(src, c, r, -0.6, b));
    case!(v, cat, "dist_spherize", "球面化", spherize(src, c, r, b));
    case!(v, cat, "dist_ripple", "波纹", ripple(src, 4.0, 40.0, b));
    case!(v, cat, "dist_wave", "波浪", wave(src, 8.0, 60.0, 0.0, false, b));
    case!(v, cat, "dist_barrel", "桶形畸变", lens_distortion(src, 0.3, 0.0, b));
    case!(v, cat, "dist_pincushion", "枕形畸变", lens_distortion(src, -0.3, 0.0, b));
    case!(v, cat, "dist_fisheye", "鱼眼", fisheye(src, 0.8, b));
    case!(v, cat, "dist_polar", "极坐标 (矩形→极坐标)", polar(src, true, b));
    case!(v, cat, "dist_polar_inverse", "极坐标 (极坐标→矩形)", polar(src, false, b));
    case!(v, cat, "dist_cylinder", "圆柱面", cylinder(src, 0.7, b));
    case!(v, cat, "dist_shear_curve", "切变曲线", shear_curve(src, &[0.0, 20.0, -20.0, 0.0], b));
    case!(v, cat, "dist_mesh_warp", "网格变形 (中心控制点上移)", {
        let (cols, rows) = (4usize, 4usize);
        let pts: Vec<(f32, f32)> = (0..=rows).flat_map(|j| (0..=cols).map(move |i| (i, j))).map(|(i, j)| {
            let (x, y) = (i as f32 * w / cols as f32, j as f32 * h / rows as f32);
            let (x, y) = (x.min(w - 1.0), y.min(h - 1.0));
            if i == 2 && j == 2 { (x + 20.0, y - 30.0) } else { (x, y) }
        }).collect();
        mesh_warp(src, cols, rows, &pts, b)
    });
    case!(v, cat, "dist_liquify", "液化推动", liquify_push(src, c, r * 0.6, 30.0, -10.0, b));
    case!(v, cat, "dist_displace", "置换图 (用自身亮度)", displace(src, &gaussian_blur(src, 2.0), 10.0, 10.0, b));
    v
}
