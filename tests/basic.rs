use rust_image_algorithm::binary::*;
use rust_image_algorithm::color::*;
use rust_image_algorithm::distort::*;
use rust_image_algorithm::edge::*;
use rust_image_algorithm::filter::*;
use rust_image_algorithm::transform::*;
use rust_image_algorithm::{Gray, Image, Interp};

fn square_img() -> Image {
    Image::from_fn(40, 30, |x, y| if (10..25).contains(&x) && (8..22).contains(&y) { [255, 255, 255, 255] } else { [0, 0, 0, 255] })
}

#[test]
fn contour_of_rectangle() {
    let bin = binarize(&square_img().to_gray(), 128.0, false);
    let cs = find_contours(&bin, 1);
    assert_eq!(cs.len(), 1);
    assert_eq!(bounding_rect(&cs[0]), (10, 8, 15, 14));
    assert_eq!(cs[0].len(), 2 * (15 + 14) - 4);
    assert!((contour_area(&cs[0]) - 14.0 * 13.0).abs() < 1e-3);
}

#[test]
fn canny_finds_square_edges() {
    let e = canny(&square_img(), CannyParams { sigma: 1.0, low: 30.0, high: 80.0, ..Default::default() });
    assert!(e.get(10, 15) > 0.0 || e.get(9, 15) > 0.0 || e.get(11, 15) > 0.0);
    assert_eq!(e.get(2, 2), 0.0);
    assert_eq!(e.get(17, 15), 0.0);
}

#[test]
fn otsu_separates_two_levels() {
    let g = Gray { width: 4, height: 1, data: vec![10.0, 12.0, 200.0, 210.0] };
    let t = otsu_threshold(&g);
    assert!(t > 12.0 && t < 200.0);
}

#[test]
fn morphology_and_components() {
    let bin = binarize(&square_img().to_gray(), 128.0, false);
    let er = erode(&bin, Kernel::Rect(1));
    assert_eq!(er.get(10, 8), 0.0);
    assert_eq!(er.get(11, 9), 255.0);
    let (_, n) = connected_components(&bin);
    assert_eq!(n, 1);
    assert!(distance_transform(&bin).get(17, 15) > 5.0);
}

#[test]
fn invertible_ops() {
    let img = square_img();
    assert_eq!(invert(&invert(&img)), img);
    assert_eq!(flip_horizontal(&flip_horizontal(&img)), img);
    assert_eq!(rotate90(&rotate90(&img, 1), 3), img);
    assert_eq!(rotate(&img, 0.0, false, Interp::Nearest, [0; 4]), img);
}

#[test]
fn perspective_identity_and_rectify() {
    let img = square_img();
    let q = [(0.0, 0.0), (39.0, 0.0), (39.0, 29.0), (0.0, 29.0)];
    assert_eq!(perspective_distort(&img, &q, 40, 30, Interp::Nearest, [0; 4]).unwrap(), img);
    assert_eq!(perspective_rectify(&img, &q, 40, 30, Interp::Nearest).unwrap(), img);
}

#[test]
fn distortions_zero_is_identity() {
    let img = square_img();
    let c = (20.0, 15.0);
    assert_eq!(swirl(&img, c, 10.0, 0.0, Interp::Nearest), img);
    assert_eq!(pinch(&img, c, 10.0, 0.0, Interp::Nearest), img);
    assert_eq!(wave(&img, 0.0, 10.0, 0.0, false, Interp::Nearest), img);
    assert_eq!(lens_distortion(&img, 0.0, 0.0, Interp::Nearest), img);
    assert_eq!(polar(&img, true, Interp::Bilinear).width, 40);
    let pts: Vec<(f32, f32)> = (0..=3).flat_map(|r| (0..=4).map(move |c| (c as f32 * 10.0, r as f32 * 10.0))).collect();
    assert_eq!(mesh_warp(&img, 4, 3, &pts, Interp::Nearest).width, 40);
}

#[test]
fn color_ops() {
    let img = square_img();
    assert_eq!(levels(&img, 0.0, 255.0, 1.0, 0.0, 255.0), img);
    let id = [(0.0, 0.0), (255.0, 255.0)];
    assert_eq!(curves(&img, &id, &id, &id), img);
    let (h, s, l) = rgb_to_hsl(200, 100, 50);
    assert_eq!(hsl_to_rgb(h, s, l), (200, 100, 50));
    let r = resize(&img, 20, 15, Interp::Bicubic);
    assert_eq!((r.width, r.height), (20, 15));
    assert_eq!(gaussian_blur(&img, 0.0), img);
}

fn ring_img() -> Gray {
    // 外框 x 10..=39, y 8..=31, 孔洞 x 20..=29, y 14..=25 (背景)
    gfn(60, 40, |x, y| {
        let outer = (10..40).contains(&x) && (8..32).contains(&y);
        let hole = (20..30).contains(&x) && (14..26).contains(&y);
        if outer && !hole { 255.0 } else { 0.0 }
    })
}

#[test]
fn contour_params_holes_and_hierarchy() {
    use rust_image_algorithm::contour::*;
    let g = ring_img();
    let all = extract_contours(&g, &ContourParams::default());
    assert_eq!(all.len(), 2);
    let outer = all.iter().position(|c| !c.is_hole).unwrap();
    let hole = all.iter().position(|c| c.is_hole).unwrap();
    assert!((all[outer].area - 30.0 * 24.0).abs() < 1.0, "{}", all[outer].area);
    assert!((all[hole].area - 10.0 * 12.0).abs() < 1.0, "{}", all[hole].area);
    assert_eq!(all[hole].parent, Some(outer));
    let only = extract_contours(&g, &ContourParams { keep_holes: false, ..Default::default() });
    assert_eq!(only.len(), 1);
    // 面积过滤去掉孔洞; 阈值高于前景时无轮廓; 反相后前景变为背景(外框+孔洞整体外轮廓为画面边界)
    assert_eq!(extract_contours(&g, &ContourParams { min_area: 200.0, ..Default::default() }).len(), 1);
    assert!(extract_contours(&g, &ContourParams { threshold: 300.0, ..Default::default() }).is_empty());
}

#[test]
fn contour_offset_and_join() {
    use rust_image_algorithm::contour::*;
    let g = ring_img();
    let area = |p: &ContourParams| -> (f32, f32) {
        let c = extract_contours(&g, p);
        let o = c.iter().find(|c| !c.is_hole).unwrap();
        let h = c.iter().find(|c| c.is_hole).unwrap();
        (signed_area(&o.points).abs(), signed_area(&h.points).abs())
    };
    let base = area(&ContourParams::default());
    // 尖角外扩 2px: 外框 (30+4)*(24+4), 孔洞缩小为 (10-4)*(12-4)
    let m = area(&ContourParams { offset: 2.0, join: JoinType::Miter, ..Default::default() });
    assert!((m.0 - 34.0 * 28.0).abs() < 8.0, "{m:?}");
    assert!((m.1 - 6.0 * 8.0).abs() < 3.0, "{m:?}");
    // 圆角外扩: A + P*d + πd² (多边形近似略小), 小于尖角
    let r = area(&ContourParams { offset: 2.0, join: JoinType::Round, ..Default::default() });
    let want = base.0 + 108.0 * 2.0 + std::f32::consts::PI * 4.0;
    assert!((r.0 - want).abs() < 6.0, "{r:?} {want}");
    let b = area(&ContourParams { offset: 2.0, join: JoinType::Bevel, ..Default::default() });
    assert!(b.0 < r.0 && r.0 < m.0, "{b:?} {r:?} {m:?}");
    // 内缩
    let s = area(&ContourParams { offset: -2.0, ..Default::default() });
    assert!((s.0 - 26.0 * 20.0).abs() < 8.0, "{s:?}");
    // 内缩超过半宽: 孔洞外扩后吞并, 轮廓被丢弃或保持有效
    for c in extract_contours(&g, &ContourParams { offset: -20.0, ..Default::default() }) {
        assert!(signed_area(&c.points).abs() > 0.0);
    }
}

#[test]
fn contour_translate_and_subpixel() {
    use rust_image_algorithm::contour::*;
    let g = ring_img();
    let a = extract_contours(&g, &ContourParams::default());
    let b = extract_contours(&g, &ContourParams { translate: (5.0, -3.0), ..Default::default() });
    assert!((b[0].points[0].0 - a[0].points[0].0 - 5.0).abs() < 1e-4);
    assert!((b[0].points[0].1 - a[0].points[0].1 + 3.0).abs() < 1e-4);
    // 线性渐变: 亚像素边缘落在 128 对应的位置 (x=12.8 px), 非亚像素则在格边中点
    let ramp = gfn(30, 10, |x, _| (x as f32 * 10.0).min(255.0));
    let c = extract_contours(&ramp, &ContourParams::default());
    let min_x = c[0].points.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    assert!((min_x - 12.8).abs() < 0.05, "{min_x}");
}
fn gfn(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Gray {
    Gray { width: w, height: h, data: (0..w * h).map(|i| f(i % w, i / w)).collect() }
}
