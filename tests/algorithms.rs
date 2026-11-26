//! 为每一种算法生成一个独立测试: 以 tests/test_image2.jpg 为输入, 效果图输出到 tests/output/<name>.png。
use rust_image_algorithm::gallery::{cases, Case};
use rust_image_algorithm::{Image, Interp};
use std::sync::OnceLock;

/// 输入图缩放到长边 360 像素 (原图 2000 像素级, 直接跑慢速算法太耗时)。
const MAX_SIDE: usize = 360;

fn all() -> &'static Vec<Case> {
    static CASES: OnceLock<Vec<Case>> = OnceLock::new();
    CASES.get_or_init(|| {
        let orig = Image::load("tests/test_image2.jpg").expect("tests/test_image2.jpg 必须存在");
        let k = MAX_SIDE as f32 / orig.width.max(orig.height) as f32;
        let src = rust_image_algorithm::transform::resize(&orig, ((orig.width as f32 * k) as usize).max(1), ((orig.height as f32 * k) as usize).max(1), Interp::Bicubic);
        std::fs::create_dir_all("tests/output").unwrap();
        src.save("tests/output/00_source.png").unwrap();
        cases(&src)
    })
}

fn run(name: &str) {
    let c = all().iter().find(|c| c.name == name).unwrap_or_else(|| panic!("未注册的算法 {name}"));
    assert!(c.image.width > 0 && c.image.height > 0, "{name} 输出尺寸为 0");
    assert_eq!(c.image.data.len(), c.image.width * c.image.height * 4);
    c.image.save(&format!("tests/output/{name}.png")).unwrap();
}

macro_rules! algo_tests {
    ($($name:ident),* $(,)?) => {
        $( #[test] fn $name() { run(stringify!($name)); } )*
        const NAMES: &[&str] = &[$(stringify!($name)),*];
    };
}

algo_tests!(
    color_grayscale,
    color_brightness_contrast,
    color_invert,
    color_gamma,
    color_levels,
    color_curves,
    color_hue_saturation,
    color_vibrance,
    color_channel_mixer,
    color_sepia,
    color_posterize,
    color_threshold,
    color_exposure,
    color_balance,
    color_photo_filter,
    color_auto_levels,
    color_equalize,
    color_vignette,
    color_gradient_map,
    space_lab_contrast,
    space_lab_saturation,
    space_channel_L,
    space_channel_S,
    space_channel_Cr,
    blend_multiply,
    blend_screen,
    blend_overlay,
    blend_soft_light,
    blend_difference,
    blend_masked,
    filter_gaussian,
    filter_box,
    filter_motion,
    filter_radial_spin,
    filter_radial_zoom,
    filter_bilateral,
    filter_median,
    filter_sharpen,
    filter_unsharp,
    filter_high_pass,
    filter_emboss,
    filter_noise,
    filter_pixelate,
    filter_oil,
    filter_halftone,
    filter_sketch,
    filter_despeckle,
    adv_kuwahara,
    adv_nlm,
    adv_lens_blur,
    adv_bloom,
    adv_dehaze,
    adv_cartoon,
    edge_roberts,
    edge_prewitt,
    edge_sobel,
    edge_scharr,
    edge_laplacian,
    edge_log,
    edge_dog,
    edge_canny,
    edge_find_edges_color,
    edge_hough_lines,
    edge_hough_circles,
    feature_harris,
    bin_otsu,
    bin_triangle,
    bin_adaptive,
    morph_erode,
    morph_dilate,
    morph_open,
    morph_close,
    morph_gradient,
    morph_top_hat,
    morph_skeleton,
    bin_remove_small,
    bin_fill_holes,
    bin_distance,
    contour_all,
    contour_simplified,
    contour_canny_on_edges,
    contour_precise_holes,
    contour_precise_no_holes,
    contour_precise_simplified,
    contour_offset_miter,
    contour_offset_round,
    contour_offset_bevel,
    contour_inset,
    contour_translate,
    contour_threshold_invert,
    tf_resize_nearest,
    tf_resize_bilinear,
    tf_resize_bicubic,
    tf_crop,
    tf_flip_h,
    tf_flip_v,
    tf_rotate90,
    tf_rotate30,
    tf_affine,
    tf_skew,
    tf_perspective_distort,
    tf_perspective_rectify,
    dist_swirl,
    dist_pinch_bulge,
    dist_pinch_squeeze,
    dist_spherize,
    dist_ripple,
    dist_wave,
    dist_barrel,
    dist_pincushion,
    dist_fisheye,
    dist_polar,
    dist_polar_inverse,
    dist_cylinder,
    dist_shear_curve,
    dist_mesh_warp,
    dist_liquify,
    dist_displace,
);

/// 保证上面的列表与 gallery 注册表完全一致, 新增算法漏写测试时会失败。
#[test]
fn test_list_matches_registry() {
    let mut reg: Vec<&str> = all().iter().map(|c| c.name).collect();
    let mut names = NAMES.to_vec();
    reg.sort();
    names.sort();
    assert_eq!(reg, names);
}

/// 生成 tests/output/output.md: 每个算法的分类、说明、调用代码、输出尺寸与效果图。
#[test]
fn write_output_md() {
    let list = all();
    let mut md = String::from("# 输出结果说明\n\n输入: `tests/test_image2.jpg` (缩放到长边 360 像素, 见 `00_source.png`), 输出: 本目录下同名 PNG。\n由 `cargo test --release --test algorithms` 自动生成, 请勿手工编辑。\n\n![源图](00_source.png)\n");
    let mut cat = "";
    for c in list.iter() {
        if c.category != cat {
            cat = c.category;
            md += &format!("\n## {cat}\n");
        }
        md += &format!(
            "\n### {} (`{}`)\n\n- 输出尺寸: {}x{}\n- 调用: `{}`\n\n![{}]({}.png)\n",
            c.title, c.name, c.image.width, c.image.height, c.code.replace('\n', " "), c.title, c.name
        );
    }
    std::fs::create_dir_all("tests/output").unwrap();
    std::fs::write("tests/output/output.md", md).unwrap();
}