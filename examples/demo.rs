//! 演示: `cargo run --release --example demo [输入.ppm] [输出目录]`
//! 未提供输入时使用内置合成图。输出 BMP 文件到 `out/` 目录。

use rust_image_algorithm::binary::*;
use rust_image_algorithm::color::*;
use rust_image_algorithm::distort::*;
use rust_image_algorithm::edge::*;
use rust_image_algorithm::filter::*;
use rust_image_algorithm::transform::*;
use rust_image_algorithm::{Image, Interp};

/// 合成测试图: 渐变背景 + 圆 + 矩形 + 网格。
pub fn synthetic(w: usize, h: usize) -> Image {
    Image::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32, y as f32);
        let mut p = [(40.0 + 120.0 * fx / w as f32) as u8, (60.0 + 100.0 * fy / h as f32) as u8, 150, 255];
        if (fx - w as f32 * 0.35).powi(2) + (fy - h as f32 * 0.5).powi(2) < (h as f32 * 0.25).powi(2) {
            p = [230, 200, 60, 255];
        }
        if x > w * 11 / 20 && x < w * 17 / 20 && y > h / 4 && y < h * 3 / 4 {
            p = [200, 60, 60, 255];
        }
        if x % 32 == 0 || y % 32 == 0 {
            p = [255, 255, 255, 255];
        }
        p
    })
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let img = match args.get(1) {
        Some(p) => Image::load_pnm(p)?,
        None => synthetic(320, 240),
    };
    let dir = args.get(2).cloned().unwrap_or_else(|| "out".into());
    std::fs::create_dir_all(&dir)?;
    let save = |name: &str, i: &Image| i.save_bmp(format!("{dir}/{name}.bmp"));
    let (w, h) = (img.width as f32, img.height as f32);
    let c = (w / 2.0, h / 2.0);
    let b = Interp::Bilinear;

    save("00_original", &img)?;
    // 调色
    save("color_grayscale", &grayscale(&img, GrayMode::Luma709))?;
    save("color_brightness_contrast", &brightness_contrast(&img, 20.0, 30.0))?;
    save("color_curves", &curves(&img, &[(0.0, 0.0), (64.0, 40.0), (192.0, 220.0), (255.0, 255.0)], &[(0.0, 0.0), (255.0, 255.0)], &[(0.0, 0.0), (255.0, 255.0)]))?;
    save("color_hue_sat", &hue_saturation(&img, 40.0, 20.0, 0.0))?;
    save("color_sepia", &sepia(&img))?;
    save("color_vignette", &vignette(&img, 0.8, 0.4, 0.6))?;
    // 滤镜
    save("filter_gaussian", &gaussian_blur(&img, 3.0))?;
    save("filter_motion", &motion_blur(&img, 30.0, 15))?;
    save("filter_unsharp", &unsharp_mask(&img, 1.5, 2.0, 0.0))?;
    save("filter_emboss", &emboss(&img, 135.0, 1.0))?;
    save("filter_pixelate", &pixelate(&img, 8))?;
    save("filter_oil", &oil_paint(&img, 3, 20))?;
    save("filter_sketch", &sketch(&img, 5.0, 1.0))?;
    // 边缘与轮廓
    save("edge_sobel", &gradient_edges(&img, GradientOperator::Sobel, 1.0, 0.0).to_image())?;
    let canny_edges = canny(&img, CannyParams::default());
    save("edge_canny", &canny_edges.to_image())?;
    let bin = binarize(&img.to_gray(), otsu_threshold(&img.to_gray()), false);
    let contours = find_contours(&bin, 10);
    let mut drawn = img.clone();
    draw_contours(&mut drawn, &contours, [255, 0, 255, 255], 2);
    save("contours", &drawn)?;
    save("morph_skeleton", &skeletonize(&bin).to_image())?;
    // 变换
    save("tf_rotate30", &rotate(&img, 30.0, true, Interp::Bicubic, [0, 0, 0, 255]))?;
    save("tf_skew", &skew(&img, 20.0, 0.0, b, [0, 0, 0, 255]))?;
    let quad = [(40.0, 20.0), (w - 20.0, 0.0), (w - 60.0, h - 10.0), (0.0, h - 40.0)];
    if let Some(p) = perspective_distort(&img, &quad, img.width, img.height, b, [0, 0, 0, 255]) {
        save("tf_perspective", &p)?;
    }
    // 扭曲
    save("dist_swirl", &swirl(&img, c, h / 2.0, 120.0, b))?;
    save("dist_pinch", &pinch(&img, c, h / 2.0, 0.6, b))?;
    save("dist_ripple", &ripple(&img, 4.0, 40.0, b))?;
    save("dist_wave", &wave(&img, 8.0, 60.0, 0.0, false, b))?;
    save("dist_barrel", &lens_distortion(&img, 0.3, 0.0, b))?;
    save("dist_fisheye", &fisheye(&img, 0.8, b))?;
    save("dist_polar", &polar(&img, true, b))?;
    save("dist_spherize", &spherize(&img, c, h / 2.0, b))?;
    println!("done -> {dir}/");
    Ok(())
}
