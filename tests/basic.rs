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
