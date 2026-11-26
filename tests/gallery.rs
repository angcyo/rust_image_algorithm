use rust_image_algorithm::gallery::cases;
use rust_image_algorithm::{Image, Interp};

/// 对 tests/test_image.jpg 运行每一种算法: 尺寸合法、名称唯一, 且除"恒等"类外结果应与输入不同。
#[test]
fn every_algorithm_runs_on_test_image() {
    let orig = Image::load("tests/test_image.jpg").expect("tests/test_image.jpg 必须存在");
    let src = rust_image_algorithm::transform::resize(&orig, 120, 120, Interp::Bicubic);
    let list = cases(&src);
    assert!(list.len() > 100, "示例数量: {}", list.len());
    let mut names = std::collections::HashSet::new();
    let mut unchanged = Vec::new();
    for c in &list {
        assert!(names.insert(c.name), "重复名称 {}", c.name);
        assert!(c.image.width > 0 && c.image.height > 0, "{} 尺寸为 0", c.name);
        assert_eq!(c.image.data.len(), c.image.width * c.image.height * 4);
        if c.image == src {
            unchanged.push(c.name);
        }
    }
    assert!(unchanged.is_empty(), "以下算法没有产生任何变化: {unchanged:?}");
}
