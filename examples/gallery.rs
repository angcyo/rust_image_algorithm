//! 用测试图生成全部算法的示例图与文档:
//! `cargo run --release --example gallery [输入图=tests/test_image.jpg] [基准边长=300]`
//! 输出: `docs/images/*.png` 与 `docs/GALLERY.md`。

use rust_image_algorithm::gallery::cases;
use rust_image_algorithm::transform::{resize, Interp};
use rust_image_algorithm::Image;
use std::fmt::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "tests/test_image.jpg".into());
    let size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let orig = Image::load(&path).expect("无法读取输入图片");
    // 原图只有 100x100 时效果不明显, 先双三次放大到 size 再处理
    let src = if orig.width.max(orig.height) < size {
        let s = size as f32 / orig.width.max(orig.height) as f32;
        resize(&orig, (orig.width as f32 * s) as usize, (orig.height as f32 * s) as usize, Interp::Bicubic)
    } else {
        orig.clone()
    };
    std::fs::create_dir_all("docs/images").unwrap();
    src.save("docs/images/00_source.png").unwrap();

    let t0 = Instant::now();
    let list = cases(&src);
    println!("{} 个示例, 耗时 {:.1?}", list.len(), t0.elapsed());

    let mut md = String::new();
    writeln!(md, "# 算法效果图集\n\n> 由 `cargo run --release --example gallery` 自动生成, 输入: `{path}` ({}x{}, 放大到 {}x{} 后处理)。\n", orig.width, orig.height, src.width, src.height).unwrap();
    writeln!(md, "原图:\n\n![source](images/00_source.png)\n").unwrap();
    let mut last = "";
    for c in &list {
        c.image.save(format!("docs/images/{}.png", c.name)).unwrap();
        if c.category != last {
            writeln!(md, "\n## {}\n", c.category).unwrap();
            last = c.category;
        }
        writeln!(md, "### {}\n\n<img src=\"images/{}.png\" width=\"300\">\n\n```rust\n{}\n```\n", c.title, c.name, c.code).unwrap();
    }
    std::fs::write("docs/GALLERY.md", md).unwrap();
}
