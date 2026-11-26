//! 基础图像类型与读写 (无第三方依赖)。
//!
//! * [`Image`]: 8 位 RGBA 图像, 行优先存储, 每像素 4 字节。
//! * [`Gray`]: 单通道浮点图 (范围通常为 0..=255), 供边缘检测/形态学等算法使用。
//! * 支持 PPM(P6)/PGM(P5) 读写以及 24 位 BMP 写入, 便于直接查看结果。

use std::fs;
use std::io;
use std::path::Path;

/// 8 位 RGBA 图像。
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    /// 长度 = width*height*4, 排列 R,G,B,A。
    pub data: Vec<u8>,
}

/// 单通道浮点图。
#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

/// 越界像素的处理方式 (对应 Photoshop 滤镜的边缘处理)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// 复制边缘像素 (Clamp / Replicate)。
    Clamp,
    /// 镜像反射 (Reflect)。
    Reflect,
    /// 平铺环绕 (Wrap)。
    Wrap,
    /// 常量 0 (透明黑)。
    Zero,
}

/// 将可能越界的坐标按 [`Border`] 规则映射到 `0..n`; `Zero` 且越界时返回 `None`。
pub fn map_coord(i: isize, n: usize, b: Border) -> Option<usize> {
    let n = n as isize;
    if (0..n).contains(&i) {
        return Some(i as usize);
    }
    match b {
        Border::Clamp => Some(i.clamp(0, n - 1) as usize),
        Border::Wrap => Some(i.rem_euclid(n) as usize),
        Border::Reflect => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * (n - 1);
            let m = i.rem_euclid(period);
            Some(if m < n { m } else { period - m } as usize)
        }
        Border::Zero => None,
    }
}

impl Image {
    /// 创建纯色图像。
    pub fn new(width: usize, height: usize, fill: [u8; 4]) -> Self {
        let mut data = Vec::with_capacity(width * height * 4);
        for _ in 0..width * height {
            data.extend_from_slice(&fill);
        }
        Image { width, height, data }
    }

    /// 由 RGBA 缓冲创建, 长度不符时 panic。
    pub fn from_rgba(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height * 4, "RGBA buffer size mismatch");
        Image { width, height, data }
    }

    /// 由生成函数 `f(x,y) -> [r,g,b,a]` 创建。
    pub fn from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> [u8; 4]) -> Self {
        let mut data = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&f(x, y));
            }
        }
        Image { width, height, data }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [u8; 4] {
        let i = (y * self.width + x) * 4;
        [self.data[i], self.data[i + 1], self.data[i + 2], self.data[i + 3]]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, p: [u8; 4]) {
        let i = (y * self.width + x) * 4;
        self.data[i..i + 4].copy_from_slice(&p);
    }

    /// 带边缘处理的取像素, `Zero` 越界返回透明黑。
    pub fn get_border(&self, x: isize, y: isize, b: Border) -> [u8; 4] {
        match (map_coord(x, self.width, b), map_coord(y, self.height, b)) {
            (Some(x), Some(y)) => self.get(x, y),
            _ => [0, 0, 0, 0],
        }
    }

    /// 对每个像素应用 `f(r,g,b,a)`。
    pub fn map_pixels(&self, f: impl Fn([u8; 4]) -> [u8; 4]) -> Image {
        let mut out = self.clone();
        for p in out.data.chunks_exact_mut(4) {
            let r = f([p[0], p[1], p[2], p[3]]);
            p.copy_from_slice(&r);
        }
        out
    }

    /// 提取单个通道 (0=R,1=G,2=B,3=A) 为浮点灰度图。
    pub fn channel(&self, c: usize) -> Gray {
        Gray {
            width: self.width,
            height: self.height,
            data: self.data.chunks_exact(4).map(|p| p[c] as f32).collect(),
        }
    }

    /// 按 Rec.601 亮度转浮点灰度图。
    pub fn to_gray(&self) -> Gray {
        Gray {
            width: self.width,
            height: self.height,
            data: self
                .data
                .chunks_exact(4)
                .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
                .collect(),
        }
    }

    /// 读取 JPEG/PNG/BMP 等常见格式 (基于 `image` crate), 格式由扩展名/内容自动识别。
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Image, ::image::ImageError> {
        let rgba = ::image::open(path)?.to_rgba8();
        Ok(Image { width: rgba.width() as usize, height: rgba.height() as usize, data: rgba.into_raw() })
    }

    /// 保存为 PNG/JPEG/BMP (由扩展名决定格式)。
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ::image::ImageError> {
        ::image::save_buffer(path, &self.data, self.width as u32, self.height as u32, ::image::ColorType::Rgba8)
    }

    /// 读取 PPM(P6)/PGM(P5) 二进制文件。
    pub fn load_pnm<P: AsRef<Path>>(path: P) -> io::Result<Image> {
        let bytes = fs::read(path)?;
        let err = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let mut pos = 0usize;
        let token = |pos: &mut usize| -> Option<String> {
            loop {
                while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
                    *pos += 1;
                }
                if *pos < bytes.len() && bytes[*pos] == b'#' {
                    while *pos < bytes.len() && bytes[*pos] != b'\n' {
                        *pos += 1;
                    }
                } else {
                    break;
                }
            }
            let s = *pos;
            while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            if s == *pos { None } else { Some(String::from_utf8_lossy(&bytes[s..*pos]).into()) }
        };
        let magic = token(&mut pos).ok_or_else(|| err("missing magic"))?;
        let w: usize = token(&mut pos).and_then(|s| s.parse().ok()).ok_or_else(|| err("bad width"))?;
        let h: usize = token(&mut pos).and_then(|s| s.parse().ok()).ok_or_else(|| err("bad height"))?;
        let _max = token(&mut pos).ok_or_else(|| err("bad maxval"))?;
        pos += 1;
        let ch = match magic.as_str() {
            "P6" => 3,
            "P5" => 1,
            _ => return Err(err("only P5/P6 supported")),
        };
        if bytes.len() < pos + w * h * ch {
            return Err(err("truncated data"));
        }
        let src = &bytes[pos..];
        Ok(Image::from_fn(w, h, |x, y| {
            let i = (y * w + x) * ch;
            if ch == 3 { [src[i], src[i + 1], src[i + 2], 255] } else { [src[i], src[i], src[i], 255] }
        }))
    }

    /// 写入 PPM(P6), 忽略 alpha。
    pub fn save_ppm<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut out = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
        for p in self.data.chunks_exact(4) {
            out.extend_from_slice(&p[..3]);
        }
        fs::write(path, out)
    }

    /// 写入 24 位 BMP (大多数系统可直接预览), 忽略 alpha。
    pub fn save_bmp<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let row = (self.width * 3 + 3) & !3;
        let size = 54 + row * self.height;
        let mut o = Vec::with_capacity(size);
        o.extend_from_slice(b"BM");
        o.extend_from_slice(&(size as u32).to_le_bytes());
        o.extend_from_slice(&[0; 4]);
        o.extend_from_slice(&54u32.to_le_bytes());
        o.extend_from_slice(&40u32.to_le_bytes());
        o.extend_from_slice(&(self.width as i32).to_le_bytes());
        o.extend_from_slice(&(self.height as i32).to_le_bytes());
        o.extend_from_slice(&1u16.to_le_bytes());
        o.extend_from_slice(&24u16.to_le_bytes());
        o.extend_from_slice(&[0; 24]);
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                let p = self.get(x, y);
                o.extend_from_slice(&[p[2], p[1], p[0]]);
            }
            o.resize(o.len() + row - self.width * 3, 0);
        }
        fs::write(path, o)
    }
}

impl Gray {
    pub fn new(width: usize, height: usize, v: f32) -> Self {
        Gray { width, height, data: vec![v; width * height] }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x] = v;
    }

    /// 带边缘处理取值, `Zero` 越界为 0。
    #[inline]
    pub fn get_border(&self, x: isize, y: isize, b: Border) -> f32 {
        match (map_coord(x, self.width, b), map_coord(y, self.height, b)) {
            (Some(x), Some(y)) => self.get(x, y),
            _ => 0.0,
        }
    }

    /// 转为 RGBA 图像 (值截断到 0..=255)。
    pub fn to_image(&self) -> Image {
        let mut data = Vec::with_capacity(self.data.len() * 4);
        for &v in &self.data {
            let c = v.round().clamp(0.0, 255.0) as u8;
            data.extend_from_slice(&[c, c, c, 255]);
        }
        Image { width: self.width, height: self.height, data }
    }

    /// 线性拉伸到 0..=255 (用于可视化梯度等)。
    pub fn normalize(&self) -> Gray {
        let (mn, mx) = self.data.iter().fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        let s = if mx > mn { 255.0 / (mx - mn) } else { 0.0 };
        Gray { width: self.width, height: self.height, data: self.data.iter().map(|v| (v - mn) * s).collect() }
    }
}

#[inline]
pub(crate) fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
