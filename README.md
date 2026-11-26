# rust_image_algorithm

参考 Photoshop 等专业图像软件实现的 **纯 Rust、零第三方依赖** 图像处理算法库: 调色、滤镜、几何变换、扭曲变形、边缘检测、二值化/形态学、轮廓提取。

```text
cargo test                                   # 单元测试
cargo run --release --example demo           # 用合成图生成全部示例效果到 out/*.bmp
cargo run --release --example demo in.ppm d  # 处理自己的 PPM(P6/P5) 图片
```

## 数据类型 (`image.rs`)

| 类型 | 说明 |
| --- | --- |
| `Image` | 8 位 RGBA, 行优先。`from_fn / get / set / map_pixels / to_gray / channel` |
| `Gray` | 单通道 `f32` 图 (边缘、二值图、形态学使用; 二值图前景 255 / 背景 0) |
| `Border` | 越界处理: `Clamp` 复制边缘 / `Reflect` 镜像 / `Wrap` 平铺 / `Zero` 补 0 |
| IO | `load_pnm`(P5/P6)、`save_ppm`、`save_bmp`(24 位, 可直接预览) |

> 为保持零依赖, 只内置 PPM/BMP 读写; 需要 PNG/JPEG 时可在应用层用 `image` crate 解码后用 `Image::from_rgba` 转入。

## 已实现的方法

### 调色 `color.rs` (Photoshop「图像 > 调整」)
`grayscale`(601/709/平均/明度) · `brightness_contrast` · `invert` · `gamma` · `levels`(色阶) · `curves`(曲线, 单调三次插值) · `hue_saturation` · `vibrance`(自然饱和度) · `channel_mixer` · `sepia` · `posterize` · `threshold` · `exposure` · `color_balance` · `photo_filter` · `histogram` · `auto_levels` · `equalize`(直方图均衡) · `vignette`(暗角) · `gradient_map`(渐变映射) · `rgb_to_hsl / hsl_to_rgb`

### 滤镜 `filter.rs`
- 卷积: `convolve` / `convolve_gray`(任意核、除数、偏移、边缘模式), 可分离卷积 `separable_gray`
- 模糊: `gaussian_blur` · `box_blur` · `motion_blur`(动感) · `radial_blur`(旋转/缩放) · `bilateral`(双边, 保边/表面模糊)
- 锐化: `sharpen` · `unsharp_mask`(USM: 数量/半径/阈值) · `high_pass`(高反差保留)
- 降噪: `median` · `bilateral` · `despeckle`
- 风格化: `emboss`(浮雕) · `pixelate`(马赛克) · `oil_paint`(油画) · `halftone`(半调) · `sketch`(素描) · `add_noise`(高斯/均匀, 单色/彩色)

### 边缘检测 `edge.rs`
`gradient`(Roberts/Prewitt/Sobel/Scharr, 输出 gx/gy/幅值/方向) · `gradient_edges` · `laplacian`(4/8 邻域) · `laplacian_of_gaussian`(LoG 零交叉) · `difference_of_gaussians`(DoG) · **`canny`**(高斯平滑+NMS+双阈值滞后) · `find_edges_color`(PS「查找边缘」) · `hough_lines`(霍夫直线)

### 二值化 / 形态学 / 轮廓 `binary.rs`
- 阈值: `binarize` · `otsu_threshold` · `triangle_threshold` · `adaptive_threshold`(均值/高斯)
- 形态学(矩形/十字/椭圆结构元): `erode` · `dilate` · `open` · `close` · `morph_gradient` · `top_hat` · `black_hat` · `skeletonize`(Zhang-Suen)
- 区域: `connected_components`(8 连通) · `remove_small_components` · `fill_holes` · `distance_transform`(精确欧氏)
- **轮廓提取**: `find_contours`(Moore 边界跟踪+Jacob 终止, 外轮廓) · `simplify_contour`(Douglas-Peucker) · `contour_area` · `contour_perimeter` · `bounding_rect` · `draw_contours`(Bresenham, 可设线宽)

### 几何变换 `transform.rs` (Photoshop「编辑 > 变换」)
`resize / scale`(最近邻/双线性/双三次, 缩小自动预滤波) · `crop` · `flip_horizontal/vertical` · `rotate90` · `rotate`(任意角度, 可扩展画布) · `warp_affine`(任意仿射) · `skew`(斜切) · **透视**: `homography_from_points`(4 点求单应) · `warp_perspective` · `perspective_rectify`(四边形拉正, 文档矫正) · `perspective_distort`(整图映射到任意四边形) · 通用 `remap`

### 扭曲变形 `distort.rs` (「滤镜 > 扭曲」「液化」「变形」)
`swirl`(旋转扭曲) · `pinch`(挤压/膨胀) · `spherize`(球面化) · `ripple`(波纹) · `wave`(波浪) · `lens_distortion`(桶形/枕形, Brown-Conrady) · `fisheye`(鱼眼) · `polar`(极坐标 ↔ 直角) · `cylinder`(圆柱) · `shear_curve`(切变曲线) · `mesh_warp`(网格变形, 逐单元双线性反解) · `liquify_push`(液化向前推) · `displace`(置换图)

## 原理与适用性说明

| 算法 | 原理 | 适用 / 注意 |
| --- | --- | --- |
| 反向映射 + 插值 | 遍历**目标**像素求源坐标再采样, 无空洞 | 所有几何/扭曲通用; 最近邻快但有锯齿, 双线性通用, 双三次最锐利(可能有轻微振铃) |
| 缩放预滤波 | 缩小 >2 倍前先高斯模糊, 满足采样定理 | 避免摩尔纹/混叠; 极大缩小可考虑多级金字塔 |
| 高斯模糊 | 可分离核, 复杂度 O(N·k) 而非 O(N·k²) | 降噪/平滑首选, σ 越大越慢(O(σ)); 超大 σ 可用多次 box blur 近似 |
| USM 锐化 | `src + a·(src−blur)`, 阈值保护平坦区 | 照片锐化; 过量会产生光晕 |
| 中值滤波 | 窗口排序取中值 | 椒盐噪声最佳, 保边; O(r²log r²) 大半径较慢 |
| 双边滤波 | 空间权重 × 颜色权重 | 磨皮/保边平滑; 暴力实现较慢 |
| Sobel/Scharr | 一阶差分+平滑 | 快速粗边缘; Scharr 方向更精确; Roberts 对噪声敏感 |
| Canny | 平滑→梯度→非极大抑制→双阈值滞后 | 细而连续的单像素边缘, 通用最优; 阈值需按图调整(高:低≈2~3:1) |
| LoG/DoG | 二阶导零交叉 / 两个高斯之差 | 对噪声需较大 σ; DoG 是 LoG 快速近似, 适合斑点与素描风 |
| Otsu / 三角法 | 类间方差最大 / 直方图到端点连线最远 | Otsu 适合双峰; 三角法适合单峰(目标占比小) |
| 自适应阈值 | 与局部均值比较 | 光照不均的文档/文字 |
| 形态学 | 局部 min/max | 开运算去小亮点, 闭运算补孔洞, 顶帽校正不均光照 |
| Moore 边界跟踪 | 沿 8 邻域顺时针追踪边界, Jacob 判据终止 | 提取外轮廓, 速度快; 不含内孔(对反相图再调用即可得孔洞); 输入应已二值化 |
| Douglas-Peucker | 递归保留偏差>ε 的点 | 轮廓/折线简化、矢量化 |
| 单应矩阵 | 4 点对解 8 元线性方程 (高斯消元, 选主元) | 透视矫正/贴图; 点共线返回 `None`; 点序须一致(左上,右上,右下,左下) |
| 极坐标/旋转/球面/挤压 | 把像素半径/角度做非线性重映射 | 中心与半径可调; 半径外保持不变, 保证连续 |
| 镜头畸变 | `r' = r(1+k1r²+k2r⁴)` | k1>0 桶形, <0 枕形; 可用于镜头校正(取反参数近似) |
| 网格变形 | 每个单元四边形用牛顿法反解 (u,v) | 类 PS「变形」; 控制点不应使单元翻转 |

通用性能说明: 当前实现以**清晰、可读、正确**为主 (单线程、f32), 大图上的中值/双边/油画/形态学为暴力算法, 可后续用积分图、滑动直方图、rayon 并行优化。

## 还可以实现 (TODO)

- 编解码: PNG/JPEG/WebP (可接 `image` crate)、ICC 色彩管理、16 位/浮点图像
- 色彩空间: Lab/LCH/HSV/YCbCr、可选颜色(Selective Color)、黑白(Black & White)调整、阴影/高光、色调均化(HDR Toning)、LUT(3D LUT 加载)
- 滤镜: 镜头模糊/景深(Lens Blur)、表面模糊优化、Kuwahara、非局部均值(NLM)/BM3D 降噪、智能锐化(Smart Sharpen)、去雾(Dark Channel Prior)、颗粒/胶片模拟、光照效果、云彩/柏林噪声、玻璃/水彩/海报边缘
- 边缘/特征: 带孔洞的层级轮廓(Suzuki-Abe)、Hough 圆/线段、角点(Harris/FAST/Shi-Tomasi)、SIFT/ORB 特征、分水岭/GrabCut/超像素(SLIC)分割、光流
- 变换: Lanczos/Mitchell 插值、内容感知缩放(Seam Carving)、傅里叶变换(频域滤波)、图像拼接/全景、薄板样条(TPS)变形、人像液化(膨胀/收缩/重建画笔)
- 合成: 图层混合模式(正片叠底/滤色/叠加/柔光等)、蒙版/羽化、泊松融合(Seamless Clone)、修复画笔/仿制图章/内容识别填充
- 工程: rayon 并行、SIMD、积分图加速、分块处理大图、`criterion` 基准测试

## 许可

MIT


---

## 扩展模块 (第二阶段)

依赖: 仅 `image` (jpeg/png/bmp) 用于 `Image::load` / `Image::save`, 算法本身全部自研。

| 模块 | 内容 |
|------|------|
| `blend` | 14 种 PS 混合模式 (正片叠底/滤色/叠加/柔光/强光/差值/排除…) + 蒙版 + 羽化 |
| `space` | HSV / Lab / YCbCr 转换, Lab 对比度/饱和度, 通道视图 |
| `advanced` | Kuwahara 绘画、非局部均值降噪、镜头模糊(光圈)、Bloom 辉光、去雾(暗通道)、卡通化、Harris 角点、Hough 圆检测 |

## 示例输出与测试方法

测试图: `tests/test_image.jpg` (100x100, 示例中双三次放大到 300px)。

```
cargo test --release                              # 单元测试 + 全部示例回归测试
cargo run --release --example gallery             # 生成 docs/images/*.png 与 docs/GALLERY.md
cargo run --release --example gallery <图片> <尺寸>  # 自定义输入
```

`src/gallery.rs` 是所有示例的唯一注册表 (109 个, 含调用代码); `tests/gallery.rs` 校验每个算法能运行、尺寸有效、结果相对输入有变化。
每种算法的输出图片与调用代码见 [docs/GALLERY.md](docs/GALLERY.md)。
## 高精度轮廓提取 (`contour` 模块)

`extract_contours(&gray, &ContourParams)` 基于 Marching Squares + 线性插值, 返回亚像素浮点闭合路径 `ContourPath { points, is_hole, area, parent }`。

| 参数 | 说明 |
|------|------|
| `threshold` / `invert` | 阈值 (>= 为前景) / 反相取暗部 |
| `smooth_sigma` | 预高斯模糊, 抑制噪点与锯齿 |
| `subpixel` | 亚像素线性插值 (关闭则取格边中点) |
| `keep_holes` | 是否保留孔洞 (内轮廓); `parent` 给出嵌套层级 |
| `min_area` / `max_area` / `min_length` | 面积、周长过滤 |
| `simplify_epsilon` | Douglas–Peucker 简化容差 |
| `smooth_iterations` | Chaikin 切角平滑次数 |
| `offset` + `join` (`Miter`/`Round`/`Bevel`) + `miter_limit` + `arc_step_deg` | 沿法线外扩(+)/内缩(-), 拐角类型, 尖角限制, 圆角精度 |
| `translate` | 整体平移 |

配套: `draw_paths` 绘制、`paths_to_svg` 导出 SVG、`signed_area` / `path_length` / `point_in_polygon`。
局限: 偏移不做自相交清理, 偏移量远大于局部曲率半径时内侧可能出现小回环。示例见 `tests/output/output.md` 中 `contour_*`。