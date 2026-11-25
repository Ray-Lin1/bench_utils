# BEV 车道线遮挡

`bev_occlusion_mask(points_xy, boxes, origin_xy=(0.0, 0.0))` 接受 `(N, 2)` 的车道线采样点与 `(M, 5)` 的障碍物旋转框，返回 `(N,)` 的布尔数组。框格式为 `[cx, cy, length, width, yaw]`，逆时针 yaw 使用弧度。输入使用同一帧的同一 BEV 米制坐标。自车框和评测范围由调用方处理。

自车原点到采样点的有限线段在终点前穿过任一框内部时输出 `True`。框内与框后点都可以命中。仅擦边、擦角、刚到框近侧边界和零长度线段输出 `False`。该定义对应 360° 虚拟自车视点下的平面遮挡。

## 在 WSL 中复现

进入 `/mnt/e/ai_learning/bench_utils/bev_bench`，在装有 `pip` 的 Python 3.10 环境执行：

```bash
python3 -m pip install -r requirements-review.txt
OPENBLAS_NUM_THREADS=1 python3 -B validate_bev_occlusion.py --output bev_review
```

验证脚本生成 20 个合成交通帧，检查车辆 footprint 互不重叠，并使用独立的分离轴判定验证每个点的 mask。额外检查框角擦碰、近侧边界、旋转和平移、后向射线及空数组。

`bev_review` 包含逐帧图、总览图、逐帧耗时 CSV 和运行环境摘要。绿色为无遮挡，红色为遮挡。每帧预热 20 次，再计时 200 次。耗时涵盖 `bev_occlusion_mask` 的输入检查、数组分配与几何计算。场景构建、交叉验证、绘图和文件 IO 在计时区间之外。
