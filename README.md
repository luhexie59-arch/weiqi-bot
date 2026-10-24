# 围棋图片计数

Python 本地桌面程序：从棋盘截图或照片识别黑白子，统计**当前棋子数 + 己方独占空点数**，并提供可修改的棋盘和占地示意图。支持 9、13、19 路；不需要联网服务或模型 API。

## Telegram Bot 服务端（long polling）

入口为 **`telegram_bot.py`**，固定按 19×19 棋盘识别，沿用本项目的面积计数规则。新增依赖 `python-telegram-bot` 已安装在 `.venv` 中。

在项目目录的 PowerShell 中设置自己的 Bot Token，然后启动：

```powershell
$env:TGBOT_WEIQI_API_KEY = "从 BotFather 获取的 Token"
.\.venv\Scripts\python.exe telegram_bot.py
```

也支持通用变量 `TELEGRAM_BOT_TOKEN`；两个变量同时存在时优先使用 `TGBOT_WEIQI_API_KEY`。代码不内置 Token，也不会借用其他机器人的环境变量。如果 Token 已设置在进程可继承的环境变量中，也可双击 `start_bot.bat`。Ctrl+C 停止。

服务端运行后，用户直接向机器人发照片即可。每张照片独立处理；相册里有多张照片时逐张回复。以文件形式发送的图片也可以处理，普通文字、命令、非图片文件和编辑消息均不触发回复。无需 `/start`、按钮、登录或其他操作。

成功时，只回复一条**透视校正后的标注照片**，照片说明严格为两行：

```text
黑子：7
白子: 7
```

其中数字是**棋子数 + 己方独占空点数**，不是单纯的落子数量。图中青色表示黑方标记、粉色表示白方标记，方点是空地归属、灰点是中立空点，橙框表示识别把握较低的交点。识别或下载失败时仅回复一句简短错误，不提供交互菜单。

图片在内存中处理，不依赖桌面界面，也不自动保存用户照片到磁盘。识别工作在后台线程中执行，照片任务顺序处理，控制 CPU 和内存占用。启动时保留 Telegram 尚未投递的消息，并由 polling 启动流程移除已有 webhook；同一 Token 同时只运行一个 polling 实例。服务端需要能访问 Telegram，**无需域名、公网入站端口或 HTTPS 证书**。

Linux 服务器中，创建并安装依赖后用同一入口启动：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export TGBOT_WEIQI_API_KEY="从 BotFather 获取的 Token"
.venv/bin/python telegram_bot.py
```

Bot 的离线测试使用模拟 Telegram 消息和网络接口，不读取真实 Token，不连接 Telegram、不发送真实消息：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_telegram_bot -v
```

## 启动

本项目使用已有的 **`.venv`** 虚拟环境，已安装 NumPy、OpenCV、Pillow。Tkinter 随当前 Python 安装提供。

双击 **`start.bat`**，或者在本目录的 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

1. 选择正确的棋盘路数，然后点击“打开图片”。
2. 程序自动定位网格、校正透视并识别棋子。在“原图”页核对绿色四角，在“校正图”页核对照片与识别标记。
3. 若网格错位，点击“手动四角”，依次点击**最外侧网格交点**的左上、右上、右下、左下。不是木板的四个角；被棋子覆盖时可以按网格位置估计。Esc 取消。
4. 在“校正图”或“棋盘”页左键交点，默认按 **空 → 黑 → 白 → 空** 循环，也可以从下拉框指定黑子、白子或空点。橙框表示建议人工核对的交点，不能视为精确的错误概率。
5. **右键空点**，可以将整块连通空区标为中立，再次右键取消。适用于未完成棋局中不希望按几何边界归属的争议区域。修改棋子后，中立标记会清空，避免旧标记错误影响新的区域；Ctrl+Z 可以撤销。
6. 点击“导出结果”，同时保存 JSON、带统计的棋盘 PNG、透视校正后的原照片 PNG。“载入结果”可恢复 JSON 中的棋盘与中立标记，继续修改；不会自动重新打开原照片。

灵敏度默认为 1。调高更容易将有色差的点识别成棋子，调低更保守；修改后点击“重新识别棋子”，这会重置人工修改。自动识别也会重置人工修改。

## 棋子偏离交点时的识别

识别分两阶段：先检查交点中心；若置信度低于 0.25，再在水平、竖直方向各 **±⅓ 格间距** 的范围内搜索候选棋子中心。局部混入明显棋子颜色、黑白颜色证据冲突或浅色棋子只遮住部分网格线时，也会标为不确定点并重试。

扩大搜索会检查候选位置是否具有足够完整的棋子圆盘；浅色棋盘上的白子还会检查圆形边缘，减少将相邻棋子的边缘、星位或空白误识别为另一颗棋子。结果仍归入原来的交点，仍不确定的位置保留橙色核对框。这是对偏移棋子的额外容错，不能保证所有遮挡、重叠或严重反光情况下的准确率。

Bot 和桌面程序共用 `vision.py`，不需要新增依赖。部署 ZIP 只在明确要求重新打包时更新；修改代码后，如需部署此项增强，单独替换服务器上的 `vision.py` 并重启 Bot。

## 计数规则与未完成棋局

这里的“面积”以**交点数**为单位，不是照片中的像素面积，也不是只数空的“目”。

- 黑子、白子各占 1 点。
- 用上下左右四连通将空点分成连通区域，检查每个区域接触的棋子颜色。
- 只接触黑子的空区全部归黑，只接触白子的空区全部归白。
- 同时接触黑白双方、完全没有接触棋子，或被人工标为中立的空区不计入双方。
- 棋盘边界可以构成空区的边界，斜对角不算连通或接触。
- **不自动删除死子、不推演最优后续、不判断打劫或双活、不加贴目、不计算提子数。** 照片里的棋子都按现存棋子处理。若需要死子移除后的结果，可以先在界面把相应棋子改为空。

始终满足：`黑方面积 + 白方面积 + 中立空点 = 路数²`。

**未完成棋局没有仅凭几何边界就能确定的“实战安全控制面积”。** 本程序采用上述明确的当前棋面口径。例如，棋盘上只有一颗黑子时，其他空点是同一块空区且只接触黑色，默认会全部归黑；这不代表黑棋已在实战中控制全盘。若要保守计数，右键将该空区标为中立，此时黑方只计这颗棋子。类似地，一块空区即使大部分被黑棋包围，只要还有通路连接到白子，就保持中立，不估计黑棋未来能围住多少。

## 命令行

无参数启动桌面界面；提供图片路径则直接计数并导出文件：

```powershell
.\.venv\Scripts\python.exe main.py "你的棋盘照片.jpg" --size 19 --output output/my_board
```

输出包括 `my_board.json`、`my_board.png`、`my_board_corrected.png`。自动定位失败时会返回非零退出码及说明，可以使用图形界面选角，或在命令行提供四个网格角点的像素坐标：

```powershell
.\.venv\Scripts\python.exe main.py photo.jpg --size 19 --corners 120 80 920 100 950 900 90 930
```

角点坐标基于**按 EXIF 信息转正后的原图**，顺序为左上、右上、右下、左下。JSON 中的 `corners` 使用相同坐标系。

已有 JSON 可以跳过识别直接重算；`--neutral 行 列` 把对应空点所在的整块空区标为中立，行列均从 1 开始，可多次指定：

```powershell
.\.venv\Scripts\python.exe main.py --board examples/expected.json --neutral 1 2 --output output/recount
```

JSON 的 `board` 和 `score.ownership` 都是从上到下、从左到右的矩阵：`0=空/中立、1=黑、2=白`。`board` 保留棋子，`ownership` 给出含空地的最终归属。JSON 内的 `neutral_points` 则使用从 0 开始的 `[行, 列]`；这是与命令行不同的程序数据表示。`confidence` 为识别核对指标，不是统计意义上的概率。

## 示例与验证

生成一张含透视、渐变照明和少量反光的合成照片，然后识别：

```powershell
.\.venv\Scripts\python.exe make_demo.py
.\.venv\Scripts\python.exe main.py examples/demo_photo.png --output output/demo
.\.venv\Scripts\python.exe -m unittest discover -v
```

该示例的正确结果为：**黑 7 点（6 子 + 1 空点），白 7 点（6 子 + 1 空点），中立 347 点**。`examples/expected.json` / `.png` 保存了预期结果。

测试覆盖：单方眼、边角空地、双方相邻的中立区、死子不自动移除、中立标记、面积守恒、9/13/19 路棋子识别、密集棋子、星位排除、边线棋子、渐变照明、少量反光、浅色截图、手动与自动透视校正、相机 EXIF 旋转、无棋盘图片拒绝、中文路径及导出。

目前用可重复的合成图验证；**尚未针对你的真实摄像头照片验证准确率**。严重反光、手部遮挡、模糊、小分辨率、极端斜视、广角镜头畸变、带编号棋子或复杂背景都可能造成误判。建议使用完整、清晰、尽量正对棋盘的图片。透视校正不等于相机镜头畸变校正；必要时手动标定并逐点修正。

## 环境与依赖

当前环境：Windows、Python 3.13；依赖版本见 `requirements.txt`。`opencv-python-headless` 不创建 OpenCV 窗口，桌面界面由 Tkinter 提供。

重新创建环境时：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本机的 `wheels/` 保留了这次使用的 Windows 64 位 / CPython 3.13 安装包，网络不可用时可离线重装：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
```

其他平台或 Python 版本请在线安装兼容包；不要直接使用这里的 Windows wheels。

## 文件结构

| 文件 | 用途 |
| --- | --- |
| `main.py` | 桌面/命令行入口 |
| `telegram_bot.py` | Telegram long polling 服务端；固定 19×19，回复标注照片和两行面积统计 |
| `start_bot.bat` | Windows Bot 启动脚本 |
| `gui.py` | 图片显示、选四角、修改棋子、中立标记、撤销、导出 |
| `vision.py` | EXIF 转正、候选四边形、等距网格拟合、透视校正、局部棋子识别 |
| `scoring.py` | 四连通空区归属与面积统计，可独立调用 |
| `reporting.py` | 棋盘绘制与 JSON/PNG 导出 |
| `make_demo.py` | 生成已知答案的测试照片 |
| `tests/` | 计数与图像识别测试 |

图像变换与局部阈值处理参考 OpenCV 官方文档：[透视变换](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)、[自适应阈值](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)。
