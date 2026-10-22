# 用户手动启动：Painter-3D seed=1、2、3

这些脚本和命令仅作准备，未启动任何 screen、训练、推理或评价进程。
每个数据集按 1→2→3 顺序独立初始化训练，共 12 次训练；
不同 seed 不接着前一个 seed 的 checkpoint 训练。

使用独立 screen 会话 `painter3d`，其中四个窗口名为原数据集名称，
与已存在的旧版数据集会话区分。以下命令每项只执行一次；
如已有本会话，请直接重新连接，不要重复创建窗口。

```bash
cd /home/wzk/WorkSpace/ViTPose

screen -dmS painter3d -t acdc bash -c 'bash datasets_analysis/commands/painter3d/train_seeds.sh acdc 0 1 2 3; exec bash'
screen -S painter3d -X screen -t hc18 bash -c 'bash datasets_analysis/commands/painter3d/train_seeds.sh hc18 1 1 2 3; exec bash'
screen -S painter3d -X screen -t TG3K bash -c 'bash datasets_analysis/commands/painter3d/train_seeds.sh TG3K 1 1 2 3; exec bash'
screen -S painter3d -X screen -t ChestX-ray bash -c 'bash datasets_analysis/commands/painter3d/train_seeds.sh ChestX-ray 0 1 2 3; exec bash'
```

参数中，数据集名后的第一个数字是物理 GPU ID，后三个数字是 seed。
默认 GPU0 为 acdc + ChestX-ray，GPU1 为 hc18 + TG3K，每张卡并发两个任务；
每个任务仍为 batch size=8。可根据当时 GPU 占用手动调整。
训练脚本只训练，遇到失败停止后续 seed，并拒绝覆盖已有训练目录。
脚本完成或失败后，screen 窗口保留终端供检查。

```bash
screen -r painter3d
# 或直接进入指定窗口：
screen -r painter3d -p acdc
```

窗口内按 `Ctrl+a`，松开后按 `"` 列出窗口，选择数据集。
按 `Ctrl+a`，松开后按 `d` 退出连接，任务继续运行。
`screen -ls` 可查看会话。

## 训练完成后，手动推理和评价

训练脚本不会自动执行此阶段。确认训练结束后，进入对应窗口，手动执行其中一条：

```bash
# acdc 窗口
bash datasets_analysis/commands/painter3d/infer_eval_seeds.sh acdc 0 1 2 3

# hc18 窗口
bash datasets_analysis/commands/painter3d/infer_eval_seeds.sh hc18 1 1 2 3

# TG3K 窗口
bash datasets_analysis/commands/painter3d/infer_eval_seeds.sh TG3K 1 1 2 3

# ChestX-ray 窗口
bash datasets_analysis/commands/painter3d/infer_eval_seeds.sh ChestX-ray 0 1 2 3
```

每条命令按 seed 顺序执行测试集推理及评价。脚本检查训练成功标记、
最终 epoch checkpoint 和唯一的验证集最佳 checkpoint，
只使用验证 MRE 最佳权重；最佳 epoch 无需提前指定。
ACDC/HC18/ChestX-ray 使用 `best_MRE_mm_epoch_*.pth`，
TG3K 使用 `best_MRE_epoch_*.pth`。不会通过 test 结果选择 checkpoint。
脚本拒绝覆盖已有评价目录。

结果路径：

```text
work_dirs/painter3d_seed<1|2|3>/<acdc|hc18|tg3k|chestxray>/
├── train_console.log
├── training_complete.txt
├── best_MRE[_mm]_epoch_*.pth
└── test_evaluation/
    ├── predictions.json
    ├── metrics.json
    ├── inference.log
    └── evaluation.log
```

仅统计 MRE、MSE、RMSE、MAE，无 PCK。
ACDC=1.5625、ChestX-ray=1.4、HC18=0.108 mm/原图像素；
距离指标为 mm，MSE 为 mm²。TG3K 保留 px/px²。
预测文件保留原图像素坐标，方便 TG3K 后续确定 spacing 后换算。

如只需手动启动一个尚未运行的 seed，可将最后的 `1 2 3` 改为对应数字，
例如 `bash datasets_analysis/commands/painter3d/train_seeds.sh acdc 0 2`。
这不是中断续训命令，已有目录仍会被保护。
