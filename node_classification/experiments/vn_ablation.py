"""Paired SF ablation: class virtual nodes versus train-node prototypes."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import torch

from node_classification.data import load_planetoid_graph
from node_classification.models.forward_gnn import ForwardGNN, train_forwardgnn

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--steps-per-layer", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/Planetoid")
    parser.add_argument("--output", type=Path, default=ROOT / "results/own_experiments/sf_vn_ablation_mps")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; refusing to fall back to CPU")
    args.output.mkdir(parents=True, exist_ok=True)
    all_records = []

    for dataset_name in args.datasets:
        data = load_planetoid_graph(dataset_name, args.data_root).to(device)
        for depth in args.layers:
            for seed in args.seeds:
                paired_initial = None
                for with_vn in (True, False):
                    variant = "sf_vn" if with_vn else "sf_prototype_no_vn"
                    output_file = args.output / f"{dataset_name}_L{depth}_seed{seed}_{variant}.json"
                    if output_file.exists():
                        record = json.loads(output_file.read_text())
                        all_records.append(record)
                        print(f"skip existing {output_file.name}", flush=True)
                        continue

                    torch.manual_seed(seed)
                    if device.type == "mps":
                        torch.mps.manual_seed(seed)
                    model = ForwardGNN(
                        data.x.size(1), args.hidden, data.num_classes,
                        num_layers=depth, temperature=args.temperature,
                        use_virtual_nodes=with_vn,
                    ).to(device)
                    layer_parameters = [
                        parameter.detach().cpu().clone()
                        for layer in model.layers for parameter in layer.parameters()
                    ]
                    if paired_initial is None:
                        paired_initial = layer_parameters
                    elif any(not torch.equal(left, right) for left, right in zip(paired_initial, layer_parameters)):
                        raise RuntimeError("paired VN/no-VN models did not receive identical GNN weights")

                    validation_history = []

                    def record(step, loss, current_model):
                        if step % args.eval_every == 0 or step == args.steps_per_layer * depth:
                            current_model.eval()
                            with torch.no_grad():
                                predictions = current_model(data.x, data.edge_index).argmax(dim=1)
                                score = (predictions[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                            validation_history.append({"step": step, "local_loss": loss,
                                                       "validation_accuracy": score})

                    if device.type == "mps":
                        torch.mps.synchronize()
                    started = time.perf_counter()
                    train_result = train_forwardgnn(
                        model, data, epochs=args.steps_per_layer,
                        learning_rate=args.learning_rate, weight_decay=5e-4,
                        patience=-1, verbose=False, callback=record,
                    )
                    if device.type == "mps":
                        torch.mps.synchronize()
                    elapsed = time.perf_counter() - started
                    record_data = {
                        "dataset": dataset_name,
                        "split": "Planetoid public fixed split",
                        "seed": seed,
                        "model": "GCN-SF",
                        "variant": variant,
                        "uses_virtual_nodes": with_vn,
                        "no_vn_readout": "leave-one-out local training prototypes; train-node class means at inference" if not with_vn else None,
                        "device": str(device),
                        "layers": depth,
                        "hidden": args.hidden,
                        "steps_per_layer": args.steps_per_layer,
                        "learning_rate": args.learning_rate,
                        "temperature": args.temperature,
                        "layer_epochs": train_result.layer_epochs,
                        "layer_best_epochs": train_result.layer_best_epochs,
                        "train_accuracy": train_result.train_accuracy,
                        "validation_accuracy": train_result.val_accuracy,
                        "test_accuracy": train_result.test_accuracy,
                        "wall_seconds": elapsed,
                        "validation_history": validation_history,
                        "test_set_used_for_selection": False,
                    }
                    output_file.write_text(json.dumps(record_data, indent=2) + "\n")
                    all_records.append(record_data)
                    print(f"{dataset_name} L={depth} seed={seed} {variant}: "
                          f"val={train_result.val_accuracy:.4f} test={train_result.test_accuracy:.4f} "
                          f"time={elapsed:.1f}s [{device}]", flush=True)

    summaries = []
    for dataset_name in args.datasets:
        for depth in args.layers:
            for variant in ("sf_vn", "sf_prototype_no_vn"):
                records = [row for row in all_records if row["dataset"] == dataset_name
                           and row["layers"] == depth and row["variant"] == variant]
                values = [row["test_accuracy"] for row in records]
                summaries.append({
                    "dataset": dataset_name, "layers": depth, "variant": variant,
                    "n_initializations": len(values),
                    "test_accuracy_mean": statistics.mean(values) if values else None,
                    "test_accuracy_std_ddof0": statistics.pstdev(values) if values else None,
                    "validation_accuracy_mean": statistics.mean(row["validation_accuracy"] for row in records) if records else None,
                    "wall_seconds_mean": statistics.mean(row["wall_seconds"] for row in records) if records else None,
                })
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    write_report(args, summaries, all_records, device)


def write_report(args, summaries, records, device):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-mps-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = args.datasets
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for variant, label, color, marker in (
            ("sf_vn", "SF with virtual nodes", "#c05c19", "o"),
            ("sf_prototype_no_vn", "SF without VN (train prototypes)", "#3454a5", "s"),
        ):
            rows = [row for row in summaries if row["dataset"] == dataset and row["variant"] == variant]
            axis.errorbar([row["layers"] for row in rows],
                          [100 * row["test_accuracy_mean"] for row in rows],
                          yerr=[100 * row["test_accuracy_std_ddof0"] for row in rows],
                          marker=marker, capsize=4, label=label, color=color)
        axis.set(title=dataset, xlabel="GNN layers", ylabel="Test accuracy (%)")
        axis.set_xticks(args.layers)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"SF virtual-node ablation ({device}, {len(args.seeds)} initializations)")
    fig.tight_layout()
    plot = args.output / "sf_vn_ablation.png"
    fig.savefig(plot, dpi=180)
    plt.close(fig)

    lines = [
        "# SF 有/无虚拟节点对照实验", "",
        f"设备：`{device}`；数据集：{', '.join(args.datasets)}；固定 Planetoid public split；随机初始化种子：{args.seeds}。", "",
        f"GCN 隐藏维度 {args.hidden}，层数 {args.layers}，每层固定 {args.steps_per_layer} 次局部更新；Adam lr={args.learning_rate}、weight_decay=0.0005。", "",
        "VN 组沿用类别虚拟节点及训练标签连边。无 VN 组不扩展图、不增加节点或边；以训练节点隐藏表示的类别均值作为原型，训练时正类原型采用 leave-one-out，避免样本与自身直接匹配。推理只使用训练节点生成原型，不使用验证/测试标签。", "",
        "两组的 GNN 层权重按种子配对初始化，层宽、深度、局部交叉熵、训练预算和验证策略一致。固定预算训练，不按测试集选检查点。无 VN 组的原型读出与 VN 组并非完全相同的参数化，因此应将结果解释为“虚拟节点上下文 vs. 无节点原型上下文”，而非只删除节点后的完全等价算法。", "",
        "测试准确率均值 ± 总体标准差（不同初始化；固定公开划分）：", "",
        "| 数据 | 层数 | 条件 | n | 测试准确率 | 平均训练秒 |", "|---|---:|---|---:|---:|---:|",
    ]
    for row in summaries:
        if row["n_initializations"]:
            label = "SF + VN" if row["variant"] == "sf_vn" else "SF，无 VN（原型）"
            lines.append(f"| {row['dataset']} | {row['layers']} | {label} | {row['n_initializations']} | "
                         f"{100*row['test_accuracy_mean']:.2f} ± {100*row['test_accuracy_std_ddof0']:.2f}% | "
                         f"{row['wall_seconds_mean']:.2f} |")
    lines += ["", "按相同初始化配对的测试准确率差值（无 VN 原型 − SF-VN）：", "",
              "| 数据 | 层数 | 配对数 | 平均差值 ± 标准差（百分点） | 每个初始化方向 |",
              "|---|---:|---:|---:|---|"]
    for dataset in args.datasets:
        for depth in args.layers:
            paired = {}
            for row in records:
                if row["dataset"] == dataset and row["layers"] == depth:
                    paired.setdefault(row["seed"], {})[row["variant"]] = row["test_accuracy"]
            differences = [100 * (values["sf_prototype_no_vn"] - values["sf_vn"])
                           for _, values in sorted(paired.items())
                           if set(values) == {"sf_vn", "sf_prototype_no_vn"}]
            if differences:
                std = statistics.pstdev(differences) if differences else 0.0
                directions = ", ".join(f"{value:+.1f}" for value in differences)
                lines.append(f"| {dataset} | {depth} | {len(differences)} | "
                             f"{statistics.mean(differences):+.2f} ± {std:.2f} | {directions} |")
    lines += ["", "![SF 有/无虚拟节点测试准确率对比](sf_vn_ablation.png)", "",
              "测试集只在每个模型训练结束后用于最终评估；验证集曲线用于检查训练过程但不选择检查点。当前种子只改变初始化，不改变 Planetoid 的固定公开划分，因此结果不能估计数据划分不确定性。", "",
              "逐次原始结果见同目录 JSON；测试指标为描述性统计，不代表普遍结论。", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(f"summary={args.output / 'summary.json'} plot={plot} report={args.output / 'report.md'}")


if __name__ == "__main__":
    main()
