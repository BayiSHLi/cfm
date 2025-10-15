import os
import torch
import torch.nn as nn
from torch import optim
# from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from cleanfid import fid
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
import math
# from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.models import inception_v3, Inception_V3_Weights

import swanlab

# 你可能需要在这个库里找到 teacher model class、采样 / trajectory 方法、加载 checkpoint 的方法


# ------- student 模型 skeleton（MeanFlow 风格） -------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        # t is [B] tensor in [0,1], map to sin/cos embedding
        # use gaussian fourier features
        B = t.shape[0]
        feat_dim = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), feat_dim, device=t.device))
        # ravel: t[:,None] * freqs[None,:]
        args = t.unsqueeze(1) * freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, dim]
        return self.net(emb)
    
class StudentMeanFlow(UNetModelWrapper):
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__(        
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=128, #base channel of UNet
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
            )
        # process time embeddings into spatial maps
        self.time_emb_dim = time_emb_dim
        self.time_emb_net = TimeEmbedding(time_emb_dim)

        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim * 2, base_ch),  # we will concat t1 and t2 embeddings
            nn.ReLU()
        )

    def forward(self, x_t1, t1, t2, y=None):
        """
        x_t1: [B, C, H, W]
        t1, t2: [B] scalars in [0,1]
        returns v_bar_pred: [B, C, H, W]
        """
        B = x_t1.shape[0]
        te1 = self.time_emb_net(t1)  # [B, time_emb_dim]
        te2 = self.time_emb_net(t2)
        te = torch.cat([te1, te2], dim=1)  # [B, 2*time_emb_dim]
        tp = self.time_proj(te)  # [B, base_ch]

        return super().forward(tp, x_t1, y=y)


def load_teacher_and_node(ckpt_path, device, integration_method):
    teacher = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128, #base channel of UNet
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    # Load the model
    print("path: ", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["ema_model"]
    try:
        teacher.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        teacher.load_state_dict(new_state_dict)
    teacher.eval()

    node = NeuralODE(teacher, solver=integration_method)

    return teacher, node


# ---------------------------
# Utility: compute average velocity from teacher trajectory
# Given trajectory tensor traj [T, B, C, H, W] and indices idx1, idx2 (with t_idx values)
# compute v_bar = (x(t2) - x(t1)) / (t2 - t1)
# t values correspond to t_span indices
# ---------------------------
def compute_vbar_from_traj(traj, t_span, idx1, idx2):
    # traj: [T, B, C, H, W]
    x1 = traj[idx1]  # [B,C,H,W]
    x2 = traj[idx2]
    t1 = float(t_span[idx1].cpu().item())
    t2 = float(t_span[idx2].cpu().item())
    denom = (t2 - t1)
    if abs(denom) < 1e-12:
        raise ValueError("t2 and t1 are equal")
    vbar = (x2 - x1) / denom
    return vbar, x1, x2, t1, t2



def gen_1_img(student, batch, img_size, device, sample_steps=2):
    with torch.no_grad():
        # 起始噪声 z at t=1
        z = torch.randn(batch, 3, img_size, img_size, device=device)
        # 构造时间格点，从 1 -> 0
        t = torch.linspace(0.0, 1.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t1 = torch.full((batch,), t[i], device=device)
            t2 = torch.full((batch,), t[i+1], device=device)
            vbar = student(z, t1, t2)  # 预测 average velocity
            # 乘时间差
            delta = (t2 - t1)  # 这是标量，通常负数
            # delta 相对于每样本可以是向量，但这里常量
            # reshape delta to broadcast to (batch,1,1,1)
            delta_map = delta.view(-1, *([1] * (z.dim()-1)))
            z = z + delta_map * vbar  # 更新
        # 生成图像
        imgs = torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)
    return imgs


def fast_eval_student(student, teacher_model, node, lpips_metric,
                      feat_net, device, args):
    """
    使用 torchmetrics 的 LPIPS 作为感知距离，快速评估 student 的进展。

    student: MeanFlow 风格模型，接口 student(x_t1, t1, t2) → v_bar  
    teacher_model, node: 用于生成 teacher trajectory  
    device: torch.device  
    args: 包含 fast_eval_samples, fast_eval_batch, integration_steps, method, tol, etc.
    """
    student.eval()
    teacher_model.eval()

    total_pixel = 0.0
    total_lpips = 0.0
    total_feat = 0.0
    count = 0

    with torch.no_grad():
        pbar = tqdm(range(0, args.fast_eval_samples, args.fast_eval_batch), desc="FastEval")
        for i in pbar:
            b = min(args.fast_eval_batch, args.fast_eval_samples - i)
            # sample initial noise (or x1) consistent with your distillation setup
            x0 = torch.randn(b, 3, args.img_size, args.img_size, device=device)

            # teacher trajectory
            traj, t_span = sample_teacher_trajectory(
                teacher_model, node, device,
                x0, args.integration_method,
                args.integration_steps,
                y=None, tol=args.tol
            )
            # we choose t1 = idx1, t2 = idx2 (e.g. endpoints)
            T = traj.shape[0]
            idx1 = 0
            idx2 = T - 1
            x1 = traj[idx1]  # state at t1
            x2 = traj[idx2].view([-1, 3, 32, 32]).clip(-1, 1)

            t1_val = float(t_span[idx1].cpu().item())
            t2_val = float(t_span[idx2].cpu().item())

            # student predict average velocity
            vbar_pred = student(
                x1,
                torch.full((b,), t1_val, device=device),
                torch.full((b,), t2_val, device=device)
            )
            # reconstruct x2_pred
            x2_pred = x1 + (t2_val - t1_val) * vbar_pred
            x2_pred = torch.clamp(x2_pred, -1.0, 1.0)

            # —— pixel-level L2 / MSE loss —— #
            loss_pixel = nn.functional.mse_loss(x2_pred, x2.to(device), reduction='mean')
            total_pixel += loss_pixel.item() * b

            # —— LPIPS 感知距离 —— #
            # LPIPS 期望输入在 [-1,1] 区间，并返回 scalar
            lpips_val = lpips_metric(x2_pred, x2.to(device))
            total_lpips += lpips_val.item() * b

            # —— feature-level MSE (用 Inception 特征) —— #
            # 需要把 x2_pred 和 x2 调整到 Inception 输入尺寸（299×299）和范围
            x2p_resized = nn.functional.interpolate(x2_pred, size=(299, 299), mode='bilinear', align_corners=False)
            x2_resized = nn.functional.interpolate(x2.to(device), size=(299, 299), mode='bilinear', align_corners=False)
            feat_pred = feat_net(x2p_resized)
            feat_true = feat_net(x2_resized)
            loss_feat = nn.functional.mse_loss(feat_pred, feat_true, reduction='mean')
            total_feat += loss_feat.item() * b

            count += b

    avg_pixel = total_pixel / count
    avg_lpips = total_lpips / count
    avg_feat = total_feat / count

    print(f"[FastEval] LPIPS: {avg_lpips:.6f}, FeatMSE: {avg_feat:.6f}, PixelMSE: {avg_pixel:.6f}.")

    student.train()
    return avg_pixel, avg_lpips, avg_feat


def visualize_student_outputs(student, device, args, prefix: str = "eval"):
    """
    从 student 中采样一小批图像并保存，用于快速视觉检查。
    prefix 用于文件命名，如 "eval_epoch5"
    """
    student.eval()
    with torch.no_grad():
        # 采样少量（如 16 或 25 张图）用于可视化
        n_vis = min(args.visualize_num, args.eval_batch)
        x1 = torch.randn(n_vis, 3, args.img_size, args.img_size, device=device)
        t1 = torch.ones(n_vis, device=device) * args.t1_vis  # e.g. t1 = 1.0
        t2 = torch.zeros(n_vis, device=device)  # e.g. t2 = 0.0
        vbar = student(x1, t1, t2)
        x0_pred = x1 + (t2 - t1) * vbar
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # 转换到 [0,1]
        imgs = (x0_pred + 1.0) / 2.0
        imgs = torch.clamp(imgs, 0.0, 1.0)

        save_dir = os.path.join(args.out_dir, "vis")
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{prefix}_samples.png")
        save_image(imgs, fname, nrow=int(math.sqrt(n_vis)))
        print(f"Visualized student outputs to {fname}")

    student.train()

# ---------------------------
# Evaluation function: generate samples from student and compute FID using CleanFID
# ---------------------------
def evaluate_student(student, device, args, sample_steps=2):
    """
    Evaluate distilled MeanFlow model using CleanFID.
    假设student生成的样本范围为[-1,1]
    """
    student.eval()

    # 3️⃣ 使用 CleanFID 计算 FID
    print("[Eval] Calculating CleanFID score...")
    fid_score = fid.compute_fid(
        gen=lambda n: gen_1_img(student, args.batch_size_fid, args.img_size, device, sample_steps),
        dataset_name="cifar10",
        batch_size=args.batch_size_fid,
        dataset_res=32,
        num_gen=args.num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
    )

    print(f"[Eval] Student MeanFlow FID: {fid_score:.4f}")
    student.train()

    visualize_student_outputs(student, device, args, prefix="eval_final")



def safe_cuda_cleanup(fast=True):
    """轻量级 GPU 清理：仅在异常时调用"""
    if fast:
        torch.cuda.empty_cache()
    else:
        import gc, time
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)


def train_one_step(teacher_model, node, student, device, args, retry_on_fail=True):
    """单步训练，带轻量级 GPU 异常恢复"""
    try:
        # 1️⃣ 初始化输入噪声
        x_noise = torch.randn(args.batch, 3, args.img_size, args.img_size, device=device)

        # 2️⃣ 生成教师轨迹
        traj, t_span = sample_teacher_trajectory(
            teacher_model, node, device,
            x_noise, args.integration_method,
            args.integration_steps, y=None, tol=args.tol
        )

        # 3️⃣ 随机采样两帧
        T = traj.shape[0]
        idx1 = torch.randint(0, T - 1, (1,)).item()
        idx2 = torch.randint(idx1 + 1, T, (1,)).item()

        vbar_true, x1, x2, t1_val, t2_val = compute_vbar_from_traj(traj, t_span, idx1, idx2)

        # 4️⃣ 准备输入
        B = x1.shape[0]
        x1 = x1.to(device)
        t1_tensor = torch.full((B,), float(t1_val), device=device)
        t2_tensor = torch.full((B,), float(t2_val), device=device)

        # 5️⃣ 学生预测与损失
        vbar_pred = student(x1, t1_tensor, t2_tensor)
        loss = nn.functional.mse_loss(vbar_pred, vbar_true.to(device))

        # 可选重构损失
        if args.recon_loss_weight > 0:
            x2_pred = x1 + (t2_val - t1_val) * vbar_pred
            loss_recon = nn.functional.mse_loss(x2_pred, x2.to(device))
            loss = loss + args.recon_loss_weight * loss_recon

        return loss, idx1, idx2

    except RuntimeError as e:
        err_msg = str(e)
        if any(keyword in err_msg for keyword in ["CUDA", "ECC", "device-side", "out of memory"]):
            print(f"⚠️ GPU error detected: {err_msg}")
            safe_cuda_cleanup(fast=True)
            if retry_on_fail:
                print("🔁 Retrying one more time...")
                return train_one_step(teacher_model, node, student, device, args, retry_on_fail=False)
        raise e


def sample_teacher_trajectory(teacher_model, node, device, x0, integration_method,
                              integration_steps, y=None, tol=1e-5, retry_on_fail=True):
    """生成教师轨迹（高性能版，异常时快速恢复）"""
    try:
        with torch.no_grad():
            if integration_method == "euler":
                t_span = torch.linspace(0.0, 1.0, steps=integration_steps + 1, device=device)
                if node is not None and hasattr(node, "trajectory"):
                    traj = node.trajectory(x0, t_span=t_span)
                else:
                    traj_list = [x0]
                    dt = 1.0 / integration_steps
                    x = x0
                    for k in range(integration_steps):
                        t = torch.full((x0.shape[0],), k * dt, device=device)
                        v = teacher_model(t, x, y)
                        x = x + dt * v
                        traj_list.append(x)
                    traj = torch.stack(traj_list, dim=0)
            else:
                t_span = torch.tensor([0.0, 1.0], device=device)
                def rhs(t, x_flat):
                    t_b = torch.full((x0.shape[0],), float(t.cpu().numpy()), device=device)
                    return teacher_model(t_b, x_flat, y)
                traj = odeint(rhs, x0, t_span, rtol=tol, atol=tol, method=integration_method)
        return traj, t_span

    except RuntimeError as e:
        err_msg = str(e)
        if any(keyword in err_msg for keyword in ["CUDA", "ECC", "device-side"]):
            print(f"⚠️ GPU error in sample_teacher_trajectory: {err_msg}")
            safe_cuda_cleanup(fast=True)
            if retry_on_fail:
                print("🔁 Retrying trajectory computation...")
                return sample_teacher_trajectory(
                    teacher_model, node, device, x0, integration_method,
                    integration_steps, y, tol, retry_on_fail=False
                )
        raise e


def save_checkpoint(student, name):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(args.out_dir, name))
    print(f"✅ Saved checkpoint: {name}")

# ---------------------------
# Training loop: online distillation using teacher trajectories
# ---------------------------
def train_distillation(args):
    run = swanlab.init(project="distill_meanflow", name=args.exp_name, config=vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # load teacher and node wrapper from conditional-flow-matching repo
    ckpt_path = f"{args.input_dir}/{args.model}/{args.model}_cifar10_weights_step_{args.step}.pt"

    teacher_model, node = load_teacher_and_node(ckpt_path, device, args.integration_method)

    # init student
    student = StudentMeanFlow(in_channels=3, base_ch=args.base_ch, time_emb_dim=args.time_emb).to(device)
    # resume from checkpoint if provided
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        print(f"Resuming student from {args.resume_ckpt}...")
        student.load_state_dict(torch.load(args.resume_ckpt, map_location=device))
    
    optim_stu = optim.Adam(student.parameters(), lr=args.lr)

    # eval LPIPS metric
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', reduction='mean', normalize=False
    ).to(device)

    # 准备 feature net（例如 InceptionV3，用于 feature-level差异）
    feat_net = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
    feat_net.eval()
    for p in feat_net.parameters():
        p.requires_grad = False

    best_pixel = float('inf')
    best_lpips = float('inf')
    best_feat = float('inf')

    # training: for each batch sample initial noise x0 and compute full trajectory via teacher
    for epoch in range(args.epochs):
        if args.eval_only:
            print("Eval only mode, skipping training...")
            avg_pixel, avg_lpips, avg_feat = fast_eval_student(student, teacher_model, 
                                                               node, lpips_metric, 
                                                               feat_net, device, args)
            return student
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        for i in pbar:
            loss, idx1, idx2 = train_one_step(teacher_model, node, student, device, args)

            optim_stu.zero_grad()
            loss.backward()
            optim_stu.step()

            pbar.set_description(f"ep{epoch} step{i} loss:{loss.item():.6f} idx1:{idx1} idx2:{idx2}")
            swanlab.log({"train/loss": loss.item(), "epoch": epoch})

        # end epoch eval (partial)
        if (epoch + 1) % args.eval_every == 0:
            print(f"Epoch {epoch}: running eval...")
            avg_pixel, avg_lpips, avg_feat = fast_eval_student(student, teacher_model, 
                                                               node, lpips_metric, 
                                                               feat_net, device, args)
            swanlab.log({
                "eval/avg_pixel_mse": avg_pixel,
                "eval/avg_lpips": avg_lpips,
                "eval/avg_feat_mse": avg_feat,
                "epoch": epoch
            })
            # 主指标：LPIPS（越小越好）
            if avg_lpips < best_lpips:
                best_lpips = avg_lpips
                best_pixel = avg_pixel  # 同步更新辅助指标记录
                best_feat = avg_feat
                save_checkpoint(student, "student_meanflow_best.pth")

            # 若LPIPS变化不大，则检查辅助指标
            elif abs(avg_lpips - best_lpips) < 0.01:
                # 优先比较Feature MSE
                if avg_feat < best_feat - 0.001:
                    best_feat = avg_feat
                    save_checkpoint(student, "student_meanflow_best.pth")
                # 再比较Pixel MSE
                elif avg_pixel < best_pixel - 0.001:
                    best_pixel = avg_pixel
                    save_checkpoint(student, "student_meanflow_best.pth")

    # save final student
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(args.out_dir, "student_meanflow_last.pth"))
    
    # test on best model
    print("Evaluating best model on FID...")
    student.load_state_dict(torch.load(os.path.join(args.out_dir, "student_meanflow_best.pth")))
    evaluate_student(student, device, args, sample_steps=2)

    return student



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="unet_student")
    parser.add_argument("--model", type=str, default="cfm")
    parser.add_argument("--step", type=int, default=400000)
    parser.add_argument("--input_dir", type=str, default="/workspace/SHLi/conditional-flow-matching/examples/images/cifar10/ckpt")
    parser.add_argument("--resume_ckpt", type=str, default="/workspace/SHLi/conditional-flow-matching/examples/images/cifar10/distill_outputs_1014_2/student_meanflow_best.pth", help="path to student checkpoint to resume")
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--integration_method", type=str, default="euler")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--time_emb", type=int, default=128)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--batch_size_fid", type=int, default=5000)
    parser.add_argument("--fast_eval_samples", type=int, default=1000)
    parser.add_argument("--fast_eval_batch", type=int, default=100)
    parser.add_argument("--num_gen", type=int, default=50000)
    parser.add_argument("--out_dir", type=str, default="./distill_outputs")
    parser.add_argument("--real_dir", type=str, default="./cifar10_real_png")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recon_loss_weight", type=float, default=0.0, help="weight for optional reconstruction loss")
    parser.add_argument("--eval_only", default=False, help="only run eval using existing student model")
    args = parser.parse_args()

    train_distillation(args)
