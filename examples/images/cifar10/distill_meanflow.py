import os
import torch
import torch.nn as nn
from torch import optim
# from torchvision import transforms, datasets
# from torchvision.utils import save_image
from tqdm import tqdm
from cleanfid import fid
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
import math
# from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.models import inception_v3, Inception_V3_Weights

from torch.cuda import OutOfMemoryError
import time
import gc

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
    
class StudentMeanFlow(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        # a lightweight conv encoder-decoder with time conditioning
        self.time_emb_dim = time_emb_dim
        self.time_emb_net = TimeEmbedding(time_emb_dim)

        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
        )
        # process time embeddings into spatial maps
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim * 2, base_ch),  # we will concat t1 and t2 embeddings
            nn.ReLU()
        )

        # decoder / residual convs
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
        )

    def forward(self, x_t1, t1, t2):
        """
        x_t1: [B, C, H, W]
        t1, t2: [B] scalars in [0,1]
        returns v_bar_pred: [B, C, H, W]
        """
        B = x_t1.shape[0]
        feat = self.enc(x_t1)  # [B, base_ch, H, W]
        te1 = self.time_emb_net(t1)  # [B, time_emb_dim]
        te2 = self.time_emb_net(t2)
        te = torch.cat([te1, te2], dim=1)  # [B, 2*time_emb_dim]
        tp = self.time_proj(te)  # [B, base_ch]
        # expand to spatial map
        tp_map = tp.unsqueeze(-1).unsqueeze(-1)  # [B, base_ch, 1,1]
        feat = feat + tp_map  # broadcast add
        out = self.dec(feat)
        # output is v_bar prediction (no activation)
        return out


def load_teacher_and_node(ckpt_path, device, integration_method):
    # 你要在 conditional-flow-matching 代码库里写这个接口，加载模型权重、初始化采样函数
    # 下面是伪代码 /骨架
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



def gen_1_img(student, batch, img_size, device):
    with torch.no_grad():
        z = torch.randn(batch, 3, img_size, img_size, device=device)
        t1 = torch.zeros(batch, device=device)
        t2 = torch.ones(batch, device=device)
        x_pred = student(z, t1, t2)
        imgs = torch.clamp((x_pred + 1) / 2, 0, 1)
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

# ---------------------------
# Evaluation function: generate samples from student and compute FID using CleanFID
# ---------------------------
def evaluate_student(student, device, args):
    """
    Evaluate distilled MeanFlow model using CleanFID.
    假设student生成的样本范围为[-1,1]
    """
    student.eval()

    # 3️⃣ 使用 CleanFID 计算 FID
    print("[Eval] Calculating CleanFID score...")
    fid_score = fid.compute_fid(
        gen=lambda n: gen_1_img(student, args.batch_size_fid, args.img_size, device),
        dataset_name="cifar10",
        batch_size=args.batch_size_fid,
        dataset_res=32,
        num_gen=args.num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
    )

    print(f"[Eval] Student MeanFlow FID: {fid_score:.4f}")
    student.train()



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


# def train_one_step(teacher_model, node, student, device, args):
#     # sample initial noise x(t=0?) NOTE: in gen_1_img they used x = randn(..., device), then integrated to get image at final time. 
#     # Their convention: they start from noise at t=0 and integrate to t=1 (or vice versa). 
#     # We will mirror gen_1_img: start x0 as noise at t=0 and traj[-1] is final image at t=1. # But earlier we used t in [0,1] with x(1)=image; choose convention consistent with teacher.
#     x_noise = torch.randn(args.batch, 3, args.img_size, args.img_size, device=device)
#     # traj: [T, B, C, H, W] 
#     # compute trajectory
#     traj, t_span = sample_teacher_trajectory(teacher_model, node, device, x_noise, 
#                                              args.integration_method, args.integration_steps, 
#                                              y=None, tol=args.tol)
    
#     # now randomly select pairs of (idx1, idx2) for supervision, ensure idx2 > idx1
#     T = traj.shape[0]
#     # choose random indices: we want t1 < t2 ideally (or t1 > t2 sign handled)
#     # We'll pick idx1 < idx2 so denom positive (t2 - t1 > 0)
#     idx1 = torch.randint(0, T - 1, (1,)).item()
#     idx2 = torch.randint(idx1 + 1, T, (1,)).item()
#     vbar_true, x1, x2, t1_val, t2_val = compute_vbar_from_traj(traj, t_span, idx1, idx2)

#     # x1 corresponds to traj[idx1] (state at t1), this is the student input
#     # vbar_true = (x2 - x1)/(t2 - t1)
#     # Prepare input tensors
#     x1 = x1.to(device)
#     # prepare time tensors of shape [B]
#     B = x1.shape[0]
#     t1_tensor = torch.full((B,), float(t1_val), device=device)
#     t2_tensor = torch.full((B,), float(t2_val), device=device)
#     # student forward
#     vbar_pred = student(x1, t1_tensor, t2_tensor) 
#     # loss: regression MSE on vbar 
#     loss = nn.functional.mse_loss(vbar_pred, vbar_true.to(device)) 
#     # optional: add small pixel loss on reconstructed images to stabilise 
#     if args.recon_loss_weight > 0: 
#         # reconstruct x2_pred
#         x2_pred = x1 + (t2_val - t1_val) * vbar_pred 
#         loss_recon = nn.functional.mse_loss(x2_pred, x2.to(device)) 
#         loss = loss + args.recon_loss_weight * loss_recon 
#     return loss, idx1, idx2


# def sample_teacher_trajectory(teacher_model, node, device, x0, integration_method, integration_steps, y=None, tol=1e-5): 
#     """ 
#     Produce trajectory for batch x0 (initial noise) over t_span [0,1] 
#     - If node is provided and has .trajectory(x, t_span, y) use it (preferred for efficiency) 
#     - Else use odeint with teacher_model as RHS (teacher_model(x, t, y) -> dx/dt) 
#     Return: traj: [T, B, C, H, W] t_span: [T] tensor 
#     """ 
#     with torch.no_grad(): 
#         # define t_span 
#         if integration_method == "euler": 
#             t_span = torch.linspace(0.0, 1.0, steps=integration_steps + 1, device=device) 
#             if node is not None and hasattr(node, "trajectory"): 
#                 traj = node.trajectory(x0, t_span=t_span) 
#                 # hopefully returns [T, B, C, H, W] 
#             else: 
#                 # fallback: simple Euler integration using teacher_model 
#                 x = x0 
#                 traj_list = [x0] 
#                 dt = 1.0 / integration_steps 
#                 for k in range(integration_steps): 
#                     t = torch.full((x0.shape[0],), k * dt, device=device) 
#                     # teacher_model expects (x, timesteps, y) 
#                     v = teacher_model(t, x, y) # instantaneous velocity 
#                     x = x + dt * v 
#                     traj_list.append(x) 
#                     traj = torch.stack(traj_list, dim=0) 
#         else: # use odeint for coarse integration (2 points as gen_1_img did) 
#             t_span = torch.tensor([0.0, 1.0], device=device) 
#             # odeint expects func(t, x) 
#             def rhs(t, x_flat): 
#                 # x_flat: [B, C, H, W] flattened to [B, ...] by odeint, but it works with same shape 
#                 # build t_batch 
#                 t_b = torch.full((x0.shape[0],), float(t.cpu().numpy()), device=device) 
#                 return teacher_model(t_b, x_flat, y) 
#             traj = odeint(rhs, x0, t_span, rtol=tol, atol=tol, method=integration_method) 
#     return traj, t_span

def save_checkpoint(student, name):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(args.out_dir, name))
    print(f"✅ Saved checkpoint: {name}")

# ---------------------------
# Training loop: online distillation using teacher trajectories
# ---------------------------
def train_distillation(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # load teacher and node wrapper from conditional-flow-matching repo
    ckpt_path = f"{args.input_dir}/{args.model}/{args.model}_cifar10_weights_step_{args.step}.pt"

    teacher_model, node = load_teacher_and_node(ckpt_path, device, args.integration_method)

    # init student
    student = StudentMeanFlow(in_channels=3, base_ch=args.base_ch, time_emb_dim=args.time_emb).to(device)
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

        # end epoch eval (partial)
        if (epoch + 1) % args.eval_every == 0:
            print(f"Epoch {epoch}: running eval...")
            avg_pixel, avg_lpips, avg_feat = fast_eval_student(student, teacher_model, 
                                                               node, lpips_metric, 
                                                               feat_net, device, args)
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
    evaluate_student(student, device, args)

    return student



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cfm")
    parser.add_argument("--step", type=int, default=400000)
    parser.add_argument("--input_dir", type=str, default="/workspace/SHLi/conditional-flow-matching/examples/images/cifar10/ckpt")
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--integration_method", type=str, default="euler")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--time_emb", type=int, default=128)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
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
