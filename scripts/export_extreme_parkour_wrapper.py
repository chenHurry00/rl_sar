# Usage:
# python3 scripts/export_extreme_parkour_wrapper.py --base "policy/go2/robot_lab/base_jit.pt" --vision "policy/go2/robot_lab/vision_weight.pt" --output "policy/go2/robot_lab/extreme_parkour_wrapper.pt"
import argparse
import sys
from typing import Optional
import torch
import torch.nn as nn

RSL_RL_ROOT = "/home/yuchen/usetest/RL/Extreme-Parkour-Onboard/rsl_rl"
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

from rsl_rl.modules import StateHistoryEncoder, DepthOnlyFCBackbone58x87


class RecurrentDepthBackboneTS(nn.Module):
    def __init__(self, base_backbone) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + 53, 128),
            activation,
            nn.Linear(128, 32),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32 + 2),
            last_activation,
        )
        self.hidden_states = torch.jit.Attribute(None, Optional[torch.Tensor])

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        if self.hidden_states is None:
            depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :])
        else:
            depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        return depth_latent

    @torch.jit.export
    def reset_hidden(self):
        self.hidden_states = None


class ExtremeParkourWrapper(nn.Module):
    def __init__(self, base_model, depth_encoder, history_encoder, n_proprio=53, n_hist=10):
        super().__init__()
        self.base_model = base_model
        self.depth_encoder = depth_encoder
        self.history_encoder = history_encoder
        self.n_proprio = n_proprio
        self.n_hist = n_hist

        self.estimator = base_model.estimator.estimator
        self.actor_backbone = base_model.actor.actor_backbone

    def forward(self, proprio_and_hist, depth_flat):
        # proprio_and_hist: [B, n_proprio + n_hist*n_proprio]
        # depth_flat: [B, 58*87]
        if proprio_and_hist.dim() != 2:
            raise RuntimeError("proprio_and_hist must be 2D")
        if depth_flat.dim() != 2:
            raise RuntimeError("depth_flat must be 2D")

        batch = proprio_and_hist.size(0)
        if depth_flat.size(1) != 58 * 87:
            raise RuntimeError("depth_flat size mismatch, expect 58*87")

        proprio = proprio_and_hist[:, :self.n_proprio]
        hist = proprio_and_hist[:, self.n_proprio:]
        if hist.numel() == 0:
            raise RuntimeError("history buffer is empty")

        depth_img = depth_flat.view(batch, 58, 87)
        depth_latent_yaw = self.depth_encoder(depth_img, proprio)

        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5

        proprio = proprio.clone()
        #proprio[:, 6:8] = yaw

        lin_vel_latent = self.estimator(proprio)
        priv_latent = self.history_encoder(hist.view(batch, self.n_hist, self.n_proprio))

        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        actions = self.actor_backbone(obs)
        return actions

    @torch.jit.export
    def reset_hidden(self):
        self.depth_encoder.reset_hidden()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path to base_jit.pt")
    parser.add_argument("--vision", required=True, help="Path to vision_weight.pt")
    parser.add_argument("--output", required=True, help="Output wrapper .pt path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device

    base_model = torch.jit.load(args.base, map_location=device)
    base_model.eval()

    vision_state = torch.load(args.vision, map_location=device)

    n_proprio = 53
    n_hist = 10

    history_state = base_model.actor.history_encoder.state_dict()
    history_out_dim = history_state["linear_output.0.weight"].shape[0]
    history_encoder = StateHistoryEncoder(nn.ELU(), n_proprio, n_hist, history_out_dim)
    history_encoder.load_state_dict(history_state)
    history_encoder.to(device)
    history_encoder.eval()

    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackboneTS(depth_backbone).to(device)
    depth_encoder.load_state_dict(vision_state["depth_encoder_state_dict"])
    depth_encoder.eval()

    wrapper = ExtremeParkourWrapper(base_model, depth_encoder, history_encoder, n_proprio=n_proprio, n_hist=n_hist)
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    scripted.save(args.output)


if __name__ == "__main__":
    main()

'''
"/home/yuchen/miniconda3/envs/extreme38/bin/python" "scripts/export_extreme_parkour_wrapper.py" --base \
"policy/go2/robot_lab/base_jit.pt" --vision "policy/go2/robot_lab/vision_weight.pt" --output "scripts/export_extreme_parkour_wrapper_yawrate.pt"

"/home/yuchen/miniconda3/envs/extreme38/bin/python" "scripts/export_extreme_parkour_wrapper.py" --base \
"/home/yuchen/usetest/rl_sar/policy/go2/robot_lab/extreme-21500-base_jit.pt" --vision "/home/yuchen/usetest/rl_sar/policy/go2/robot_lab/yy3-go2-WHATEVER-21500-vision_weight.pt" --output "scripts/export_21500_extreme_parkour_wrapper_yawrate.pt"
'''
