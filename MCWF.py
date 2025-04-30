import torch
import torch.nn as nn

class Multi_Channel_Wiener(nn.Module):
    def __init__(self, alpha_xx, alpha_xy):
        super(Multi_Channel_Wiener, self).__init__()
        self.alpha_xx = alpha_xx
        self.alpha_xy = alpha_xy

    def inverse(self, mat, vec):
        # mat: (B, F, C, C)
        # vec: (B, F, C)
        lamda = self.alpha_xx / (1 - self.alpha_xx)  # scalar
        vec_conj = torch.conj(vec)  # (B, F, C)
        outer_prod = torch.matmul(vec.unsqueeze(-1) , vec_conj.unsqueeze(-2))  # (B, F, C, C)

        numerator = torch.matmul(torch.matmul(mat, outer_prod), mat)  # (B, F, C, C)

        quadratic_term = torch.matmul(vec_conj.unsqueeze(2), torch.matmul(mat, vec.unsqueeze(-1)))  # (B, 1, 1)

        denominator = 1 + lamda * quadratic_term  # (B, F, 1, 1)
      

        mat_inv = mat - lamda * numerator / denominator  # (B, F, C, C)
        return mat_inv

    def wiener_filter(self, enhanced, mic):
        # enhanced: (B, F, T)
        # mic: (B, F, T, C)
        device = enhanced.device
        dtype = enhanced.dtype
        eps = 1e2
        B, F, T, C = mic.shape

        weight = torch.zeros(B, F, T, C, device=device, dtype=dtype)

        eps_mat = eps * torch.eye(C, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, C, C)

        # Batch frequency
        medium = eps_mat.repeat(B, F, 1, 1)  # (B, F, C, C)
        corr_vec_est = torch.zeros(B, F, C, 1, device=device, dtype=dtype) + 1e-2  # (B, F, C, 1)

        for j in range(T):
            mic_vec = mic[:, :, j, :]  # (B, F, C)
            enhanced_frame = enhanced[:, :, j]  # (B, F)

            corr_mat_est_inv = self.inverse(medium, mic_vec)  # (B, F, C, C)

            corr_mat_est_inv = corr_mat_est_inv / (1 - self.alpha_xx)
            medium = corr_mat_est_inv

            corr_vec = mic_vec * torch.conj(enhanced_frame).unsqueeze(-1)  # (B, F, C)
            corr_vec_est = (1 - self.alpha_xy) * corr_vec_est + self.alpha_xy * corr_vec.unsqueeze(-1)  # (B, F, C, 1)

            weight[:, :, j, :] = torch.matmul(corr_mat_est_inv, corr_vec_est).squeeze(-1)  # (B, F, C)

        return weight  # (B, F, T, C)

    def forward(self, spec_enhanced, spec_mic):
        # spec_enhanced: (B, F, T)  complex
        # spec_mic: (B, F, T, C)  complex
        spec_enhanced_split = torch.chunk(spec_enhanced, chunks=4, dim=-1)
        spec_mic_split = torch.chunk(spec_mic, chunks=4, dim=2)

        wiener_output = []
        for i in range(4):
            wiener_weight = self.wiener_filter(spec_enhanced_split[i], spec_mic_split[i])  # (B, F, T, C)
            wiener_weight_conj = torch.conj(wiener_weight).unsqueeze(3)  # (B, F, T, 1, C)
            wiener_output.append(torch.matmul(wiener_weight_conj, spec_mic_split[i].unsqueeze(-1)))  # (B, F, T, 1, 1)

        wiener_output = torch.cat(wiener_output, dim=2)  # (B, F, T, 1, 1)

        return wiener_output.squeeze(-1).squeeze(-1)  # (B, F, T)