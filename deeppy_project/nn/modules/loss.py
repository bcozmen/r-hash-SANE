import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_Xent(nn.Module):
    def __init__(self, temp):
        super(NT_Xent, self).__init__()
        self.temperature = temp
        self.mask = None

        
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        self.mask = mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # dimension of similarity matrix
        batch_size = z_i.size(0)


        N = 2 * batch_size
        if self.mask is None or self.batch_size != batch_size:
            self.batch_size = batch_size
            self.mask_correlated_samples()

        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).long()
        # the logits are NxN (N+1?) predictions for imaginary classes.
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class NTXentLoss(nn.Module):
    def __init__(self, temp=0.5):
        super(NTXentLoss, self).__init__()
        self.temp = temp

    def forward(self, z_i, z_j):
        """
        z_i: Tensor of shape [N, D] - first view embeddings
        z_j: Tensor of shape [N, D] - second view embeddings
        Returns:
            Scalar NT-Xent loss
        """
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Normalize embeddings
        z = nn.functional.normalize(z, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T)  # [2N, 2N]
        sim_matrix = sim_matrix / self.temp

        # Mask to remove self-similarity
        mask = (~torch.eye(2 * N, 2 * N, dtype=bool, device=z.device)).float()

        # Numerator: positive pairs (i, j)
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temp)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2N]

        # Denominator: sum over all except self
        denom = torch.sum(torch.exp(sim_matrix) * mask, dim=1)  # [2N]

        loss = -torch.log(pos_sim / denom)
        return loss.mean()





class QuaternionLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super(QuaternionLoss, self).__init__()
        assert loss_type in ['mse', 'relative'], "loss_type must be 'mse' or 'relative'"
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss_fn = self.quaternion_mse_loss
        else:
            self.loss_fn = self.quaternion_relative_loss

    def forward(self, q_pred, q1_true, q2_true):
        return self.loss_fn( q_pred, q1_true, q2_true)

    def euler_to_quaternion(self, euler: torch.Tensor, order: str = 'xyz') -> torch.Tensor:
        roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        w = cr * cp * cy - sr * sp * sy
        x = sr * cp * cy + cr * sp * sy
        y = cr * sp * cy - sr * cp * sy
        z = cr * cp * sy + sr * sp * cy

        return torch.stack((w, x, y, z), dim=-1)

    def normalize_quaternion(self, q: torch.Tensor, eps=1e-8) -> torch.Tensor:
        return q / (q.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))

    def quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        w = q[..., 0:1]
        xyz = -q[..., 1:]
        return torch.cat([w, xyz], dim=-1)

    def quaternion_multiply(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = r.unbind(-1)

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack((w, x, y, z), dim=-1)

    def compute_relative_quaternion_matrix(self, quats: torch.Tensor) -> torch.Tensor:
        """
        quats: (N, 4) unit quaternions
        returns (N, N, 4) matrix where [i,j] = quats[i] * quats[j]^{-1}
        """
        N = quats.shape[0]
        # Make sure unit quats
        quats = quats / quats.norm(dim=-1, keepdim=True)

        quats_conj = self.quaternion_conjugate(quats)  # (N, 4)

        # Broadcast:
        q_i = quats[:, None, :]      # (N, 1, 4)
        q_j_conj = quats_conj[None, :, :]  # (1, N, 4)

        # Compute pairwise multiplication
        rel_matrix = self.quaternion_multiply(q_i, q_j_conj)  # (N, N, 4)

        return rel_matrix
    def rotate_points_with_quaternions(self,points: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
        """
        points: (3,), (1,3), or (B,3)
        quats: (N,4) unit quaternions
        
        Returns:
            - If points is (3,) or (1,3): (N,3)
            - If points is (B,3) with B>1: (N,B,3)
        """
        N = quats.shape[0]

        # Normalize quaternions
        quats = self.normalize_quaternion(quats)
        quats_conj = self.quaternion_conjugate(quats)  # (N, 4)

        # Handle input shapes
        if points.ndim == 1:
            # (3,) -> (1,3)
            points = points.unsqueeze(0)
        
        B = points.shape[0]

        # Convert points to pure quaternions (0, x, y, z)
        zeros = torch.zeros((B, 1), device=points.device, dtype=points.dtype)
        p_quats = torch.cat([zeros, points], dim=-1)  # (B, 4)

        if B == 1:
            # single point case -> output (N,3)
            p_quats = p_quats.expand(N, -1)  # (N,4)
            qp = self.quaternion_multiply(quats, p_quats)       # (N,4)
            qpq_inv = self.quaternion_multiply(qp, quats_conj)  # (N,4)
            return qpq_inv[:, 1:]  # (N,3)

        else:
            # multiple points -> apply all quats to all points
            # output (N,B,3)
            p_quats = p_quats[None, :, :].expand(N, B, 4)   # (N,B,4)
            q = quats[:, None, :].expand(N, B, 4)           # (N,B,4)
            q_conj = quats_conj[:, None, :].expand(N, B, 4) # (N,B,4)

            qp = self.quaternion_multiply(q, p_quats)       # (N,B,4)
            qpq_inv = self.quaternion_multiply(qp, q_conj)  # (N,B,4)

            return qpq_inv[..., 1:]  # (N,B,3)
    def quaternion_relative_loss(self, q_pred, q1_true, q2_true):
        q_pred = self.normalize_quaternion(q_pred)
        q1_true = self.normalize_quaternion(q1_true)
        q2_true = self.normalize_quaternion(q2_true)

        
        q1_true_inv = self.quaternion_conjugate(q1_true)


        q_true = self.quaternion_multiply(q2_true, q1_true_inv)

        dot = torch.abs(torch.sum(q_pred * q_true, dim=-1))
        dot = torch.clamp(dot, 0.0, 1.0)

        #loss = (1.0 - dot) ** 2
        
        loss = (2 * torch.acos(dot)) ** 2
        return loss.mean()

    def quaternion_mse_loss(self, q1_pred, q2_pred, q1_true, q2_true):
        q1_pred = self.normalize_quaternion(q1_pred)
        q2_pred = self.normalize_quaternion(q2_pred)
        q1_true = self.normalize_quaternion(q1_true)
        q2_true = self.normalize_quaternion(q2_true)

        dot1 = torch.sum(q1_pred * q1_true, dim=-1)
        dot1 = torch.clamp(dot1, -1.0, 1.0)

        dot2 = torch.sum(q2_pred * q2_true, dim=-1)
        dot2 = torch.clamp(dot2, -1.0, 1.0)

        dots = torch.cat([dot1, dot2], dim=0)
        return F.mse_loss(dots, torch.ones_like(dots))
