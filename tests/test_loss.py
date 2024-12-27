import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'DKL'))
from DKL.loss import my_loss
import torch


def test_loss_1():
    nModality = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ = torch.randn(nModality, 64, 64).to(device)

    loss_type = ["MSE", "centeral_alignment_cortes", "centeral_alignment_cris", "u_centeral_alignment_cortes",
                 "u_centeral_alignment_cris", "FSM", "structural_risk", "structural_risk_logloss"]
    output = torch.randn(1, 64, 64).to(device)
    predict = torch.randn(1, 64, 64).to(device)
    for i in range(len(loss_type)):
        print(f"Loss type: {loss_type[i]}")
        loss = my_loss(output, predict, loss_type[0])
        print(loss.item())

if __name__ == '__main__':
    test_loss_1()

