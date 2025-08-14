import torch
import torch.nn as nn
from UniNet_lib.loss import loss
import numpy as np


class UniNet(nn.Module):
    def __init__(
        self, c, domain_teacher, frozen_teacher, bottleneck, student, DFS=None
    ):
        super.__init__()
        self.T = c.T
        self.teachers = Teachers(
            frozen_teacher=frozen_teacher, domain_teacher=domain_teacher
        )
        self.bn = BN(bottleneck)
        self.student = Student(student)
        self.dfs = DFS

    def train_eval(self, phase="train"):
        self.phase = phase
        self.teachers.train_eval(phase)
        self.bn.train_eval(phase)
        self.student.train_eval(phase)


class Teachers(nn.Module):
    def __init__(self, domain_teacher, frozen_teacher):
        super.__init__()
        self.d_t = domain_teacher
        self.f_t = frozen_teacher

    def train_eval(self, phase="train"):
        self.phase = phase
        self.f_t.eval()
        if phase == "train":
            self.d_t.train()
        else:
            self.d_t.eval()

    def forward(self, x):
        with torch.no_grad():
            domain_features = self.d_t(x)
        frozen_features = self.f_t(x)

        bn_input = [
            torch.cat(d_f, f_f, dim=0)
            for d_f, f_f in zip(domain_features, frozen_features)
        ]

        return domain_features + frozen_features, bn_input


class BN(nn.Module):
    def __init__(self, bottleneck):
        super.__init__()
        self.bn = bottleneck

    def train_eval(self, phase="train"):
        if phase == "train":
            self.bn.train()
        else:
            self.bn.eval()

    def forward(self, x):
        return self.bn(x)


class Student(nn.Module):
    def __init__(self, student):
        super.__init__()
        self.student = student

    def train_eval(self, phase="train"):
        if phase == "train":
            self.student.train()
        else:
            self.student.eval()

    def forward(self, x):
        return self.student(x)
