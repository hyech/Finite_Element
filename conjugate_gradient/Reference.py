import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

infile = open('FEM_converted.json', 'r')
var_dict = json.load(infile)
infile.close()

sigmoid = nn.Sigmoid()

nnode = var_dict["nnode"]
xcoord = torch.from_numpy(np.asarray([i[0] for i in var_dict["coord"]]))
ycoord = torch.from_numpy(np.asarray([i[1] for i in var_dict["coord"]]))
nelem = var_dict["nelem"]
connect = var_dict["connect"]
nfix = var_dict["nfix"]
fixnodes = var_dict["fixnodes"]
ndload = var_dict["ndload"]
dloads = var_dict["dloads"]


class steelAl(nn.Module):
    def __init__(self, nnode):
        super().__init__()
        self.stiff = torch.zeros([2 * nnode, 2 * nnode], dtype=torch.float64)
        self.resid = torch.zeros([2 * nnode, 1], dtype=torch.float64)


def create_Dmat(nu, E):
    out = torch.mul(E / ((1 + nu) * (1 - 2 * nu)),
                    torch.tensor([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]], dtype=torch.float64))
    return out.type(torch.float64)


def elresid(xa, ya, xb, yb, tx, ty):
    length = np.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb))
    length.type(torch.float64)
    out = torch.mul(torch.tensor([tx, ty, tx, ty]), length / 2)
    return out


def elstiff(xa, ya, xb, yb, xc, yc, Dmat):
    # Define B matrix
    nax = -(yc - yb) / ((ya - yb) * (xc - xb) - (xa - xb) * (yc - yb))
    nay = (xc - xb) / ((ya - yb) * (xc - xb) - (xa - xb) * (yc - yb))
    nbx = -(ya - yc) / ((yb - yc) * (xa - xc) - (xb - xc) * (ya - yc))
    nby = (xa - xc) / ((yb - yc) * (xa - xc) - (xb - xc) * (ya - yc))
    ncx = -(yb - ya) / ((yc - ya) * (xb - xa) - (xc - xa) * (yb - ya))
    ncy = (xb - xa) / ((yc - ya) * (xb - xa) - (xc - xa) * (yb - ya))
    area = (1 / 2) * torch.abs((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))
    area = area.type(torch.float64)
    Bmat = torch.tensor([[nax, 0, nbx, 0, ncx, 0],
                         [0, nay, 0, nby, 0, ncy],
                         [nay, nax, nby, nbx, ncy, ncx]])

    # Return element stifffness
    out = torch.mul(torch.mm(torch.mm(torch.transpose(Bmat, 0, 1), Dmat), Bmat), area)
    return out.type(torch.float64)


# ----------Init stiff and resid----------


class InitAll(nn.Module):
    def __init__(self, var_dict):
        super().__init__()
        self.nnode = var_dict["nnode"]
        self.xcoord = torch.from_numpy(np.asarray([i[0] for i in var_dict["coord"]]))
        self.ycoord = torch.from_numpy(np.asarray([i[1] for i in var_dict["coord"]]))
        self.nelem = var_dict["nelem"]
        self.connect = var_dict["connect"]
        self.nfix = var_dict["nfix"]
        self.fixnodes = var_dict["fixnodes"]
        self.ndload = var_dict["ndload"]
        self.dloads = var_dict["dloads"]

        self.steelAL = steelAl(self.nnode)
        self.stiff = self.steelAL.stiff
        self.resid = self.steelAL.resid

    def forward(self):
        # Young's Modulus and Poisson's Ratio for Steel
        E_st = 200
        nu_st = 0.3
        # For Aluminum
        E_al = 69
        nu_al = 0.33

        # Steel D-matrix
        Dmat_st = create_Dmat(nu_st, E_st)
        # Aluminum D-matrix
        Dmat_al = create_Dmat(nu_al, E_al)

        for lmn in range(self.nelem):
            a = connect[lmn][0]
            b = connect[lmn][1]
            c = connect[lmn][2]

            if self.elem_material[lmn] > 0.5:
                k = elstiff(self.xcoord[a], self.ycoord[a],
                            self.xcoord[b], self.ycoord[b],
                            self.xcoord[c], self.ycoord[c], Dmat_al)
            else:
                k = elstiff(self.xcoord[a], self.ycoord[a],
                            self.xcoord[b], self.ycoord[b],
                            self.xcoord[c], self.ycoord[c], Dmat_st)

            for i in range(3):
                for ii in range(2):
                    for j in range(3):
                        for jj in range(2):
                            rw = 2 * (self.connect[lmn][i]) + ii
                            cl = 2 * (self.connect[lmn][j]) + jj
                            self.stiff[rw][cl] = self.stiff[rw][cl] + k[2 * i + ii][2 * j + jj]

        pointer = [1, 2, 0]
        for i in range(self.ndload):
            lmn = self.dloads[i][0]
            face = self.dloads[i][1]
            a = connect[lmn][face]
            b = connect[lmn][pointer[face]]
            r = elresid(self.xcoord[a], self.ycoord[a],
                        self.xcoord[b], self.ycoord[b],
                        self.dloads[i][2], self.dloads[i][3])

            self.resid[2 * a] = self.resid[2 * a] + r[0]
            self.resid[2 * a + 1] = self.resid[2 * a + 1] + r[1]
            self.resid[2 * b] = self.resid[2 * b] + r[2]
            self.resid[2 * b + 1] = self.resid[2 * b + 1] + r[3]

        for i in range(self.nfix):
            rw = 2 * (self.fixnodes[i][0]) + self.fixnodes[i][1]
            for j in range(2 * self.nnode):
                self.stiff[rw][j] = 0
            self.stiff[rw][rw] = 1.0
            self.resid[rw] = self.fixnodes[i][2]

        return self.stiff, self.resid

'''
class SolveAll(nn.Module):
    def __init__(self, var_dict):
        super().__init__()
        self.var_dict = var_dict
        self.nnode = var_dict["nnode"]
        self.xcoord = torch.from_numpy(np.asarray([i[0] for i in self.var_dict["coord"]]))
        self.ycoord = torch.from_numpy(np.asarray([i[1] for i in self.var_dict["coord"]]))
        self.nelem = var_dict["nelem"]

        elem_material = torch.randn(self.nelem, dtype=torch.float64)
        self.elem_material = nn.Parameter(elem_material, requires_grad=True)

        self.steelAL = steelAl(self.nnode)
        self.stiff = self.steelAL.stiff
        self.resid = self.steelAL.resid

        self.net1 = CreateNet(self.var_dict, self.elem_material)
        self.net2 = SolverNet(self.xcoord, self.ycoord)

    def forward(self):
        self.stiff, self.resid, out = self.net1()
        u, uxcoord, uycoord = self.net2()
        return u, uxcoord, uycoord, sigmoid(self.elem_material), stiff


# ----------Step 1----------


class CreateNet(nn.Module):
    def __init__(self, var_dict, elem_material):
        super().__init__()
        self.var_dict = var_dict
        self.elem_material = elem_material
        self.nnode = self.var_dict["nnode"]
        self.xcoord = torch.from_numpy(np.asarray([i[0] for i in self.var_dict["coord"]]))
        self.ycoord = torch.from_numpy(np.asarray([i[1] for i in self.var_dict["coord"]]))
        self.connect = self.var_dict["connect"]
        self.nfix = self.var_dict["nfix"]
        self.fixnodes = self.var_dict["fixnodes"]
        self.ndload = var_dict["ndload"]
        self.dloads = self.var_dict["dloads"]

        self.net = AssembleStiff(self.var_dict, self.elem_material)

    def forward(self):
        stiff = self.net()

        pointer = [1, 2, 0]
        for i in range(self.ndload):
            lmn = self.dloads[i][0]
            face = self.dloads[i][1]
            a = self.connect[lmn][face]
            b = self.connect[lmn][pointer[face]]
            r = elresid(self.xcoord[a], self.ycoord[a], self.xcoord[b], self.ycoord[b], self.dloads[i][2],
                        self.dloads[i][3])

            resid[2 * a] = resid[2 * a] + r[0]
            resid[2 * a + 1] = resid[2 * a + 1] + r[1]
            resid[2 * b] = resid[2 * b] + r[2]
            resid[2 * b + 1] = resid[2 * b + 1] + r[3]

        for i in range(self.nfix):
            rw = 2 * (self.fixnodes[i][0]) + self.fixnodes[i][1]
            for j in range(2 * self.nnode):
                stiff[rw][j] = 0
            stiff[rw][rw] = 1.0
            resid[rw] = self.fixnodes[i][2]

        return stiff, resid, self.elem_material


class AssembleStiff(nn.Module):
    def __init__(self, var_dict, elem_material):
        super().__init__()
        self.var_dict = var_dict
        self.nnode = var_dict["nnode"]
        self.xcoord = torch.from_numpy(np.asarray([i[0] for i in self.var_dict["coord"]]))
        self.ycoord = torch.from_numpy(np.asarray([i[1] for i in self.var_dict["coord"]]))
        self.nelem = var_dict["nelem"]
        self.connect = var_dict["connect"]

        sigmoid = nn.Sigmoid()
        self.elem_material = sigmoid(elem_material)

        # self.stiff = torch.zeros([2 * self.nnode, 2 * self.nnode], dtype=torch.float64)
        print(self.elem_material[1])
        # Young's Modulus and Poisson's Ratio for Steel
        E_st = 200
        nu_st = 0.3
        # For Aluminum
        E_al = 69
        nu_al = 0.33

        # Steel D-matrix
        self.Dmat_st = create_Dmat(nu_st, E_st)
        # Aluminum D-matrix
        self.Dmat_al = create_Dmat(nu_al, E_al)

        # Mixed D-matrix
        self.Dmat = torch.zeros([3, 3, self.nelem], dtype=torch.float64)
        for i in range(self.nelem):
            self.Dmat[:, :, i] = torch.mul(self.Dmat_al, self.elem_material[i]) + torch.mul(self.Dmat_st,
                                                                                            (1 - self.elem_material[i]))

        layers = []
        for i in range(self.nelem):
            layers.append(ElstiffLayer(self.connect, i, self.xcoord, self.ycoord, self.Dmat))

        self.layers = nn.Sequential(*layers)

    def forward(self):
        global stiff, resid

        input = stiff
        out = self.layers(input)

        return out


class ElstiffLayer(nn.Module):
    def __init__(self, connect, lmn, xcoord, ycoord, Dmat_st, Dmat_al):
        super().__init__()
        a = connect[lmn][0]
        b = connect[lmn][1]
        c = connect[lmn][2]
        self.Dmat_st = Dmat_st
        self.Dmat_al = Dmat_al
        self.Dmat =

        self.indicator = torch.tensor([2 * a, 2 * a + 1, 2 * b, 2 * b + 1, 2 * c, 2 * c + 1])
        self.elstiff = elstiff(xcoord[a], ycoord[a], xcoord[b], ycoord[b], xcoord[c], ycoord[c], self.Dmat)

    def forward(self, stiff):
        for j in range(6):
            for k in range(6):
                stiff[self.indicator[j], self.indicator[k]] = stiff[self.indicator[j], self.indicator[k]] + \
                                                              self.elstiff[j, k]

        return stiff
'''

# ----------Step 2----------


class SolverNet(nn.Module):
    def __init__(self, stiff, resid, xcoord, ycoord):
        super().__init__()
        self.resid = resid
        self.stiff = stiff
        self.xcoord = xcoord
        self.ycoord = ycoord

        self.net = CustomModule()

    def forward(self):
        torch.split()
        out = self.net()
        result, stiff = out.spilt(1, 2)
        r = result[:, 0]
        u = result[:, 1]
        p = result[:, 2]

        uxcoord = [0] * nnode
        uycoord = [0] * nnode
        for i in range(nnode):
            with torch.no_grad():
                uxcoord[i] = self.xcoord[i] + u[2 * i]
                uycoord[i] = self.ycoord[i] + u[2 * i + 1]
        return u, uxcoord, uycoord


class CustomModule(nn.Module):
    def __init__(self, resid, stiff):
        super().__init__()
        self.stiff = stiff
        self.resid = resid
        self.u = torch.zeros([2 * nnode, 1], dtype=torch.float64)
        self.r = resid - torch.mm(stiff, self.u)

        layers = []
        for i in range(2 * nnode):
            layers.append(CustomLayer())

        self.layers = nn.Sequential(*layers)

    def forward(self):
        input = torch.zeros([2 * nnode, 2 * nnode, 2], dtype=torch.float64)
        input[:, 0, 0] = self.r
        input[:, 1, 0] = self.u
        input[:, 2, 0] = self.r
        input[:, :, 1] = self.stiff
        out = self.layers(input)
        return out


class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        torch.split()
        input, stiff = x.spilt(1, 2)
        r = input[:, 0]
        u = input[:, 1]
        p = input[:, 2]

        alpha = torch.mm(torch.transpose(r, 0, 1), r) / torch.mm(torch.mm(torch.transpose(p, 0, 1), stiff), p)
        u = u + torch.mul(alpha, p)
        r_new = r - torch.mul(alpha, torch.mm(stiff, p))
        beta = torch.mm(torch.transpose(r_new, 0, 1), r_new) / torch.mm(torch.transpose(r, 0, 1), r)
        p = r_new + torch.mul(beta, p)
        r = r_new

        out = torch.zeros([2 * nnode, 2 * nnode, 2], dtype=torch.float64)
        out[:, 0, 0] = r
        out[:, 1, 0] = u
        out[:, 2, 0] = p
        out[:, :, 1] = stiff
        return out


'''
class SolverNet(nn.Module):
    def __init__(self, nnode, stiff, resid, xcoord, ycoord):
        super().__init__()
        self.inverse = None
        self.u = None
        self.size_in = 2 * nnode
        self.size_out = 2 * nnode
        self.nnode = nnode
        self.xcoord = xcoord
        self.ycoord = ycoord
        weights = torch.tensor(stiff)
        self.resid = torch.tensor(resid)
        self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self):
        self.inverse = torch.linalg.inv(self.weights)
        self.u = torch.matmul(self.inverse, self.resid)
        uxcoord = [0] * nnode
        uycoord = [0] * nnode
        for i in range(self.nnode):
            with torch.no_grad():
                uxcoord[i] = self.xcoord[i] + self.u[2 * i]
                uycoord[i] = self.ycoord[i] + self.u[2 * i + 1]
        return self.u, self.inverse, uxcoord, uycoord


def backward(u, resid, stiff):
    inverse = torch.linalg.inv(torch.tensor(stiff))
    loss_fn = nn.MSELoss()
    target = torch.zeros(2 * nnode)
    loss = loss_fn(u.float(), target.float())
    #grad_1 = torch.autograd.grad(loss, u, grad_outputs=torch.zeros_like(u))
    grad_2 = torch.zeros(2 * nnode, 2 * nnode)
    for i in range(2 * nnode):
        grad_2[i, :] = torch.tensor(resid)
    grad_3 = torch.kron(torch.mul(inverse, -1), torch.transpose(inverse, 0, 1))
    return torch.matmul(grad_2, grad_3)
'''
net_init = initAll(var_dict)
st, rs = net_init()
net_solve = SolveAll(var_dict)
u, uxcoord, uycoord, elem_material, stiff = net_solve()
'''
resid = torch.reshape(torch.tensor(resid), (resid.size, 1))
stiff = torch.tensor(stiff)
print(resid.shape)
print(stiff.shape)
print(torch.norm(resid, p=2, dim=0))
print(torch.matmul(torch.transpose(resid, 0, 1), resid))

net_solve = SolverNet(stiff, resid, xcoord, ycoord)
u, uxcoord, uycoord, temp = net_solve()
'''

# print部分，勿动
# print(elem_material)
# print(stiff)
facecolors = [0] * nelem

for lmn in range(nelem):
    if elem_material[lmn] > 0.5:
        facecolors[lmn] = 1
    else:
        facecolors[lmn] = 0

fig, ax = plt.subplots()
triang = mtri.Triangulation(xcoord, ycoord, connect)
triang_2 = mtri.Triangulation(uxcoord, uycoord, connect)
plt.tripcolor(triang, ax, facecolors=facecolors, edgecolors='green', linewidth=2, cmap='Greys', alpha=0.5)
plt.tripcolor(triang_2, ax, facecolors=facecolors, edgecolors='red', linewidth=2, cmap='Greys', alpha=0.5)
plt.show()

# ----------Training Process----------


num_epochs = 3
learning_rate = 0.001
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net_solve.parameters(), lr=learning_rate)

target = torch.zeros([2 * nnode, 1], dtype=torch.float64)

for epoch in range(num_epochs):
    if epoch > 2:
        learning_rate = 50

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    u, uxcoord, uycoord, temp, tempii = net_solve()

    loss = loss_fn(u, target)
    print(loss)
    if loss < 0.001:
        break
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

u, uxcoord, uycoord, elem_materialii, stiffii = net_solve()
# print(stiffii)
facecolorsii = [0] * nelem

for lmn in range(nelem):
    if elem_materialii[lmn] > 0.5:
        facecolorsii[lmn] = 1
    else:
        facecolorsii[lmn] = 0

fig, ax = plt.subplots()
triang = mtri.Triangulation(xcoord, ycoord, connect)
triang_2 = mtri.Triangulation(uxcoord, uycoord, connect)
plt.tripcolor(triang, ax, facecolors=facecolorsii, edgecolors='green', linewidth=2, cmap='Greys', alpha=0.5)
plt.tripcolor(triang_2, ax, facecolors=facecolorsii, edgecolors='red', linewidth=2, cmap='Greys', alpha=0.5)
plt.show()

'''
temp1, temp2, temp3, stiff_result = net_solve()
u_result = torch.mm(torch.linalg.inv(stiff_result), torch.reshape(torch.tensor(resid), (resid.size, 1)))
uxcoord_result = [0] * nnode
uycoord_result = [0] * nnode
for i in range(nnode):
    with torch.no_grad():
        uxcoord_result[i] = xcoord[i] + u_result[2 * i]
        uycoord_result[i] = ycoord[i] + u_result[2 * i + 1]

print(stiff_result)
print(elem_material[0])
facecolors = [0] * nelem


for lmn in range(nelem):
    if elem_material[lmn] > 0.5:
        facecolors[lmn] = 1
    else:
        facecolors[lmn] = 0

fig, ax = plt.subplots()
triang = mtri.Triangulation(xcoord, ycoord, connect)
triang_2 = mtri.Triangulation(uxcoord_result, uycoord_result, connect)
plt.tripcolor(triang, ax, facecolors=facecolors, edgecolors='green', linewidth=2, cmap='Greys', alpha=0.5)
plt.tripcolor(triang_2, ax, facecolors=facecolors, edgecolors='red', linewidth=2, cmap='Greys', alpha=0.5)
plt.show()


#--------------------


net = steelAlNet(var_dict)

num_epochs = 5
learning_rate = 100
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net_solve.parameters(), lr=learning_rate)

ideal_weights = [0.0027] * nelem

for epoch in range(num_epochs):
    optimizer.zero_grad()
    #elem_material, elem_weight, xcoord, ycoord, stiff, resid = net()
    uxcoord, uycoord = net_solve()
    #print(elem_material[0])

    facecolors = [0] * nelem

    for lmn in range(nelem):
        if elem_material[lmn] > 0.5:
            facecolors[lmn] = 1
        else:
            facecolors[lmn] = 0

    fig, ax = plt.subplots()
    triang = mtri.Triangulation(xcoord, ycoord, connect)
    triang2 = mtri.Triangulation(uxcoord, uycoord, connect)
    plt.tripcolor(triang, ax, facecolors=facecolors, edgecolors='green', linewidth=2, cmap='Greys', alpha=0.5)
    plt.tripcolor(triang2, ax, facecolors=facecolors, edgecolors='r', linewidth=2, cmap='Greys', alpha=0.5)
    plt.show()

    x_loss = loss_fn(torch.tensor(uxcoord, requires_grad=True), torch.tensor(xcoord)) * 100
    y_loss = loss_fn(torch.tensor(uycoord, requires_grad=True), torch.tensor(ycoord)) * 100
    weight_difference = [elem_material[i] + (elem_weight[i] - ideal_weights[i]) for i in range(nelem)]
    loss = loss_fn(elem_material, torch.tensor(weight_difference))
    loss = loss - x_loss - y_loss
    print(loss)
    loss.backward()
    optimizer.step()
'''
