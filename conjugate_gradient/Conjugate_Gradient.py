# updated Nov 10 2022
# 1 -- Al, 0 -- Steel
import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import torch.nn as nn
print("gpu device name: {}".format(torch.cuda.get_device_name(torch.device("cuda:0"))))
device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.device('cuda')
    device = torch.device("cuda")

infile = open('FEM_converted.json', 'r')
var_dict = json.load(infile)
infile.close()

sigmoid = nn.Sigmoid()

global nnode
nnode = var_dict["nnode"]
xcoord = torch.tensor([i[0] for i in var_dict["coord"]], dtype=float, device=device)
ycoord = torch.tensor([i[1] for i in var_dict["coord"]], dtype=float, device=device)
nelem = var_dict["nelem"]
connect = var_dict["connect"]
nfix = var_dict["nfix"]
fixnodes = var_dict["fixnodes"]
ndload = var_dict["ndload"]
dloads = var_dict["dloads"]


def create_Dmat(nu, E):
    out = torch.mul(E / ((1 + nu) * (1 - 2 * nu)),
                    torch.tensor([[1 - nu, nu, 0],
                                  [nu, 1 - nu, 0],
                                  [0, 0, (1 - 2 * nu) / 2]]))

    out = out.type(torch.float64)
    return out.cuda()

def elresid(xa, ya, xb, yb, tx, ty):
    length = torch.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb))

    out = torch.mul(torch.tensor([tx, ty, tx, ty], device=device), length / 2)
    return out.cuda()



def elstiff(xa, ya, xb, yb, xc, yc, Dmat):
    # Define B matrix
    nax = -(yc - yb) / ((ya - yb) * (xc - xb) - (xa - xb) * (yc - yb))
    nay = (xc - xb) / ((ya - yb) * (xc - xb) - (xa - xb) * (yc - yb))
    nbx = -(ya - yc) / ((yb - yc) * (xa - xc) - (xb - xc) * (ya - yc))
    nby = (xa - xc) / ((yb - yc) * (xa - xc) - (xb - xc) * (ya - yc))
    ncx = -(yb - ya) / ((yc - ya) * (xb - xa) - (xc - xa) * (yb - ya))
    ncy = (xb - xa) / ((yc - ya) * (xb - xa) - (xc - xa) * (yb - ya))
    area = (1 / 2) * torch.abs((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))
    Bmat = torch.tensor([[nax, 0, nbx, 0, ncx, 0],
                         [0, nay, 0, nby, 0, ncy],
                         [nay, nax, nby, nbx, ncy, ncx]], device=device)

    # Return element stifffness
    out = torch.mul(torch.mm(torch.mm(torch.transpose(Bmat, 0, 1), Dmat), Bmat), area)
    out = out.type(torch.float64)
    return out.cuda()



# ----------Init stiff and resid----------


class SolveAll(nn.Module):
    def __init__(self, var_dict):
        super().__init__()
        self.var_dict = var_dict
        self.nnode = var_dict["nnode"]
        self.xcoord = torch.tensor([i[0] for i in var_dict["coord"]], dtype=float, device=device)
        self.ycoord = torch.tensor([i[1] for i in var_dict["coord"]], dtype=float, device=device)
        self.nelem = var_dict["nelem"]
        elem_material = torch.randn(self.nelem, dtype=torch.float64, device=device)

        self.elem_material = nn.Parameter(elem_material, requires_grad=True)

        self.u = torch.zeros([2 * nnode, 1], dtype=torch.float64, device=device)

        self.net = nn.Sequential(CreateNet(self.var_dict, self.elem_material),
                SolverNet(self.xcoord, self.ycoord))


    def forward(self):
        resid = torch.zeros([2 * nnode, 1], dtype=torch.float64, device=device)

        self.u, uxcoord, uycoord = self.net(resid)
        material = sigmoid(self.elem_material)
        material_reshape = torch.reshape(material, [nelem, 1])
        temp = torch.ones([1, nelem], dtype=torch.float64)
        #sum = torch.reshape(torch.mm(temp, material_reshape), [1, 1])
        sum = torch.mm(temp, material_reshape)

        return self.u, uxcoord, uycoord, material, sum


# ----------Step 1----------


class CreateNet(nn.Module):
    def __init__(self, var_dict, elem_material):
        super().__init__()
        self.var_dict = var_dict
        self.elem_material = elem_material
        self.nnode = self.var_dict["nnode"]
        self.xcoord = torch.tensor([i[0] for i in var_dict["coord"]], dtype=float, device=device)
        self.ycoord = torch.tensor([i[1] for i in var_dict["coord"]], dtype=float, device=device)

        self.connect = self.var_dict["connect"]
        self.nfix = self.var_dict["nfix"]
        self.fixnodes = self.var_dict["fixnodes"]
        self.ndload = var_dict["ndload"]
        self.dloads = torch.tensor(self.var_dict["dloads"], device=device)
        print(self.dloads.shape)


        self.net = AssembleStiff(self.var_dict, self.elem_material)

    def forward(self, resid):
        #print(self.elem_material[25])
        stiff = self.net()

        pointer = [1, 2, 0]
        for i in range(self.ndload):
            lmn = int(self.dloads[i][0])
            face = int(self.dloads[i][1])

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

        return torch.cat([stiff, resid], dim=1)


class AssembleStiff(nn.Module):
    def __init__(self, var_dict, elem_material):
        super().__init__()
        self.var_dict = var_dict
        self.nnode = var_dict["nnode"]
        self.xcoord = torch.tensor([i[0] for i in var_dict["coord"]], dtype=float, device=device)
        self.ycoord = torch.tensor([i[1] for i in var_dict["coord"]], dtype=float, device=device)

        self.nelem = var_dict["nelem"]
        self.connect = var_dict["connect"]

        self.elem_material = elem_material


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

        layers = []
        for i in range(self.nelem):
            layers.append(ElstiffLayer(self.connect, i, self.xcoord, self.ycoord, self.Dmat_st, self.Dmat_al, self.elem_material))

        self.layers = nn.Sequential(*layers)

    def forward(self):
        stiff = torch.zeros([2 * nnode, 2 * nnode], dtype=torch.float64, device=device)

        out = self.layers(stiff)
        return out


class ElstiffLayer(nn.Module):
    def __init__(self, connect, lmn, xcoord, ycoord, Dmat_st, Dmat_al, elem_material):
        super().__init__()
        self.lmn = lmn
        self.a = connect[self.lmn][0]
        self.b = connect[self.lmn][1]
        self.c = connect[self.lmn][2]
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.Dmat_st = Dmat_st
        self.Dmat_al = Dmat_al

        self.elem_material = elem_material

        self.indicator = torch.tensor([2 * self.a, 2 * self.a + 1, 2 * self.b, 2 * self.b + 1, 2 * self.c, 2 * self.c + 1])

    def forward(self, stiff):
        #with torch.no_grad():
        elem_m = sigmoid(self.elem_material)
        Dmat = torch.mul(self.Dmat_al, elem_m[self.lmn]) + torch.mul(self.Dmat_st, (1 - elem_m[self.lmn]))
        estiff = elstiff(self.xcoord[self.a], self.ycoord[self.a], self.xcoord[self.b], self.ycoord[self.b], self.xcoord[self.c], self.ycoord[self.c], Dmat)
        for j in range(6):
            for k in range(6):
                stiff[self.indicator[j], self.indicator[k]] = stiff[self.indicator[j], self.indicator[k]] + estiff[j, k]

        return stiff


# ----------Step 2----------


class SolverNet(nn.Module):
    def __init__(self, xcoord, ycoord):
        super().__init__()
        self.xcoord = xcoord
        self.ycoord = ycoord

        self.net = CustomModule()

    def forward(self, x):
        stiff, resid = x.split(2 * nnode, 1)
        u = torch.zeros([2 * nnode, 1], dtype=torch.float64, device=device)

        r = resid - torch.mm(stiff, u)

        out = self.net(u, r, stiff)
        result, temp = out.split(1, 2)
        r = result[:, 0]
        u = result[:, 1]
        p = result[:, 2]

        u = torch.reshape(u, [2 * nnode, 1])

        uxcoord = [0] * nnode
        uycoord = [0] * nnode
        for i in range(nnode):
            with torch.no_grad():
                uxcoord[i] = self.xcoord[i] + u[2 * i]
                uycoord[i] = self.ycoord[i] + u[2 * i + 1]
        return u, uxcoord, uycoord


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        for i in range(2 * nnode):
            layers.append(CustomLayer())

        self.layers = nn.Sequential(*layers)

    def forward(self, u, r, stiff):
        u = torch.reshape(u, (2 * nnode, 1, 1))
        r = torch.reshape(r, (2 * nnode, 1, 1))
        input = torch.zeros([2 * nnode, 2 * nnode, 2], dtype=torch.float64, device=device)

        input[:, 0, 0] = r[:, 0, 0]
        input[:, 1, 0] = u[:, 0, 0]
        input[:, 2, 0] = r[:, 0, 0]
        input[:, :, 1] = stiff
        out = self.layers(input)
        return out


class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        input, stiff = x.split(1, 2)
        stiff = torch.reshape(stiff, (2 * nnode, 2 * nnode))
        r = input[:, 0]
        u = input[:, 1]
        p = input[:, 2]

        alpha = torch.mm(torch.transpose(r, 0, 1), r) / torch.mm(torch.mm(torch.transpose(p, 0, 1), stiff), p)
        u = u + torch.mul(alpha, p)
        r_new = r - torch.mul(alpha, torch.mm(stiff, p))
        beta = torch.mm(torch.transpose(r_new, 0, 1), r_new) / torch.mm(torch.transpose(r, 0, 1), r)
        p = r_new + torch.mul(beta, p)
        r = r_new

        out = torch.zeros([2 * nnode, 2 * nnode, 2], dtype=torch.float64, device=device)

        out[:, 0, 0] = r[:, 0]
        out[:, 1, 0] = u[:, 0]
        out[:, 2, 0] = p[:, 0]
        out[:, :, 1] = stiff
        return out


# ----------Init the net----------


net = SolveAll(var_dict)
u, uxcoord, uycoord, elem_material, sum = net()



# ----------Plotting----------


facecolors = np.zeros([nelem])


for lmn in range(nelem):
    facecolors[lmn] = elem_material[lmn].detach().numpy()


fig, ax = plt.subplots()
triang = mtri.Triangulation(xcoord, ycoord, connect)
triang_2 = mtri.Triangulation(uxcoord, uycoord, connect)
plt.tripcolor(triang, ax, facecolors=np.zeros([nelem]), edgecolors='green', linewidth=2, cmap='Greys', alpha=0.5)
plt.tripcolor(triang_2, ax, facecolors=facecolors, edgecolors='red', linewidth=2, cmap='Greys', alpha=0.5)
print('a')

plt.show()


# ----------Training Process----------


num_epochs = 50
num_iteration = 5

learning_rate_1 = 100
learning_rate_2 = 0.001
loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()

optimizer_1 = torch.optim.SGD(net.parameters(), lr=learning_rate_1)
#optimizer_2 = torch.optim.SGD(net.parameters(), lr=learning_rate_2)


target = torch.zeros([2 * nnode, 1], dtype=torch.float64, device=device)
target_2 = torch.ones([nelem, 1], dtype=torch.float64, device=device)

#limit = torch.reshape(torch.tensor(0.4 * nelem, dtype=torch.float64), [1, 1])
limit = 0.4 * nelem
print('budget =', limit)

loss_old = 1000
x = 100
alpha = 0.01

for epoch in range(num_epochs):
    if torch.abs(sum - limit) < 1:
        learning_rate_1 = 1000
        x = 10000
        learning_rate_2 = 0.0001

    for param_group in optimizer_1.param_groups:
        param_group["lr"] = learning_rate_1

    for iter in range(num_iteration):
        u, uxcoord, uycoord, material_1, sum = net()
        if sum - limit > 0:
            loss_1 = x * loss_fn(u, target) + alpha * (sum - limit)
        else:
            loss_1 = x * loss_fn(u, target) - alpha * (sum - limit)

        #if loss_old - loss_1 < 0.00001:
           #break

        optimizer_1.zero_grad()
        loss_1.backward(retain_graph=True)
        optimizer_1.step()

    alpha = alpha - learning_rate_2 * (sum - limit)

    #inside loop: optimized for loss1 + y * loss2
    #outside loop: optimized for y

    print('epoch =', epoch, '; sum =', sum[0, 0].detach().numpy(), '; loss =', loss_1[0, 0].detach().numpy(), '; alpha =', alpha[0, 0].detach().numpy())

    if epoch % 1 == 0:

        for lmn in range(nelem):
            facecolors[lmn] = material_1[lmn].detach().numpy()

        fig, ax = plt.subplots()
        triang = mtri.Triangulation(xcoord, ycoord, connect)
        triang_2 = mtri.Triangulation(uxcoord, uycoord, connect)
        #plt.tripcolor(triang, ax, facecolors=[0] * nelem, edgecolors='green', linewidth=2, cmap='Greys', alpha=0.5)
        plt.tripcolor(triang_2, ax, facecolors=facecolors, edgecolors='red', linewidth=2, cmap='Greys', alpha=0.5)
        plt.show()

    loss_old = loss_1



u, uxcoord, uycoord, material_2, temp_2 = net()

facecolors_2 = np.zeros([nelem])


for lmn in range(nelem):
    if material_2[lmn] > 0.5:
        facecolors_2[lmn] = 1
    else:
        facecolors_2[lmn] = 0

fig, ax = plt.subplots()
triang = mtri.Triangulation(xcoord, ycoord, connect)
triang_2 = mtri.Triangulation(uxcoord, uycoord, connect)
plt.tripcolor(triang, ax, facecolors=facecolors_2, edgecolors='green', linewidth=2, cmap='Greys', alpha=0.5)
plt.tripcolor(triang_2, ax, facecolors=facecolors_2, edgecolors='red', linewidth=2, cmap='Greys', alpha=0.5)
plt.show()

