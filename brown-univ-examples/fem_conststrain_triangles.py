import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

# Function to create D-matrix
def create_Dmat(nu, E):
    return E/((1+nu)*(1-2*nu)) * np.asarray([[1-nu,nu,0],[nu,1-nu,0],[0,0,(1-2*nu)/2]])

# Function to create element stiffness matrix
def elstif(xa,ya,xb,yb,xc,yc,Dmat):
    # Define B matrix
    nax = -(yc-yb)/( (ya-yb)*(xc-xb) - (xa-xb)*(yc-yb) )
    nay =  (xc-xb)/( (ya-yb)*(xc-xb) - (xa-xb)*(yc-yb) )
    nbx = -(ya-yc)/( (yb-yc)*(xa-xc) - (xb-xc)*(ya-yc) )
    nby =  (xa-xc)/( (yb-yc)*(xa-xc) - (xb-xc)*(ya-yc) )
    ncx = -(yb-ya)/( (yc-ya)*(xb-xa) - (xc-xa)*(yb-ya) )
    ncy =  (xb-xa)/( (yc-ya)*(xb-xa) - (xc-xa)*(yb-ya) )
    area = (1/2)*np.abs( (xb-xa)*(yc-ya) - (xc-xa)*(yb-ya) )
    Bmat = np.asarray([[nax, 0, nbx, 0, ncx, 0],
                        [0, nay, 0, nby, 0, ncy],
                        [nay, nax, nby, nbx, ncy, ncx]])
    
    # Return element stiffness
    return area*np.dot(np.dot(np.transpose(Bmat), Dmat), Bmat)

# Function to create element residual force vector
def elresid(xa, ya, xb, yb, tx, ty):
    length = np.sqrt((xa-xb)*(xa-xb)+(ya-yb)*(ya-yb))
    return np.asarray([tx,ty,tx,ty])*length/2

# main function
def main():
    # Get variables
    infile = open('FEM_converted.json', 'r')

    var_dict = json.load(infile)

    infile.close()

    E = var_dict["E"]
    nu = var_dict["nu"]
    nnode = var_dict["nnode"]
    xcoord = np.asarray([i[0] for i in var_dict["coord"]])
    ycoord = np.asarray([i[1] for i in var_dict["coord"]])
    nelem = var_dict["nelem"]
    connect = var_dict["connect"]
    nfix = var_dict["nfix"]
    fixnodes = var_dict["fixnodes"]
    ndload = var_dict["ndload"]
    dloads = var_dict["dloads"]

    # Initialize Dmat
    Dmat = create_Dmat(nu, E)

    # Create global stiffness matrix
    Stif = np.zeros((2*nnode, 2*nnode))

    # For each element, compute the stiffness matrix
    #       then add to global stiffness matrix
    for lmn in range(nelem):
        a = connect[lmn][0]
        b = connect[lmn][1]
        c = connect[lmn][2]
        k = elstif(xcoord[a], ycoord[a], xcoord[b], ycoord[b], xcoord[c], ycoord[c], Dmat)

        for i in range(3):
            for ii in range(2):
                for j in range(3):
                    for jj in range(2):
                        rw = 2*(connect[lmn][i])+ii
                        cl = 2*(connect[lmn][j])+jj
                        Stif[rw][cl] = Stif[rw][cl] + k[2*i+ii][2*j+jj]    
    
    # Create global residual force vector
    resid = np.zeros(2*nnode)
    
    # Populate global residual force vector
    pointer = [1, 2, 0]
    for i in range(ndload):
        lmn = dloads[i][0]
        face = dloads[i][1]
        a = connect[lmn][face]
        b = connect[lmn][pointer[face]]
        r = elresid(xcoord[a], ycoord[a], xcoord[b], ycoord[b], dloads[i][2], dloads[i][3])
        
        resid[2*a]=resid[2*a]+r[0]
        resid[2*a+1]=resid[2*a+1]+r[1]
        resid[2*b]=resid[2*b]+r[2]
        resid[2*b+1]=resid[2*b+1]+r[3]

    # Add fixed node constraints
    for i in range(nfix):
        rw=2*(fixnodes[i][0])+fixnodes[i][1]
        for j in range(2*nnode):
            Stif[rw][j] = 0
        Stif[rw][rw] = 1.0
        resid[rw]=fixnodes[i][2]

    # Solve the linear equation of form Ax = B
    V, D = np.linalg.eig(Stif)
    eigenvecs = np.transpose(V)
    eigenvals = D
    u = np.linalg.solve(Stif, resid)

    # Calculate displacements
    uxcoord = np.zeros(nnode)
    uycoord = np.zeros(nnode)
    for i in range(nnode):
        uxcoord[i] = xcoord[i] + u[2*i]
        uycoord[i] = ycoord[i] + u[2*i+1]

    # Plot the resultant figure
    fig, ax = plt.subplots()
    triang = mtri.Triangulation(xcoord, ycoord, connect)
    triang2 = mtri.Triangulation(uxcoord, uycoord, connect)
    plt.triplot(triang, 'go-', ax)
    plt.triplot(triang2, 'ro-', ax)
    plt.show()

if __name__ == "__main__":
    main()