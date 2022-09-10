import json

def convert_data_to_json(filename):
    infile = open(filename, 'r')

    var_dict = {}
    nodeCoordFlag = False
    connectFlag = False
    fixNodeFlag = False
    dloadFlag = False
    coordArray = []
    connectArray = []
    fixNodeArray = []
    dloadArray = []

    for line in infile:
        splits = line.split()
        if splits[0] == "Young's_modulus:":
            var_dict["E"] = eval(splits[1])
        if splits[0] == "Poissons_ratio:":
            var_dict["nu"] = eval(splits[1])
        if splits[0] == "No._nodes:":
            var_dict["nnode"] = eval(splits[1])
        if splits[0] == "No._elements:":
            nodeCoordFlag = False
            var_dict["coord"] = coordArray
            var_dict["nelem"] = eval(splits[1])
        if nodeCoordFlag:
            coordArray.append([eval(splits[0]), eval(splits[1])])
        if splits[0] == "Nodal_coords:":
            nodeCoordFlag = True
        if splits[0] == "No._nodes_with_prescribed_DOFs:":
            connectFlag = False
            var_dict["connect"] = connectArray
            var_dict["nfix"] = eval(splits[1])
        if connectFlag:
            connectArray.append([eval(splits[0]) - 1, eval(splits[1]) - 1, eval(splits[2]) - 1])
        if splits[0] == "Element_connectivity:":
            connectFlag = True
        if splits[0] == "No._elements_with_prescribed_loads:":
            fixNodeFlag = False
            var_dict["fixnodes"] = fixNodeArray
            var_dict["ndload"] = eval(splits[1])
        if fixNodeFlag:
            fixNodeArray.append([eval(splits[0]) - 1, eval(splits[1]) - 1, eval(splits[2])])
        if splits[0] == "Node_#":
            fixNodeFlag = True
        if dloadFlag:
            dloadArray.append([eval(splits[0]) - 1, eval(splits[1]) - 1, eval(splits[2]), eval(splits[3])])
        if splits[0] == "Element_#":
            dloadFlag = True

    var_dict["dloads"] = dloadArray

    infile.close()

    with open("FEM_converted.json", "w") as outfile:
        json.dump(var_dict, outfile)

def main():
    #convert_data_to_json('FEM_conststrain_input.txt')
    convert_data_to_json('FEM_constrain_holeplate.txt')

if __name__ == "__main__":
    main()