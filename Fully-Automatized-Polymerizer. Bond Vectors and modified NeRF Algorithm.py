#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import numpy as np  # used to account for the mathematical and vectorial calculation functions
import re  # regex will be used in computing the bond and torsion angles
import matplotlib.pyplot as plt  # required for the example plot in the Computations sequence (Part 4)

#%matplotlib inline

"""store atomic radii (Angstroms), nr. of bonds, Atomic mass (amu) for elements required for simple 
hydrocarbons and  their derivatives as dictionary"""

# only the most usitated ones were introduced

rad = {
    "H": 0.37,
    "C": 0.77,
    "O": 0.73,
    "N": 0.75,
    "F": 0.71,  # atomic raddii
    "P": 1.10,
    "S": 1.03,
    "Cl": 0.99,
    "Br": 1.14,
    "I": 1.33,
}

no_valencies = {
    "H": 1,
    "C": 4,
    "O": 2,
    "N": 3,
    "F": 1,  # number of bonds an atom can form
    "P": 4,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

atomic_masses = {
    "H": 1,
    "He": 4,
    "Li": 7,
    "Be": 9,
    "B": 11,
    "C": 12,
    "N": 14,
    "O": 16,
    "F": 19,
    "Cl": 35.5,
}

NA = 6.022e23  # Avogadro constant

Central_Atom = [
    "C",
    "O",
    "N",
    "S",
]  # introduce notion of central atom for most molecules in txt file this will be "C"


# In[ ]:


"""CLASSES OF VECTORIAL GEOMETRY CALCULATIONS"""


class PositionVectors:
    def __init__(
        self, a, b
    ):  # a and b represent 3D cartesian coordinates for 2 atoms/points
        self.a = a
        self.b = b

        self.r_AB = self.b - self.a  # position vector

        self.distance2 = (
            self.r_AB[0] * self.r_AB[0]
            + self.r_AB[1] * self.r_AB[1]
            + self.r_AB[2] * self.r_AB[2]
        )

        self.bond_length = np.sqrt(self.distance2)  # distance between two atoms

        if self.bond_length == 0:

            self.unit_vector = self.r_AB / 1
        else:
            self.unit_vector = self.r_AB / self.bond_length  # normalised unit vector


class VectorProducts:
    def __init__(self, unit_vec_1, unit_vec_2):
        self.unit_vec_1 = unit_vec_1  # the unit vectors will be computed with the aid of the PositionVectors class
        self.unit_vec_2 = unit_vec_2

    # class function that calculates the dot product
    def Dot_product(self):

        dot_p = (
            self.unit_vec_1[0] * self.unit_vec_2[0]
            + self.unit_vec_1[1] * self.unit_vec_2[1]
            + self.unit_vec_1[2] * self.unit_vec_2[2]
        )

        return dot_p

    # class function that calculates the cross product divided by the sinus of the angle formed by the intersecting planes
    def Cross_product(self):

        cos = self.Dot_product()
        sin = np.sqrt(abs(1 - cos * cos))  # sin^2 = 1 - cos^2
        if sin != 0:
            sin = sin
        else:
            sin = 1
        cross_p = [[] for p in range(3)]
        cross_p[0] = (
            self.unit_vec_1[1] * self.unit_vec_2[2]
            - self.unit_vec_1[2] * self.unit_vec_2[1]
        ) / sin
        cross_p[1] = (
            self.unit_vec_1[2] * self.unit_vec_2[0]
            - self.unit_vec_1[0] * self.unit_vec_2[2]
        ) / sin
        cross_p[2] = (
            self.unit_vec_1[0] * self.unit_vec_2[1]
            - self.unit_vec_1[1] * self.unit_vec_2[0]
        ) / sin
        return cross_p


# In[ ]:


"""Functions that return the bond angles and digedral angles"""


# Function to calculate bond angles between any 3 bonded atoms
def Bond_Angle(at_A_xyz_coord, at_B_xyz_coord, at_C_xyz_coord):
    r_AB = PositionVectors(
        at_A_xyz_coord, at_B_xyz_coord
    )  # position vector defined by cartesian points A and B
    r_BC = PositionVectors(
        at_C_xyz_coord, at_B_xyz_coord
    )  # position vector defined by cartesian points C and B

    u_ABC = VectorProducts(r_AB.unit_vector, r_BC.unit_vector)
    angle_ABC = (
        u_ABC.Dot_product()
    )  # compute the dot product between the corresponding unit vectors of the position vectors defined above in that direction
    round_angle_ABC = round(angle_ABC, 3)
    Bond_Angle_ABC = (
        180 * (np.arccos(round_angle_ABC))
    ) / np.pi  # radians to degrees conversion

    return Bond_Angle_ABC


# Function to calculate dihedral angle between any 4 bonded atoms
def Torsion_Angle(at_A_xyz_coord, at_B_xyz_coord, at_C_xyz_coord, at_D_xyz_coord):
    r_AB = PositionVectors(
        at_A_xyz_coord, at_B_xyz_coord
    )  # position vector defined by cartesian points A and B
    r_BC = PositionVectors(
        at_C_xyz_coord, at_B_xyz_coord
    )  # position vector defined by cartesian points C and B

    u_ABC = VectorProducts(
        r_AB.unit_vector, r_BC.unit_vector
    )  # compute the dot product between the corresponding unit vectors

    r_CB = PositionVectors(
        at_B_xyz_coord, at_C_xyz_coord
    )  # position vector defined by cartesian points B and C
    r_CD = PositionVectors(
        at_D_xyz_coord, at_C_xyz_coord
    )  # position vector defined by cartesian points C and D

    u_BCD = VectorProducts(
        r_CB.unit_vector, r_CD.unit_vector
    )  # dot product between coresponding unit vectors
    u_sgn = VectorProducts(
        r_AB.unit_vector, r_CD.unit_vector
    )  # relates to sgn of dihedral angle

    t_ABCD = (
        u_ABC.Cross_product()
    )  # compute the normal vector to plane defined by atoms A, B, C
    t_ACBD = (
        u_BCD.Cross_product()
    )  # compute the normal vector to plane defined by atoms B,C, D
    t_sgn = u_sgn.Cross_product()  # establish sgn (sign) of the dihedral angle

    phi = VectorProducts(t_ABCD, t_ACBD)

    phi_sgn = VectorProducts(t_sgn, r_BC.unit_vector)  # sgn dihedral angle
    Dot_sgn = phi_sgn.Dot_product()  # sgn
    if Dot_sgn > 0 or Dot_sgn < 0:
        sgn = Dot_sgn / abs(Dot_sgn)  # get the sgn only by division to absolute value

    else:
        sgn = 1  # condition to avoid divizion by zero

    tosion_ABCD = phi.Dot_product()  # compute sin of dihedral angle
    torsion_ABCD = round(tosion_ABCD, 3)

    Torsion_Angle_ABCD = (
        sgn * (180 * (np.arccos(torsion_ABCD))) / np.pi
    )  # radians to degrees conversion and sgn attribution
    return Torsion_Angle_ABCD


# In[ ]:


"""User input to introduce desired molecule in the form resembling a molecular formula. In the begginig all molecules
are going to be linear  and so to access the geometry of ethane for eg introduce C2H6. The list of available molecules
in text file will be provided if requested at the start of the code """

Instructions = input(
    'Would you like to see the list of available molecules ? Answer with "yes" or "no" : '
)
if Instructions == "yes" or Instructions == "no":

    if Instructions == "yes":

        print(
            "The list of available linear saturated molecules is :" "\n H2O",
            "\n CH4",
            "\n H2O2",
            "\n C2H6",
            "\n C2H5Cl" "\n C4H10",
            "\n C5H12",
            "\n C6H14",
            "\n C2H2",
            "\nLinear unsaturated molecules available for polymerization:" "\n C2H4",
            "\n C2H3Cl",
        )


molecule_1 = input("\nEnter Molecule: ")  # initial molecule

# Read the file and store the data in a list of lists
def file_read(fname):
    coord_list = []
    # test to see whether the introduced molecule is available in tha database or not
    print("Testing for errors coming from stored database")
    try:

        f = open(fname)
    except FileNotFoundError as e:
        print("Sorry the file you are looking for doesn't exist")
    else:
        # print(f.read())
        f.close()
    finally:
        print("Executing Code")
    with open(fname) as f:

        for line in f:

            coord_list.append(line.split())

    return coord_list  # returns all data as nested lists


# In[ ]:


class NeighboursLists:
    # the molecule variable referes to the molecules stored in the tx files
    def __init__(self, molecule):

        self.molecule = molecule

        self.coord_array = file_read(f"{self.molecule}.txt")
        # store the corrdinates as a list of lists.
        # The name array here only suggests that the nested lists will be later turned into arrays

        self.nr_at = int(
            len(self.coord_array) - 1
        )  # extract the number of atoms for the number of coordinates read in

        self.at_type = (
            []
        )  # empty list that will store the type of atom coordinetes correspond to as strings (see txt files)
        self.mass = (
            []
        )  # empty list that will store masses corresponding to atoms in at_type

        self.xyz_array = np.zeros(
            (self.nr_at, 3)
        )  # create "matrix-like" structure to store coordinates as arrays in order of reading
        for i in range(self.nr_at):
            self.at_type.append(
                self.coord_array[i + 1][0]
            )  # i+1 can be used as len(nr_at) = len(coord) - 1

            """this comes from the fact that first line does not have data. If you wish to use your own txt file without
             the introductory line , either copy-paste the first line from description or write "coordinate_array[i][0]" above"""

            self.mass.append(float(atomic_masses[self.coord_array[i + 1][0]]))
            # get the masses of each elements stored in atom_type by key-val dict
            # record xyz file as array of 3D cartesian coordinates
            for j in range(3):
                self.xyz_array[i][j] = float(self.coord_array[i + 1][j + 1])
                # replace by "coord_array[i][j]" if you wish to skip line 1

        # the mass list is turned into array using the numpy library to ease calculations
        self.Mass_array = np.array(self.mass)
        self.total_Mass = sum(self.Mass_array)

        self.Mass_array_cm = np.zeros((self.nr_at, 3))

        for i in range(self.nr_at):

            self.Mass_array_cm[i] = (
                self.Mass_array_cm[i] * self.xyz_array[i]
            )  # calculate the total molar mass to use in center of mass calculation

        # CENTER OF MASS CALCULATION

        self.x = []
        self.y = []
        self.z = []
        for i in range(self.nr_at):

            self.x.append(self.Mass_array_cm[i][0])
            self.y.append(self.Mass_array_cm[i][1])
            self.z.append(self.Mass_array_cm[i][2])

        self.X = np.array(self.x)
        self.Y = np.array(self.y)
        self.Z = np.array(self.z)
        self.CM = [
            sum(self.X) / self.total_Mass,
            sum(self.Y) / self.total_Mass,
            sum(self.Z) / self.total_Mass,
        ]  # center of mass coordinates

        # TRANSLATING TO THE CENTER OF MASS
        self.xyz_array_cm = self.xyz_array - self.CM

        # Create new nested list containing the atom type followed by corresponding coordinates stored as arrays
        self.list_ATtype_and_coord = []
        for i in range(self.nr_at):
            self.list_ATtype_and_coord.append(self.at_type[i])
            self.list_ATtype_and_coord.append(self.xyz_array_cm[i])
        # print(len(list_ATtype_and_coord))

        #  List comprehansion stores each atom -- xyz_array parir in an individual list
        self.listuta = [
            self.list_ATtype_and_coord[x : (2 + x)]
            for x in range(0, len(self.list_ATtype_and_coord), 2)
        ]
        # print(listuta)

        # split listuta into 2 separate lists containig the central atom coordinates and the other coord respectively

    def Central_Listuta(self):

        list_central_atoms = (
            []
        )  # will store only the lists corresponding to the central atoms in listuta
        for i in range(self.nr_at):

            if self.listuta[i][0] in Central_Atom:

                list_central_atoms.append(self.listuta[i])
        return list_central_atoms

    def Other_Listuta(self):
        list_central_atoms = self.Central_Listuta()
        for i in range(len(list_central_atoms)):

            self.listuta.remove(
                list_central_atoms[i]
            )  # by removing elements of list_central_atoms[i] from now on listuta will only contain the other atoms
        return self.listuta


# In[ ]:


# class returning a list of ordered bonded atoms and their corresponing bond lengths, normalised unit vectors and atomic coordinates
class GeometricPropertiesCentralAtoms:
    def __init__(self, list_central_atoms):

        self.list_central_atoms = list_central_atoms

        self.bond_lengths_chain = (
            []
        )  # list of bond lenghths between consecutive central atoms branched, cyclic or linear
        self.str_at_type_central = (
            []
        )  # list of central atom types as strings eg for C2H6 str_at_type_central = ['C', 'C']
        self.unit_vc_central_at = []  # list of corresponding unit vectors
        self.at_coord_central = (
            []
        )  # list of atomic coordinates corresponding to bonding atoms in the order they appear

        for i in range(len(self.list_central_atoms)):
            self.str_at_type_central.append(
                self.list_central_atoms[i][0]
            )  # appends in the order it received
            self.at_coord_central.append([self.list_central_atoms[i][1]])

            for j in range(len(self.list_central_atoms)):

                self.radii_distance = 1.20 * (
                    rad[self.list_central_atoms[i][0]]
                    + rad[self.list_central_atoms[j][0]]
                )  # formula explained in the documentaion 1.20 = threshold factor

                if i < j:

                    self.bonds = PositionVectors(
                        self.list_central_atoms[j][1], self.list_central_atoms[i][1]
                    )

                    if (
                        self.bonds.bond_length > 0
                        and self.bonds.bond_length < self.radii_distance
                    ):

                        self.bond_lengths_chain.append([self.bonds.bond_length])
                        self.unit_vc_central_at.append([self.bonds.unit_vector])


# In[ ]:


class GeometricPropertiesOtherAtoms:
    def __init__(self, list_central_atoms, listuta):

        self.list_central_atoms = list_central_atoms
        self.listuta = listuta
        self.bond_lengths_other = (
            []
        )  # list of the other atoms s.a H and X = Cl, Br, I that are bonded to a central atom
        self.unit_vc_oth_at = (
            []
        )  # list of correponding unit vectors for the bonded atoms
        self.str_at_type = (
            []
        )  # nested lists of other atoms bonded to each central one as strings eg for C2H6 O = [['H', 'H', 'H'], ['H', 'H', 'H']]
        self.at_coord_other = (
            []
        )  # list of atomic coordinates corresponding to bonding atoms in the order they appear
        # here was the big red coment
        for q in range(len(self.list_central_atoms)):
            # individual nested lists
            self.nested_str_at_type = []
            self.bond_lg_nst = []
            self.u_vc_nst = []
            self.at_coord_nst = []
            for j in range(
                len(self.listuta)
            ):  # the nested loop will fill each individual nested list

                self.radii_distance = 1.20 * (
                    rad[self.list_central_atoms[q][0]] + rad[self.listuta[j][0]]
                )
                self.bonds = PositionVectors(
                    self.listuta[j][1], self.list_central_atoms[q][1]
                )

                if (
                    self.bonds.bond_length > 0
                    and self.bonds.bond_length < self.radii_distance
                ):  # condition that atoms are bonded

                    self.u_vc_nst.append(
                        self.bonds.unit_vector
                    )  # within the if conditional only the unit vectors that leed to a bond legth respecting the if conditional are appended
                    self.at_coord_nst.append(
                        self.listuta[j][1]
                    )  # atomic coordinates forming the above unit vector
                    self.bond_lg_nst.append(
                        self.bonds.bond_length
                    )  # corresponding bond lengths
                    self.nested_str_at_type.append(
                        self.listuta[j][0]
                    )  # arranged as atoms corresp to C1 then C2 then C3 ... in order C are found in list_central_atoms

            self.str_at_type.append(
                self.nested_str_at_type
            )  # appending each list from the nested for loop => the nested lists
            self.bond_lengths_other.append(self.bond_lg_nst)
            self.unit_vc_oth_at.append(self.u_vc_nst)
            self.at_coord_other.append(self.at_coord_nst)


"""Creating this lists now it is known that the fisrt element of list str_at_type_central is bonded to the second 
one and to each of the elements in the first nested list """


# In[ ]:


def New_Coordinates(
    at_coord_central, at_coord_other, str_at_type_central, str_at_type
):  # stores the atomic coordinates xyz starting with all of the backbone atoms in order and continuing with all of the H atoms in order

    New_at_coord = []
    for j in range(len(at_coord_central)):

        New_at_coord = New_at_coord + at_coord_central[j]
    for i in range(len(at_coord_other)):

        New_at_coord = New_at_coord + at_coord_other[i]

    return New_at_coord


# print("\nHERE ARE THE NEW COOR", New_at_coord)


def My_very_new_coordinates(str_at_type_central, str_at_type):
    ordered_atoms = [
        ([str_at_type_central[i]] + str_at_type[i])
        for i in range(len(str_at_type_central))
    ]
    ordered_atoms_new = []
    for i in range(len(ordered_atoms)):
        ordered_atoms_new.append(ordered_atoms[i][0])
        ordered_atoms[i].remove(ordered_atoms[i][0])
    for i in range(len(ordered_atoms)):

        for j in range(len(ordered_atoms[i])):
            ordered_atoms_new.append(ordered_atoms[i][j])

    ordered_atoms_new_indices = [
        elm + str(int(index) + 1) for index, elm in enumerate(ordered_atoms_new)
    ]  # add increasing indices starting at 1
    return ordered_atoms_new_indices


def Bond_Lengths(bond_lengths_chain, bond_lengths_other):

    #  Create list bond_lengths_ordered similarly to hold the bond lengths in the exact same order as ordered_atoms
    Bond_length_C = []  # bonds between central atoms
    for i in range(len(bond_lengths_chain)):
        for j in range(len(bond_lengths_chain[i])):

            Bond_length_C.append(bond_lengths_chain[i][j])
    Bond_length = []  # bonds between an other atom and central atom
    for i in range(len(bond_lengths_other)):
        for j in range(len(bond_lengths_other[i])):

            Bond_length.append(bond_lengths_other[i][j])

    bond_lengths_ordered = Bond_length_C + Bond_length
    # print(bond_lengths_ordered)
    return bond_lengths_ordered


# In[ ]:


def Z_MATRIX(
    str_at_type_central, str_at_type, ordered_atoms_new_indices, nr_at, N
):  # N = number of monomers
    # fill the lines at the end of the central atoms chain (the ones that include the other/substituent atoms )
    line_indices_position_after_chain_end = []
    line_indices_position_up_to_terminal_atom = []
    stored_line_indices_position_after_chain_end = 0
    stored_line_indices_position_up_to_terminal_atom = len(str_at_type_central)

    for i in range(len(str_at_type)):

        stored_line_indices_position_after_chain_end = (
            stored_line_indices_position_up_to_terminal_atom
        )
        stored_line_indices_position_up_to_terminal_atom = (
            stored_line_indices_position_up_to_terminal_atom + len(str_at_type[i])
        )
        line_indices_position_after_chain_end.append(
            stored_line_indices_position_after_chain_end
        )
        line_indices_position_up_to_terminal_atom.append(
            stored_line_indices_position_up_to_terminal_atom
        )

    Z_matrix = np.zeros(
        (N * nr_at, 7), list
    )  # create matrix-like structure containing arrays of strings and floats
    for i in range(N * nr_at):

        for j in range(7):

            if Z_matrix[i][j] == int(0):

                Z_matrix[i][
                    j
                ] = " "  # replaces the 0's corresponding to empty spaces with empty space
    for i in range(N * nr_at):

        Z_matrix[i][0] = ordered_atoms_new_indices[
            i
        ]  # place all atoms and their labels in order on first column of Z_matrix

        for k_0 in range(
            1, len(str_at_type_central)
        ):  # start filling the second column by placing the imediate neighbour of each central atom

            Z_matrix[k_0][1] = ordered_atoms_new_indices[
                k_0 - 1
            ]  # starts at 'Atom1' placing it adjacent to 'Atom2' on the first column

        # this bit is just for column two general for all degrees of nesaturations and number of central atoms
        # var_1 and var_2 are equivalent of var_1 = stored_line_indices_position_after_chain_end
        # var_2 = stored_line_indices_position_up_to_terminal_atom
        # a new set of variables was used both for simplicity and to differentiate between filling the 2nd column
        # as opposed to filling the 4th and 6th columns
        var_1 = 0
        var_2 = len(str_at_type_central)

        for i in range(len(str_at_type)):
            var_1 = var_2
            var_2 = var_2 + len(str_at_type[i])
            for k in range(var_1, var_2):
                Z_matrix[k][1] = ordered_atoms_new_indices[i]

    # ONE C_ATOM
    if (
        len(str_at_type_central) == 1
    ):  # this bit accounts for molecules of a single central atom such as CH4, H2O etc
        for k_s in range(2, nr_at):
            Z_matrix[k_s][3] = ordered_atoms_new_indices[1]
        for k_st in range(2, nr_at):
            Z_matrix[k_st][5] = ordered_atoms_new_indices[2]
    else:

        for k_3 in range(
            2, len(str_at_type_central)
        ):  # start filling the forth column by placing the imediate neighbour of each central atom
            Z_matrix[k_3][3] = ordered_atoms_new_indices[
                k_3 - 2
            ]  # starts at 'C1' placing it adjacent to 'C2' on the second column

        for k_5 in range(
            3, len(str_at_type_central)
        ):  # start filling the fifth column by placing the imediate neighbour of each central atom
            Z_matrix[k_5][5] = ordered_atoms_new_indices[k_5 - 3]

        """This inputs will be overwritten if they do not fulfill the criteria for 
        eg len(str_at_type_central) < 4 by the next chunck of code based on conditionals"""

        # start filling columns 4 and 6 for the middle attoms "MIDDLE"
        for k_1 in range(
            line_indices_position_after_chain_end[1],
            line_indices_position_up_to_terminal_atom[1],
        ):

            for i in range((len(str_at_type_central) - 2)):

                Z_matrix[
                    k_1
                    + i
                    * (
                        line_indices_position_up_to_terminal_atom[1]
                        - line_indices_position_after_chain_end[1]
                    )
                ][3] = ordered_atoms_new_indices[i]
                Z_matrix[
                    k_1
                    + i
                    * (
                        line_indices_position_up_to_terminal_atom[1]
                        - line_indices_position_after_chain_end[1]
                    )
                ][5] = ordered_atoms_new_indices[
                    line_indices_position_after_chain_end[1]
                    + i
                    * (
                        line_indices_position_up_to_terminal_atom[1]
                        - line_indices_position_after_chain_end[1]
                    )
                    - 1
                ]

            Z_matrix[
                k_1
                + (
                    line_indices_position_up_to_terminal_atom[len(str_at_type) - 2]
                    - line_indices_position_up_to_terminal_atom[1]
                )
            ][3] = ordered_atoms_new_indices[len(str_at_type_central) - 1]

            if len(str_at_type_central) >= 3:

                Z_matrix[
                    k_1
                    + (
                        line_indices_position_up_to_terminal_atom[len(str_at_type) - 2]
                        - line_indices_position_up_to_terminal_atom[1]
                    )
                ][5] = ordered_atoms_new_indices[
                    line_indices_position_up_to_terminal_atom[1]
                    + (
                        line_indices_position_up_to_terminal_atom[len(str_at_type) - 2]
                        - line_indices_position_up_to_terminal_atom[1]
                    )
                ]

        for k_0 in range(
            line_indices_position_after_chain_end[0],
            line_indices_position_up_to_terminal_atom[0],
        ):  # ENDS

            Z_matrix[k_0][3] = ordered_atoms_new_indices[1]

            Z_matrix[
                k_0
                + (
                    line_indices_position_up_to_terminal_atom[len(str_at_type) - 1]
                    - line_indices_position_up_to_terminal_atom[0]
                )
            ][3] = ordered_atoms_new_indices[0 + len(str_at_type_central) - 2]

            if len(str_at_type_central) >= 3:

                Z_matrix[k_0][5] = ordered_atoms_new_indices[1 + 1]
                Z_matrix[
                    k_0
                    + (
                        line_indices_position_up_to_terminal_atom[len(str_at_type) - 1]
                        - line_indices_position_up_to_terminal_atom[0]
                    )
                ][5] = ordered_atoms_new_indices[0 + len(str_at_type_central) - 2 - 1]

            else:

                for k_01 in range(
                    (line_indices_position_after_chain_end[0] + 1),
                    line_indices_position_up_to_terminal_atom[0],
                ):

                    Z_matrix[k_01][5] = ordered_atoms_new_indices[
                        (line_indices_position_after_chain_end[1])
                    ]
                    Z_matrix[
                        k_0
                        + (
                            line_indices_position_up_to_terminal_atom[
                                len(str_at_type) - 1
                            ]
                            - line_indices_position_up_to_terminal_atom[0]
                        )
                    ][5] = ordered_atoms_new_indices[
                        line_indices_position_after_chain_end[0]
                    ]

            # TRIPLE BONDS
            if line_indices_position_up_to_terminal_atom[0] == (
                line_indices_position_after_chain_end[0] + 1
            ):  # this bit is for molecules with triple bonds (eg: C2H2) or peroxide H2O2
                for k_ac in range(
                    line_indices_position_after_chain_end[0],
                    line_indices_position_up_to_terminal_atom[0],
                ):

                    Z_matrix[
                        k_ac
                        + (
                            line_indices_position_up_to_terminal_atom[
                                len(str_at_type) - 1
                            ]
                            - line_indices_position_up_to_terminal_atom[0]
                        )
                    ][5] = ordered_atoms_new_indices[
                        line_indices_position_after_chain_end[0]
                    ]

    return Z_matrix


# In[ ]:


def Z_matrix_Fillers(nr_at, bond_lengths_ordered, Z_matrix, New_at_coord):

    # Bond lengths
    for bnd in range(
        1, nr_at
    ):  # start at 1 there are N - 1  bonds in a Z_matrix where N = no of atoms

        Z_matrix[bnd][2] = round(
            bond_lengths_ordered[(bnd - 1)], 4
        )  # bond_lengths_ordered stores all of the computed bond length in order they appear in the Z matrix

    # Bond angles
    Ung = (
        []
    )  # append the atoms and their labels on each line of the Z matrix as strings
    for i in range(2, nr_at):
        Ung.append(Z_matrix[i][0])  # Atom_1
        Ung.append(Z_matrix[i][1])  # Atom_2 bonded to Atom_1
        Ung.append(Z_matrix[i][3])  # Atom_3 bonded to Atom_2

    Z_imp_u = []
    for j in range(len(Ung)):

        Z_ut_u = [
            i for i in re.split(r"([A-Z][a-z]*)", Ung[j]) if i
        ]  # parse each atom string and retain its symbol and coefficient as separately appended to a list of length 2 eg: 'C1' becomes ['C', '1']

        Z_imp_u.append(
            int(Z_ut_u[1]) - 1
        )  # only the coefficient is required, thus we ectract the value at position one from each list and turn it back from string to integer. Substact one as they will be correlated to the atomic coordinates stored in "New_at_coord"

    U_A = (
        []
    )  # create list to store the atomic coordinates for the corresponding coefficients in order to compute the bond angle
    for i in range(len(Z_imp_u)):
        U_A.append(New_at_coord[Z_imp_u[i]])

    for i in range(nr_at):
        if i < (
            nr_at - 2
        ):  # -2 because there are N-2 bond angles in a Z_matrix where N = no of atoms

            Z_matrix[i + 2][4] = round(
                Bond_Angle(U_A[0 + (i * 3)], U_A[1 + (i * 3)], U_A[2 + (i * 3)]), 4
            )  # * 3 as every 3 atoms form a bond angle and the cycle repeats every 3 atoms

    # Dihedral angles
    S = []  # append the atoms and their labels on each line of the Z matrix as strings
    for i in range(3, nr_at):
        S.append(Z_matrix[i][0])
        S.append(Z_matrix[i][1])
        S.append(Z_matrix[i][3])
        S.append(Z_matrix[i][5])

    Z_imp = []
    for j in range(len(S)):

        Z_ut = [i for i in re.split(r"([A-Z][a-z]*)", S[j]) if i]
        Z_imp.append(int(Z_ut[1]) - 1)
    # Torsion angles
    T_A = []
    for i in range(len(Z_imp)):

        T_A.append(New_at_coord[Z_imp[i]])

    for i in range(nr_at):
        if i < (
            nr_at - 3
        ):  # -3 because there are N-3 torsion angles in a Z_matrix where N = no of atoms

            Z_matrix[i + 3][6] = round(
                Torsion_Angle(
                    T_A[0 + (i * 4)],
                    T_A[1 + (i * 4)],
                    T_A[2 + (i * 4)],
                    T_A[3 + (i * 4)],
                ),
                4,
            )
            # * 4 as every 4 atoms form a dihedral angle and the cycle repeats every 4 atoms

    return Z_matrix


# In[ ]:


def Automatical_Z_matrix(
    molecule_1, N
):  # Obstain the Z matrixes using the classes and functions above

    nr_atomi = NeighboursLists(molecule_1)
    nr_at = nr_atomi.nr_at
    at_type = nr_atomi.at_type
    list_atoms = NeighboursLists(molecule_1)
    central_atoms_list = list_atoms.Central_Listuta()
    other_atoms_list = list_atoms.Other_Listuta()
    properties_lists = GeometricPropertiesOtherAtoms(
        central_atoms_list, other_atoms_list
    )
    properties_lists_C = GeometricPropertiesCentralAtoms(central_atoms_list)
    at_coord_central = properties_lists_C.at_coord_central
    at_coord_other = properties_lists.at_coord_other
    str_at_type_central = properties_lists_C.str_at_type_central
    str_at_type = properties_lists.str_at_type
    assert len(str_at_type_central) == len(
        str_at_type
    )  # str_at_type contains nested lists, of lengths equal to the nuber of other atoms bonded to each
    # individual central atom, resulting the number of lists must equal the number of central
    # atoms, otherwise an errorr had str_at_type_centralcured
    New_at_coord = New_Coordinates(
        at_coord_central, at_coord_other, str_at_type_central, str_at_type
    )
    bond_lengths_chain = properties_lists_C.bond_lengths_chain
    bond_lengths_other = properties_lists.bond_lengths_other
    bond_lengths_ordered = Bond_Lengths(bond_lengths_chain, bond_lengths_other)

    if (
        N == 1
    ):  # N = 1 means initial molecule no polymer can also not be polymerizabond_lengths_orderede molecule

        ordered_atoms_new_indices = My_very_new_coordinates(
            str_at_type_central, str_at_type
        )
        Z_matrix_i = Z_MATRIX(
            str_at_type_central, str_at_type, ordered_atoms_new_indices, nr_at, N
        )
        Z_matrix = Z_matrix_Fillers(
            nr_at, bond_lengths_ordered, Z_matrix_i, New_at_coord
        )
        return Z_matrix
    else:
        str_at_type_central_polymer = str_at_type_central  # str_at_type_centralv will form new list of C atoms corresponding to the polymer
        str_at_type_polymer = str_at_type  # str_at_type_polymer will form new list of other attoms corresponding to the polymer
        for i in range(N - 1):
            str_at_type_central_polymer = (
                str_at_type_central_polymer + str_at_type_central
            )  # adds monomer at every step
            str_at_type_polymer = str_at_type_polymer + str_at_type
        ordered_atoms_new_indices_p = My_very_new_coordinates(
            str_at_type_central_polymer, str_at_type_polymer
        )
        Z_polymer = Z_MATRIX(
            str_at_type_central_polymer,
            str_at_type_polymer,
            ordered_atoms_new_indices_p,
            nr_at,
            N,
        )
        return Z_polymer


Z_matrix_1 = Automatical_Z_matrix(
    molecule_1, 1
)  # Z matrix of the input molecule or by case of the addition polymer
print(f"\nHere is the {molecule_1} Z Matrix format  ")
print(f"\n{Z_matrix_1}")

# Extract the total number of atoms in the molecule
no_atoms_molecule = NeighboursLists(molecule_1)
central_atoms = no_atoms_molecule.Central_Listuta()
other_atoms = no_atoms_molecule.Other_Listuta()
total_atoms = len(central_atoms) + len(other_atoms)

# write a new txt file to record the Z matrix output


def output_matrix(N, output_file, Z_MATRIX, total_atoms):
    """Output the cartesian coordinates of the file"""
    with open(output_file, "w") as f:
        # the N*total_atoms range gives the number of lines in both the Z matrix polymer and Z matrix.
        # the N variable (no of monomers) is set to 1 by default for the monomer Z matrix
        for i in range(N * total_atoms):
            new_line = ""
            #
            for j in range(7):
                new_line = new_line + str(Z_MATRIX[i][j]) + "\t" + "\t"

            f.write(
                f"{new_line}\n"
            )  # dont know how to separate the lines should be 4 on each


new_file_Z_matrix = output_matrix(
    1, f"{molecule_1}_Z_matrix.txt", Z_matrix_1, total_atoms
)
# output of the initial molecule Z matrix


# In[ ]:


"""POLYMERIZER !!! currently for C2H3Cl and C2H4 only but might extend to bigger list"""
add_your_own = input(
    'Would you like to intoduce any other molecule apart from those provided in the instructions ? Answer with "yes" or "no" : '
)
if add_your_own == "yes":

    your_molecule = molecule_1  # given by molecule_1 as the initially indroduced molecule is assumed to be the onethat should undergo addition polymerisation

    if total_atoms == 6:

        polymerizable_molecules_type1 = [your_molecule]
    else:
        raise ValueError(
            "The programme is not yet capable of polymerising a molecule of this complexity"
        )
else:

    polymerizable_molecules_type1 = ["C2H3Cl", "C2H4"]


if molecule_1 in polymerizable_molecules_type1:

    """START FILLING THE Z MATRIX OF THE POLYMER THE RULE IS PRESENTED IN THE PICTURE"""

    N = int(input("Enter the number of monomers you wish to add: "))
    # Condition preventing the user to intoduce more than the computational limit of 15 monomers
    if N > 15:
        N = 15
    Z_polymer = Automatical_Z_matrix(
        molecule_1, N
    )  # empty (no geometry) polymer Z_matrix
    saturated_molecule_equiv = input("Enter a saturated equivalent of the monomer: ")
    Z_monomer = Automatical_Z_matrix(saturated_molecule_equiv, 1)
    # call the Z matrix of the SATURATED equivalent of the monomer. eg: for C2H3Cl introduce C2H5Cl

    for i in range(1, 2 * N):  # here 2*N comes from 2 central atoms

        Z_polymer[i][2] = 1.5201  # set the bond lengths between successive C atoms

    for i in range(2, 2 * N):
        Z_polymer[i][4] = 111.4691  # set bond angle formed between 3 C atoms

    for i in range(3, (2 * N - 1)):
        Z_polymer[i][6] = -180  # set dihedral between 4 C atoms
    Z_polymer[(2 * N - 1)][
        6
    ] = 180  # last one is always positive. Concluded after repeated measurments

    """Fill internal coordinates for the H---C--C---H type of entries. Hallogens fit the H description in this context
    several for loops are required becuase they do not follow the same trend. This is further explained in figure from section..
    REMINDER! Positions 2, 4, 6 within any line of a Z matrix corespond to the  bond lengths, bond angles and
    dihedrals, stored as float type variables"""

    # Fills in the 4N-2 lines Z matrix block that systematically repeats the intenal properties as explained in section...
    # The properties repeat in steps of 4 since there is a change in dihedral angle between any two consecutive lines
    # This operates under the assumption that only 6 atoms molecules with two central atoms are considered
    # Hence 2*N is equivalent with len(central_atoms)*N and 6*N to len(total_atoms)*N
    for i in range((2 * N + 2), (6 * N - 4), 4):
        Z_polymer[i][2] = Z_matrix_1[4][2]
        Z_polymer[i][4] = Z_matrix_1[4][4]
        Z_polymer[i][6] = Z_matrix_1[4][6]
    for i in range((2 * N + 3), (6 * N - 4), 4):
        Z_polymer[i][2] = Z_matrix_1[5][2]
        Z_polymer[i][4] = Z_matrix_1[5][4]
        Z_polymer[i][6] = Z_matrix_1[5][6]
    for i in range((2 * N + 4), (6 * N - 4), 4):
        Z_polymer[i][2] = Z_monomer[6][2]
        Z_polymer[i][4] = Z_monomer[6][4]
        Z_polymer[i][6] = Z_monomer[6][6]
    for i in range((2 * N + 5), (6 * N - 4), 4):
        Z_polymer[i][2] = Z_monomer[7][2]
        Z_polymer[i][4] = Z_monomer[7][4]
        Z_polymer[i][6] = Z_monomer[7][6]

    # Fills in the first 2 lines after the end of the central atoms chain and the last four lines of the Z polymer matrix
    # Under the assumption that only 6 atoms molecules with two central atoms are considered this are
    # the 2N and 2N+1 positions corresponding to the atoms labeled 2N+1 and 2N+2
    # Given the dihedral angle switches its sign or value between two consecutive lines of the Z matrix the same method as above applies

    # Fills in the first line after the end of the central atoms chain
    for i in range(2 * N, (2 * N + 2), 2):
        Z_polymer[i][2] = Z_monomer[3][2]
        Z_polymer[i][4] = Z_monomer[3][4]
        Z_polymer[i][6] = Z_monomer[3][6]

        # Fills in two of the last 4 lines of the polymer Z matrix
        Z_polymer[i + (4 * N - 4)][2] = Z_monomer[3][2]
        Z_polymer[i + (4 * N - 4)][4] = Z_monomer[3][4]
        Z_polymer[i + (4 * N - 4)][6] = Z_monomer[3][6]

        Z_polymer[i + (4 * N - 2)][2] = Z_monomer[3][2]
        Z_polymer[i + (4 * N - 2)][4] = Z_monomer[3][4]
        Z_polymer[i + (4 * N - 2)][6] = Z_monomer[3][6]

    # Fills in the second line after the end of the central atoms chain
    for i in range((2 * N + 1), (2 * N + 2), 2):
        Z_polymer[i][2] = Z_monomer[4][2]
        Z_polymer[i][4] = Z_monomer[4][4]
        Z_polymer[i][6] = Z_monomer[4][6]

        # Fills in two of the last 4 lines of the polymer Z matrix
        Z_polymer[i + (4 * N - 4)][2] = Z_monomer[4][2]
        Z_polymer[i + (4 * N - 4)][4] = Z_monomer[4][4]
        Z_polymer[i + (4 * N - 4)][6] = Z_monomer[4][6]

        Z_polymer[i + (4 * N - 2)][2] = Z_monomer[4][2]
        Z_polymer[i + (4 * N - 2)][4] = Z_monomer[4][4]
        Z_polymer[i + (4 * N - 2)][6] = Z_monomer[4][6]

    print("\nHere is the Z Matrix format for the POLYMER ")
    print(f"\n{Z_polymer}")
    new_file_Z_polymer = output_matrix(
        N, f"{molecule_1}_Z_matrix_polymer.txt", Z_polymer, total_atoms
    )
    # output of Z matrix polymer stored as txt file


# In[ ]:


# POLYMER CLASS REQUIRED GEOMETRY
class New_uv_n_ABC:  # new class that returns the normalised vectors mk and mk-1
    def __init__(
        self, at_C1, at_C2, at_C3
    ):  #  function of the SRF 3 initial coordinates C1, C2, C3
        self.at_C1 = at_C1
        self.at_C2 = at_C2
        self.at_C3 = at_C3

        self.r_C2C3 = PositionVectors(self.at_C2, self.at_C3)
        self.r_C1C2 = PositionVectors(self.at_C1, self.at_C2)

        self.r_C2C3_n = self.r_C2C3.unit_vector  # mk normalised
        self.r_C1C2_n = self.r_C1C2.unit_vector  # mk-1 normalised

    def Plane_ABC(self):
        u_ABC = VectorProducts(self.r_C1C2_n, self.r_C2C3_n)

        n_ABC = (
            u_ABC.Cross_product()
        )  # normal plane between the 3 atoms with known coordinates or nk normalised from above

        return n_ABC

    def column_y(
        self,
    ):  # nk x mk cross product between the normalised vector plane C1C2C3 and normal vector C2C3
        n_norm = self.Plane_ABC()
        n_bc = VectorProducts(n_norm, self.r_C2C3_n)
        n_bc_cartezian = n_bc.Cross_product()
        return n_bc_cartezian

    def M_rot_like_matrix(
        self,
    ):  # the fuction mapping the first 3 coordinates to a rotation matrix

        n_x_bc = self.column_y()  # second column nk x mk
        n_abc = self.Plane_ABC()  # third column nk
        # create the rotation matrix
        M = np.matrix(
            [
                [self.r_C2C3_n[0], n_x_bc[0], n_abc[0]],
                [self.r_C2C3_n[1], n_x_bc[1], n_abc[1]],
                [self.r_C2C3_n[2], n_x_bc[2], n_abc[2]],
            ]
        )
        return M


def Intitial_pozition(
    R, theta, phi
):  # initial pozition of the 4th added atom with the first 3 having known coordinates

    C4 = [
        R * np.cos(theta),
        R * np.cos(phi) * np.sin(theta),
        R * np.sin(phi) * np.sin(theta),
    ]

    C_4 = np.array(
        [C4]
    ).T  # fourth vector expressed as a column vector using polar coordinates transformation

    return C_4


# at_C4 is the output of the Initial_pozition function
def New_corrd_A(
    at_C1, at_C2, at_C3, at_C4, at_C3_real
):  # for the very first computation at_C3_real coincides with the SRF one

    Matrix = New_uv_n_ABC(at_C1, at_C2, at_C3)

    M = Matrix.M_rot_like_matrix()

    A = (
        M * at_C4 + np.array([at_C3_real]).T
    )  # translation by the real Ck-1 vector from SRF

    return A


# In[ ]:


# Apply the NeRF algoritm on the central atoms from Z polymer computed in Part 2

at_C1 = np.array([-0.37, 1.47, 0])
at_C2 = np.array([-1.52, 0, 0])
at_C3 = np.array([0, 0, 0])
at_C4 = Intitial_pozition(
    Z_polymer[3][2], Z_polymer[3][4], Z_polymer[3][6]
)  # R, theta and psi extracted from Z_polymer
at_C3_real = np.array(at_C3).T[0]
new_coord = []
# 2 = number of cental atoms in monomer
for i in range(2 * N):
    add_atom = New_corrd_A(
        at_C1, at_C2, at_C3, at_C4, at_C3_real
    )  # the first iteration computes add_atom in SRF
    # as the translation tp the real cartesian space is with the zero vector. This represents C1 !
    at_C3_real = np.array(add_atom).T[
        0
    ]  # at this step during each iteration at_C3_real takes the value of add_atom
    # which is the real value of C4 after translation/ outside SRF
    # no additional operation is required within the code as the function New_coord_A ensures the translation vectors
    # add at each step
    new_coord.append(
        np.array(add_atom).T[0]
    )  # change the returned column vector back to an array representation
# print(new_coord) # this are all of the backbone central atoms


# In[ ]:


# Storing the xyz cartesian output in a molecular visualiser compatible format
New_polymer = new_coord
new_polymer_cartesian = np.zeros((N * 2, 4), list)
# create matrix like skelet N*2 represents the no central atoms (2) multiplied by the no of monomers (N)
for i in range(len(New_polymer)):
    new_polymer_cartesian[i][1] = New_polymer[i][0]  # x axis coordinates

    new_polymer_cartesian[i][2] = New_polymer[i][1]  # y axis coordinates

    new_polymer_cartesian[i][3] = New_polymer[i][2]  # z axis coordinates

    new_polymer_cartesian[i][0] = Z_polymer[i][
        0
    ]  # Labeled atoms in order they were stored on the 1st column of Z polymer


"""write file of new cartezian coordinates"""


def output_cartesian(output_file):
    """Output the cartesian coordinates of the file"""
    with open(output_file, "w") as f:
        for i in range(N * 2):
            new_line = ""
            for j in range(4):
                new_line = new_line + str(new_polymer_cartesian[i][j]) + "\t"

            f.write(f"{new_line}\n")


new_file = output_cartesian(f"{molecule_1}_polymer_backbone_cartesian.txt")


# In[ ]:


# Function of two variables, the number of monomers and the monomer length , that calculates the radius of gyration
def radius_gyration(number_monomeric_units, monomer_unit_length):
    
    R_gyr = (np.sqrt(number_monomeric_units/6))*monomer_unit_length # Z_polymer[3][2] = length of one monomer unit 
    return R_gyr

# assertation that accounts for the reliability of the function outcome problem provided in ref.6
assert round(radius_gyration(200, 450),3) == 2598.076

number_monomeric_units = int(input("Enter the polymerisation degree:  "))

choice = input('Would you like to asess the same molecule as in parts 1-3. Please enter "yes"')
if choice == "yes":
               
    monomer_unit_length = Z_polymer[3][2]
               
else:
    monomer_unit_length = float(input('Enter the monomer length of your molecule in Å: '))
               
if monomer_unit_length < 0.74 or monomer_unit_length > 1.54:
               print("Bond length introduced is not valid")
else:
               
    R_gyr = radius_gyration(number_monomeric_units, monomer_unit_length)

    print (f'\nRgyr = {R_gyr } x 10^-10 m') # carefull all calclations are performed for a in Angstroms

