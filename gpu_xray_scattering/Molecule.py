
import numpy as np
from scipy import ndimage, interpolate, spatial, special, optimize, signal, stats, fft


from .resources import resources
electrons = resources.electrons
atomic_volumes = resources.atomic_volumes
numH = resources.numH
volH = resources.volH
vdW = resources.vdW
radii_sf_dict = resources.radii_sf_dict


class Molecule:
    def __init__(self, coordinates=None, elements=None, radius=None, replace_unknown_elements_with='C'):
        self.replace_unknown_elements_with = replace_unknown_elements_with
        self.set_coordinates(coordinates)
        self.set_elements(elements)
        self.set_radius(radius)

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.num_atoms = len(coordinates)

    def set_elements(self, elements):
        self.elements = elements
        self.electrons = None
        if self.elements is not None:
            self.electrons = np.array([electrons.get(x.upper(), electrons[self.replace_unknown_elements_with]) for x in self.elements])

    def set_radius(self, radius):
        self.radius = None
        if radius is not None:
            assert len(radius) == self.num_atoms
            self.radius = radius
        else:
            self.radius = np.zeros(self.num_atoms)
            
 
class PDB(object):
    """Load pdb file."""
    def __init__(self, filename=None, natoms=None, ignore_waters=True):
        if isinstance(filename, int):
            #if a user gives no keyword argument, but just an integer,
            #assume the user means the argument is to be interpreted
            #as natoms, rather than filename
            natoms = filename
            filename = None
        if filename is not None:
            self.read_pdb(filename, ignore_waters=ignore_waters)
        elif natoms is not None:
            self.generate_pdb_from_defaults(natoms)
        self.rij = None
        self.radius = None
        self.unique_radius = None
        self.unique_volume = None
        self.modifiable_atom_types = ['H','C','N','O']
        self.radii_sf = np.ones(len(self.modifiable_atom_types))
        for i in range(len(self.modifiable_atom_types)):
            if self.modifiable_atom_types[i] in radii_sf_dict.keys():
                self.radii_sf[i] = radii_sf_dict[self.modifiable_atom_types[i]]
            else:
                self.radii_sf[i] = 1.0

    def read_pdb(self, filename, ignore_waters=True):
        self.natoms = 0
        with open(filename) as f:
            for line in f:
                if line[0:6] == "ENDMDL":
                    break
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                if ignore_waters and ((line[17:20]=="HOH") or (line[17:20]=="TIP")):
                    continue
                self.natoms += 1
        self.atomnum = np.zeros((self.natoms),dtype=int)
        self.atomname = np.zeros((self.natoms),dtype=np.dtype((str,3)))
        self.atomalt = np.zeros((self.natoms),dtype=np.dtype((str,1)))
        self.resname = np.zeros((self.natoms),dtype=np.dtype((str,3)))
        self.resnum = np.zeros((self.natoms),dtype=int)
        self.chain = np.zeros((self.natoms),dtype=np.dtype((str,1)))
        self.coords = np.zeros((self.natoms, 3))
        self.occupancy = np.zeros((self.natoms))
        self.b = np.zeros((self.natoms))
        self.atomtype = np.zeros((self.natoms),dtype=np.dtype((str,2)))
        self.charge = np.zeros((self.natoms),dtype=np.dtype((str,2)))
        self.nelectrons = np.zeros((self.natoms),dtype=int)
        self.vdW = np.zeros(self.natoms)
        self.numH = np.zeros(self.natoms)
        self.unique_exvolHradius = np.zeros(self.natoms)
        self.exvolHradius = np.zeros(self.natoms)
        with open(filename) as f:
            atom = 0
            for line in f:
                if line[0:6] == "ENDMDL":
                    break
                if line[0:6] == "CRYST1":
                    cryst = line.split()
                    self.cella = float(cryst[1])
                    self.cellb = float(cryst[2])
                    self.cellc = float(cryst[3])
                    self.cellalpha = float(cryst[4])
                    self.cellbeta = float(cryst[5])
                    self.cellgamma = float(cryst[6])
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                if ignore_waters and ((line[17:20]=="HOH") or (line[17:20]=="TIP")):
                    continue
                try:
                    self.atomnum[atom] = int(line[6:11])
                except ValueError as e:
                    self.atomnum[atom] = int(line[6:11],36)
                self.atomname[atom] = line[12:16].split()[0]
                self.atomalt[atom] = line[16]
                self.resname[atom] = line[17:20]
                try:
                    self.resnum[atom] = int(line[22:26])
                except ValueError as e:
                    self.resnum[atom] = int(line[22:26],36)
                self.chain[atom] = line[21]
                self.coords[atom, 0] = float(line[30:38])
                self.coords[atom, 1] = float(line[38:46])
                self.coords[atom, 2] = float(line[46:54])
                self.occupancy[atom] = float(line[54:60])
                self.b[atom] = float(line[60:66])
                atomtype = line[76:78].strip()
                if len(atomtype) == 2:
                    atomtype0 = atomtype[0].upper()
                    atomtype1 = atomtype[1].lower()
                    atomtype = atomtype0 + atomtype1
                if len(atomtype) == 0:
                    #if atomtype column is not in pdb file, set to first
                    #character of atomname
                    atomtype = self.atomname[atom][0]
                self.atomtype[atom] = atomtype
                self.charge[atom] = line[78:80].strip('\n')
                self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(),6)
                if len(self.atomtype[atom])==1:
                    atomtype = self.atomtype[atom][0].upper()
                else:
                    atomtype = self.atomtype[atom][0].upper() + self.atomtype[atom][1].lower()
                try:
                    dr = vdW[atomtype]
                except:
                    try:
                        dr = vdW[atomtype[0]]
                    except:
                        #default to carbon
                        dr = vdW['C']
                self.vdW[atom] = dr
                atom += 1

    def generate_pdb_from_defaults(self, natoms):
        self.natoms = natoms
        #simple array of incrementing integers, starting from 1
        self.atomnum = np.arange((self.natoms),dtype=int)+1
        #all carbon atoms by default
        self.atomname = np.full((self.natoms),"C",dtype=np.dtype((str,3)))
        #no alternate conformations by default
        self.atomalt = np.zeros((self.natoms),dtype=np.dtype((str,1)))
        #all Alanines by default
        self.resname = np.full((self.natoms),"ALA",dtype=np.dtype((str,3)))
        #each atom belongs to a new residue by default
        self.resnum = np.arange((self.natoms),dtype=int)
        #chain A by default
        self.chain = np.full((self.natoms),"A",dtype=np.dtype((str,1)))
        #all atoms at (0,0,0) by default
        self.coords = np.zeros((self.natoms, 3))
        #all atoms 1.0 occupancy by default
        self.occupancy = np.ones((self.natoms))
        #all atoms 20 A^2 by default
        self.b = np.ones((self.natoms))*20.0
        #all atom types carbon by default
        self.atomtype = np.full((self.natoms),"C",dtype=np.dtype((str,2)))
        #all atoms neutral by default
        self.charge = np.zeros((self.natoms),dtype=np.dtype((str,2)))
        #all atoms carbon so have six electrons by default
        self.nelectrons = np.ones((self.natoms),dtype=int)*6
        self.radius = np.zeros(self.natoms)
        self.vdW = np.zeros(self.natoms)
        self.unique_volume = np.zeros(self.natoms)
        self.unique_radius = np.zeros(self.natoms)
        #set a variable with H radius to be used for exvol radii optimization
        #set a variable for number of hydrogens bonded to atoms
        # self.exvolHradius = implicit_H_radius
        self.unique_exvolHradius = np.zeros(self.natoms)
        self.implicitH = False
        self.numH = np.zeros((self.natoms))
        #for CRYST1 card, use default defined by PDB, but 100 A side
        self.cella = 100.0
        self.cellb = 100.0
        self.cellc = 100.0
        self.cellalpha = 90.0
        self.cellbeta = 90.0
        self.cellgamma = 90.0

    def calculate_unique_volume(self,n=16,use_b=False,atomidx=None):
        """Generate volumes and radii for each atom of a pdb by accounting for overlapping sphere volumes,
        i.e., each radius is set to the value that yields a volume of a sphere equal to the
        corrected volume of the sphere after subtracting spherical caps from bonded atoms."""
        #first, for each atom, find all atoms closer than the sum of the two vdW radii
        ns = np.array([8,16,32])
        corrections = np.array([1.53,1.19,1.06]) #correction for n=8 voxels (1.19 for n=16, 1.06 for n=32)
        correction = np.interp(n,ns,corrections) #a rough approximation.
        # print("Calculating unique atomic volumes...")
        if self.unique_volume is None:
            self.unique_volume = np.zeros(self.natoms)
        if atomidx is None:
            atomidx = range(self.natoms)
        for i in atomidx:
            # sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,self.natoms))
            # sys.stdout.flush()
            #for each atom, make a box of voxels around it
            ra = self.vdW[i] #ra is the radius of the main atom
            if use_b:
                ra += B2u(self.b[i])
            side = 2*ra
            #n = 8 #yields somewhere around 0.2 A voxel spacing depending on atom size
            dx = side/n
            dV = dx**3
            x_ = np.linspace(-side/2,side/2,n)
            x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
            minigrid = np.zeros(x.shape,dtype=np.bool_)
            shift = np.ones(3)*dx/2.
            #create a column stack of coordinates for the minigrid
            xyz = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
            #for simplicity assume the atom is at the center of the minigrid, (0,0,0),
            #therefore we need to subtract the vector shift (i.e. the coordinates
            #of the atom) from each of the neighboring atoms, so grab those coordinates
            p = np.copy(self.coords[i])
            #calculate all distances from the atom to the minigrid points
            center = np.zeros(3)
            xa, ya, za = center
            dist = spatial.distance.cdist(center[None,:], xyz)[0].reshape(n,n,n)
            #now, any elements of minigrid that have a dist less than ra make true
            minigrid[dist<=ra] = True
            #grab atoms nearby this atom just based on xyz coordinates
            #first, recenter all coordinates in this frame
            coordstmp = self.coords - p
            #next, get all atoms whose x, y, and z coordinates are within the nearby box 
            #of length 4 A (more than the sum of two atoms vdW radii, with the limit being about 2.5 A)
            bl = 5.0
            idx_close = np.where(
                (coordstmp[:,0]>=xa-bl/2)&(coordstmp[:,0]<=xa+bl/2)&
                (coordstmp[:,1]>=ya-bl/2)&(coordstmp[:,1]<=ya+bl/2)&
                (coordstmp[:,2]>=za-bl/2)&(coordstmp[:,2]<=za+bl/2)
                )[0]
            idx_close=idx_close[idx_close!=i] #ignore this atom
            nclose = len(idx_close)
            for j in range(nclose):
                #get index of next closest atom
                idx_j = idx_close[j]
                #get the coordinates of the  neighboring atom, and shift using the same vector p as the main atom
                cb = self.coords[idx_j] - p #center of neighboring atom in new coordinate frame
                xb,yb,zb = cb
                rb = self.vdW[idx_j]
                if use_b:
                    rb += B2u(self.b[idx_j])
                a,b,c,d = equation_of_plane_from_sphere_intersection(xa,ya,za,ra,xb,yb,zb,rb)
                normal = np.array([a,b,c]) #definition of normal to a plane
                #for each grid point, calculate the distance to the plane in the direction of the vector normal
                #if the distance is positive, then that gridpoint is beyond the plane
                #we can calculate the center of the circle which lies on the plane, so thats a good point to use
                circle_center = center_of_circle_from_sphere_intersection(xa,ya,za,ra,xb,yb,zb,rb,a,b,c,d)
                xyz_minus_cc = xyz - circle_center
                #calculate distance matrix to neighbor
                dist2neighbor = spatial.distance.cdist(cb[None,:], xyz)[0].reshape(n,n,n)
                overlapping_voxels = np.zeros(n**3,dtype=bool)
                overlapping_voxels[minigrid.ravel() & np.ravel(dist2neighbor<=rb)] = True
                #calculate the distance to the plane for each minigrid voxel
                #there may be a way to vectorize this if its too slow
                noverlap = overlapping_voxels.sum()
                # print(noverlap, overlapping_voxels.size)
                d2plane = np.zeros(x.size)
                for k in range(n**3):
                    if overlapping_voxels[k]:
                        d2plane[k] = np.dot(normal,xyz_minus_cc[k,:])
                d2plane = d2plane.reshape(n,n,n)
                #all voxels with a positive d2plane value are _beyond_ the plane
                minigrid[d2plane>0] = False
            #add up all the remaining voxels in the minigrid to get the volume
            #also correct for limited voxel size
            self.unique_volume[i] = minigrid.sum()*dV * correction

    def lookup_unique_volume(self):
        self.unique_volume = np.zeros(self.natoms)
        for i in range(self.natoms):
            notfound = False
            if (self.resname[i] in atomic_volumes.keys()):
                if (self.atomname[i] in atomic_volumes[self.resname[i]].keys()):
                    self.unique_volume[i] = atomic_volumes[self.resname[i]][self.atomname[i]]
                else:
                    notfound = True
            else:
                notfound = True
            if notfound:
                print("%s:%s not found in volumes dictionary. Calculating unique volume."%(self.resname[i],self.atomname[i]))
                # print("Setting volume to ALA:CA.")
                # self.unique_volume[i] = atomic_volumes['ALA']['CA']
                self.calculate_unique_volume(atomidx=[i])

    def add_ImplicitH(self):
        if 'H' in self.atomtype:
            self.remove_by_atomtype('H')

        for i in range(len(self.atomname)):
            res = self.resname[i]
            atom = self.atomname[i]

            #For each atom, atom should be a key in "numH", so now just look up value 
            # associated with atom
            try:
                H_count = np.rint(numH[res][atom]) #the number of H attached
                # print(res, atom, numH[res][atom])
                # Hbond_count = protein_residues.normal[res]['numH']
                # H_count = Hbond_count[atom]
                H_mean_volume = volH[res][atom] #the average volume of each H attached
            except:
                # print("atom ", atom, " not in ", res, " list. setting numH to 0.")
                H_count = 0
                H_mean_volume = 0

            #Add number of hydrogens for the atom to a pdb object so it can
            #be carried with pdb class
            self.numH[i] = H_count #the number of H attached
            self.unique_exvolHradius[i] = sphere_radius_from_volume(H_mean_volume)
            self.nelectrons[i] += H_count

    def remove_waters(self):
        idx = np.where((self.resname=="HOH") | (self.resname=="TIP"))
        self.remove_atoms_from_object(idx)

    def remove_by_atomtype(self, atomtype):
        idx = np.where((self.atomtype==atomtype))
        self.remove_atoms_from_object(idx)

    def remove_by_atomname(self, atomname):
        idx = np.where((self.atomname==atomname))
        self.remove_atoms_from_object(idx)

    def remove_by_atomnum(self, atomnum):
        idx = np.where((self.atomnum==atomnum))
        self.remove_atoms_from_object(idx)

    def remove_by_resname(self, resname):
        idx = np.where((self.resname==resname))
        self.remove_atoms_from_object(idx)

    def remove_by_resnum(self, resnum):
        idx = np.where((self.resnum==resnum))
        self.remove_atoms_from_object(idx)

    def remove_by_chain(self, chain):
        idx = np.where((self.chain==chain))
        self.remove_atoms_from_object(idx)

    def remove_atomalt(self):
        idx = np.where((self.atomalt!=' ') & (self.atomalt!='A'))
        self.remove_atoms_from_object(idx)

    def remove_atoms_from_object(self, idx):
        mask = np.ones(self.natoms, dtype=bool)
        mask[idx] = False
        self.atomnum = self.atomnum[mask]
        self.atomname = self.atomname[mask]
        self.atomalt = self.atomalt[mask]
        self.resname = self.resname[mask]
        self.resnum = self.resnum[mask]
        self.chain = self.chain[mask]
        self.coords = self.coords[mask]
        self.occupancy = self.occupancy[mask]
        self.b = self.b[mask]
        self.atomtype = self.atomtype[mask]
        self.charge = self.charge[mask]
        self.nelectrons = self.nelectrons[mask]
        self.natoms = len(self.atomnum)
        if self.radius is not None:
            self.radius = self.radius[mask]
        self.vdW = self.vdW[mask]
        self.numH = self.numH[mask]
        if self.unique_radius is not None:
            self.unique_radius = self.unique_radius[mask]
        if self.unique_volume is not None:
            self.unique_volume = self.unique_volume[mask]
        if self.unique_exvolHradius is not None:
            self.unique_exvolHradius = self.unique_exvolHradius[mask]

    def write(self, filename):
        """Write PDB file format using pdb object as input."""
        records = []
        anum,rc = (np.unique(self.atomnum,return_counts=True))
        if np.any(rc>1):
            #in case default atom numbers are repeated, just renumber them
            self_numbering=True
        else:
            self_numbering=False
        for i in range(self.natoms):
            if self_numbering:
                atomnum = '%5i' % ((i+1)%99999)
            else:
                atomnum = '%5i' % (self.atomnum[i]%99999)
            atomname = '%3s' % self.atomname[i]
            atomalt = '%1s' % self.atomalt[i]
            resnum = '%4i' % (self.resnum[i]%9999)
            resname = '%3s' % self.resname[i]
            chain = '%1s' % self.chain[i]
            x = '%8.3f' % self.coords[i,0]
            y = '%8.3f' % self.coords[i,1]
            z = '%8.3f' % self.coords[i,2]
            o = '% 6.2f' % self.occupancy[i]
            b = '%6.2f' % self.b[i]
            atomtype = '%2s' % self.atomtype[i]
            charge = '%2s' % self.charge[i]
            records.append(['ATOM  ' + atomnum + '  ' + atomname + ' ' + resname + ' ' + chain + resnum + '    ' + x + y + z + o + b + '          ' + atomtype + charge])
        np.savetxt(filename, records, fmt='%80s'.encode('ascii'))
    
    def scale_radii(self, radii_sf=None):
        """Scale all the modifiable atom type radii in the pdb"""
        if radii_sf is None:
            radii_sf = self.radii_sf
        if self.radius is None:
            self.radius = np.zeros(self.natoms)
        for i in range(len(self.modifiable_atom_types)):
            #if not self.explicitH:
            #    if self.modifiable_atom_types[i]=='H':
            #        self.exvolHradius = radii_sf[i] * self.unique_exvolHradius 
            #    else:
            #        self.radius[self.pdb.atomtype==self.modifiable_atom_types[i]] = radii_sf[i] * self.unique_radius[self.pdb.atomtype==self.modifiable_atom_types[i]]
            #else:
                self.exvolHradius = np.zeros(self.natoms)
                self.radius[self.atomtype==self.modifiable_atom_types[i]] = radii_sf[i] * self.unique_radius[self.atomtype==self.modifiable_atom_types[i]]


def sphere_volume_from_radius(R):
    V_sphere = 4*np.pi/3 * R**3
    return V_sphere

def sphere_radius_from_volume(V):
    R_sphere = (3*V/(4*np.pi))**(1./3)
    return R_sphere

def cap_heights(r1,r2,d):
    """Calculate the heights h1, h2 of spherical caps from overlapping spheres of radii r1, r2 a distance d apart"""
    h1 = (r2-r1+d)*(r2+r1-d)/(2*d)
    h2 = (r1-r2+d)*(r1+r2-d)/(2*d)
    return h1, h2

def spherical_cap_volume(R,h):
    #sphere of radius R, cap of height h
    V_cap = 1./3 * np.pi * h**2 * (3*R-h)
    return V_cap

def equation_of_plane_from_sphere_intersection(x1,y1,z1,r1,x2,y2,z2,r2):
    """Calculate coefficients a,b,c,d of equation of a plane (ax+by+cz+d=0) formed by the
    intersection of two spheres with centers (x1,y1,z1), (x2,y2,z2) and radii r1,r2.
    from: http://ambrnet.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm"""
    a = 2*(x2-x1)
    b = 2*(y2-y1)
    c = 2*(z2-z1)
    d = x1**2 - x2**2 + y1**2 - y2**2 + z1**2 - z2**2 - r1**2 + r2**2
    return a,b,c,d

def center_of_circle_from_sphere_intersection(x1,y1,z1,r1,x2,y2,z2,r2,a,b,c,d):
    """Calculate the center of the circle formed by the intersection of two spheres"""
    # print(a*(x1-x2), b*(y1-y2), c*(z1-z2))
    # print((a*(x1-x2) + b*(y1-y2) +c*(z1-z2)))
    # print((x1*a + y1*b + z1*c + d))
    t = (x1*a + y1*b + z1*c + d) / (a*(x1-x2) + b*(y1-y2) +c*(z1-z2))
    xc = x1 + t*(x2-x1)
    yc = y1 + t*(y2-y1)
    zc = z1 + t*(z2-z1)
    return (xc,yc,zc)

def calc_rho0(mw, conc):
    """Estimate bulk solvent density, rho0, from list of molecular weights
    and molar concentrations of components.
    mw and conc can be lists (nth element of mw corresponds to nth element of concentration)
    mw in g/mol
    concentration in mol/L.
    """
    mw = np.atleast_1d(mw)
    conc = np.atleast_1d(conc)
    return 0.334 * (1 + np.sum(mw*conc*0.001))

def rotate_coordinates(coordinates, degrees_x=0, degrees_y=0, degrees_z=0):
    # Convert degrees to radians
    radians_x = np.deg2rad(degrees_x)
    radians_y = np.deg2rad(degrees_y)
    radians_z = np.deg2rad(degrees_z)

    # Create rotation object for each axis
    rotation_x = spatial.transform.Rotation.from_euler('x', radians_x)
    rotation_y = spatial.transform.Rotation.from_euler('y', radians_y)
    rotation_z = spatial.transform.Rotation.from_euler('z', radians_z)

    # Apply rotations sequentially
    rotated_coordinates = rotation_z.apply(rotation_y.apply(rotation_x.apply(coordinates)))

    return rotated_coordinates

