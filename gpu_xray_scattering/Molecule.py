
import numpy as np

electrons = {'H': 1,'HE': 2,'He': 2,'LI': 3,'Li': 3,'BE': 4,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'NE': 10,'Ne': 10,'NA': 11,'Na': 11,'MG': 12,'Mg': 12,'AL': 13,'Al': 13,'SI': 14,'Si': 14,'P': 15,'S': 16,'CL': 17,'Cl': 17,'AR': 18,'Ar': 18,'K': 19,'CA': 20,'Ca': 20,'SC': 21,'Sc': 21,'TI': 22,'Ti': 22,'V': 23,'CR': 24,'Cr': 24,'MN': 25,'Mn': 25,'FE': 26,'Fe': 26,'CO': 27,'Co': 27,'NI': 28,'Ni': 28,'CU': 29,'Cu': 29,'ZN': 30,'Zn': 30,'GA': 31,'Ga': 31,'GE': 32,'Ge': 32,'AS': 33,'As': 33,'SE': 34,'Se': 34,'Se': 34,'Se': 34,'BR': 35,'Br': 35,'KR': 36,'Kr': 36,'RB': 37,'Rb': 37,'SR': 38,'Sr': 38,'Y': 39,'ZR': 40,'Zr': 40,'NB': 41,'Nb': 41,'MO': 42,'Mo': 42,'TC': 43,'Tc': 43,'RU': 44,'Ru': 44,'RH': 45,'Rh': 45,'PD': 46,'Pd': 46,'AG': 47,'Ag': 47,'CD': 48,'Cd': 48,'IN': 49,'In': 49,'SN': 50,'Sn': 50,'SB': 51,'Sb': 51,'TE': 52,'Te': 52,'I': 53,'XE': 54,'Xe': 54,'CS': 55,'Cs': 55,'BA': 56,'Ba': 56,'LA': 57,'La': 57,'CE': 58,'Ce': 58,'PR': 59,'Pr': 59,'ND': 60,'Nd': 60,'PM': 61,'Pm': 61,'SM': 62,'Sm': 62,'EU': 63,'Eu': 63,'GD': 64,'Gd': 64,'TB': 65,'Tb': 65,'DY': 66,'Dy': 66,'HO': 67,'Ho': 67,'ER': 68,'Er': 68,'TM': 69,'Tm': 69,'YB': 70,'Yb': 70,'LU': 71,'Lu': 71,'HF': 72,'Hf': 72,'TA': 73,'Ta': 73,'W': 74,'RE': 75,'Re': 75,'OS': 76,'Os': 76,'IR': 77,'Ir': 77,'PT': 78,'Pt': 78,'AU': 79,'Au': 79,'HG': 80,'Hg': 80,'TL': 81,'Tl': 81,'PB': 82,'Pb': 82,'BI': 83,'Bi': 83,'PO': 84,'Po': 84,'AT': 85,'At': 85,'RN': 86,'Rn': 86,'FR': 87,'Fr': 87,'RA': 88,'Ra': 88,'AC': 89,'Ac': 89,'TH': 90,'Th': 90,'PA': 91,'Pa': 91,'U': 92,'NP': 93,'Np': 93,'PU': 94,'Pu': 94,'AM': 95,'Am': 95,'CM': 96,'Cm': 96,'BK': 97,'Bk': 97,'CF': 98,'Cf': 98,'ES': 99,'Es': 99,'FM': 100,'Fm': 100,'MD': 101,'Md': 101,'NO': 102,'No': 102,'LR': 103,'Lr': 103,'RF': 104,'Rf': 104,'DB': 105,'Db': 105,'SG': 106,'Sg': 106,'BH': 107,'Bh': 107,'HS': 108,'Hs': 108,'MT': 109,'Mt': 109}

class Molecule:
    def __init__(self, coordinates=None, elements=None, replace_unknown_elements_with='C'):
        self.replace_unknown_elements_with = replace_unknown_elements_with
        self.set_coordinates(coordinates)
        self.set_elements(elements)

    def set_coordinates(self, coordinates):
        self.coordinates=coordinates

    def set_elements(self, elements):
        self.elements=elements
        self.electrons = None
        if self.elements is not None:
            self.electrons = np.array([electrons.get(x.upper(), electrons[self.replace_unknown_elements_with]) for x in self.elements])

 
