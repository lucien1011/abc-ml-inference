import os
import tensorflow as tf

from PyLooper.Common.Collector import Collector
from PyLooper.MLTools.Training import Training
from PyLooper.MLTools.DataWrapper import ROOTWrapper 
from PyLooper.Site import Site

from cms.HIG19007.MDNModule import MDNModule
from cms.HIG19007.InputModule import InputModule

# ______________________________________________________________________ ||
verbose         = True
nepoch          = 150
batch_per_tree  = 512
mass_points     = [1,2,3,4,7,10,15,20,25,30,35,]
bins            = [float(i)*35./100. for i in range(100+1)]
branches        = [
                "mass4l",
                "massZ1",
                "massZ2",
                "pTL1",
                "pTL2",
                "pTL3",
                "pTL4",
                "etaL1",
                "etaL2",
                "etaL3",
                "etaL4",
                "phiL1",
                "phiL2",
                "phiL3",
                "phiL4",
                "idL1",
                "idL2",
                "idL3",
                "idL4",
                ]

# ______________________________________________________________________ ||
site = Site()
if site.where == site.ufhpc:
    input_dir = "/cmsuf/data/store/user/t2/users/klo/IHEPA/raid/Higgs/DarkZ-NTuple/20191201/SkimTree_DarkPhoton_Run2016Data_m4l70/"
elif site.where == site.laptop:
    input_dir = "/Users/lucien/CMS/NTuple/lucien/Higgs/DarkZ-NTuple/20191201/SkimTree_DarkPhoton_Run2016Data_m4l70/"

# ______________________________________________________________________ ||
training_list = [
        Training("Training_m"+str(m1),ROOTWrapper({m2: os.path.join(input_dir,"HToZZdTo4L_M125_MZd%s_eps1e-2_13TeV_madgraph_pythia8.root"%str(m2)) for m2 in mass_points if m1 != m2 },"passedEvents"),)
        for m1 in [4,7,10,15,20,25,30]
        ]

collector = Collector(
        output_path = "./output/2020-11-11_01/",
        )

modules = [
        InputModule("InputModule",scale=35.),
        MDNModule(11,8,1),
        ]
