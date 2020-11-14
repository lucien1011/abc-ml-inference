import os,uproot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PyLooper.Utils.mkdir_p import mkdir_p

m               = 4
input_file_path = "/Users/lucien/CMS/NTuple/lucien/Higgs/DarkZ-NTuple/20191201/SkimTree_DarkPhoton_Run2016Data_m4l70/HToZZdTo4L_M125_MZd"+str(m)+"_eps1e-2_13TeV_madgraph_pythia8.root"
input_tree_path = "passedEvents"
model_path      = "output/2020-11-11_01/Training_m"+str(m)
plot_dir        = os.path.join(model_path,"plot/")
bins            = [float(i)*35./100. for i in range(100+1)]
plot_bins       = [i/100. for i in range(100+1)]
scale           = 35.
plot_png        = "plot.png"
branches        = [
                "mass4l",
                "massZ1",
                "massZ2",
                "genWeight",
                "passedFullSelection",
                "passedZXCRSelection",
                "dataMCWeight",
                "pileupWeight",
                "k_qqZZ_qcd_M",
                "k_qqZZ_ewk",
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

f = uproot.open(input_file_path)
t = f[input_tree_path]
model = tf.keras.models.load_model(model_path)

df = t.pandas.df(branches=branches)
x = np.reshape(df.massZ2.to_numpy()/scale,(1,df.shape[0]))
inputs = np.apply_along_axis(lambda x: np.histogram(x,bins=bins,density=1.)[0],1,x)
pred = model.predict(inputs)
ll = [model.calculate_loss(pred,tf.constant([[b]],dtype=np.float32))[0] for b in plot_bins]

mkdir_p(plot_dir)
plt.plot(bins,ll)
plt.grid()
plt.savefig(os.path.join(plot_dir,plot_png))
