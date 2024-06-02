#!/usr/bin/env python
# coding: utf-8

# In[3]:

from __future__ import print_function, division
import numpy as np
#import pandas as pd
import os
import subprocess
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import ConvLSTM2D, Dense, Input, Flatten, Conv3DTranspose, Conv2DTranspose, TimeDistributed, MaxPooling3D, UpSampling3D, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import UpSampling2D, Conv2D
import pickle
#from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import io
import glob
import shutil
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore', 'info')

# In[]

seis=np.load(r"C:\Users\omars\Downloads\facies\facies.npy")
seis=np.expand_dims(seis, axis=-1)

X_train, X_test = train_test_split(seis, test_size=0.2, shuffle=True, random_state=21)



os.chdir(r"C:\Users\omars\OneDrive\Desktop\Master's Thesis")

# In[ ]:
#!/usr/bin/env python

from importlib import reload 
import h5py
from utils import *
import time





# In[]
# coding: utf-8


from IPython.display import clear_output


#from tensorflow import keras

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU


from tensorflow.keras.optimizers import Adam

import tensorflow_probability as tfp




import sys



# # CVAE Fully Conv 

# In[6]:

class CVAE(tf.keras.Model):
    def __init__(self):
        super(CVAE, self).__init__()
        # Input shape
        self.img_rows = 45
        self.img_cols = 45
        #self.img_depth = 4
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 200
        self.latent_dim_shape = (self.latent_dim, self.latent_dim, self.channels)

        self.optimizer = Adam(1e-4)

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def encode(self, x):
      mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=-1)
      return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        # Flatten z
        #z_flattened = tf.keras.layers.Flatten()(z)
        #logits = self.decoder(z_flattened)
        
        #logits = self.decoder(z)
        z_reshaped = tf.reshape(z, [-1, self.latent_dim])
        logits = self.decoder(z_reshaped)
        
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def reparameterize(self, mean, logvar):
      eps = tf.random.normal(shape=mean.shape)
      return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def sample(self, eps=None):
      if eps is None:
        eps = tf.random.normal(shape=(1, self.latent_dim))
      return self.decode(eps, apply_sigmoid=True)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
      log2pi = tf.math.log(2. * np.pi)
      return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
    
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
     
        # Ensure consistent data types
        x = tf.cast(x, tf.float32)
        x_logit = tf.cast(x_logit, tf.float32)
    
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
    
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


    def build_encoder(self):

        model = Sequential()
        model.add(Input(shape=self.img_shape))
        model.add(Conv2D(32, kernel_size=3, strides=2, activation="relu", padding="same")) # size 24
        model.add(Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")) # size 12
        model.add(Conv2D(128, kernel_size=3, strides=2, activation="relu", padding="same")) # size 6
        model.add(Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")) # size 6
        model.add(Flatten())
        model.add(Dense(self.latent_dim + self.latent_dim))
        model.summary()

        return model

    def build_decoder(self):

        model = Sequential()
        model.add(Input(shape=(self.latent_dim,)))
        model.add(Dense(11 * 11 * 64, activation="relu"))
        model.add(Reshape((11, 11, 64)))
        model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu"))  # size 16x16
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, ))  # size 32x32
        model.summary()
        
        return model


    @tf.function
    def train_step(self, x):
      with tf.GradientTape() as tape:
        loss = self.compute_loss(x)
      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def train(self, epochs, batch_size=128, save_interval=50):

        #X_train = seis
        
        noise = tf.random.normal(shape=[batch_size, self.latent_dim, self.latent_dim])
        self.loss_function = []
        self.total_training_time = 0 
        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            start_time = time.time()
            #for img in imgs:
            self.train_step(imgs)
            end_time = time.time()
            epoch_time = end_time - start_time
            self.total_training_time += epoch_time

            if epoch % save_interval == 0:
              loss = tf.keras.metrics.Mean()
              #for img in imgs:
              loss(self.compute_loss(X_train))
              elbo = -loss.result()
              self.loss_function.append(-elbo)
              clear_output(wait=True)
              print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                    .format(epoch, elbo, epoch_time))
              
              #self.save_imgs(epoch, X_train[np.random.randint(0, X_train.shape[0], 16)])
            
    def save_imgs(self, epoch, test_sample):
      mean, logvar = self.encode(test_sample)
      z = self.reparameterize(mean, logvar)
      #z = tf.random.normal(shape=[16,self.latent_dim])
      print(z.shape, mean.shape)
      fig, axs = plt.subplots(1,2)
      axs[0].hist(mean.numpy().flatten())
      axs[0].set_title('mean true')
      axs[1].hist(z.numpy().flatten())
      axs[1].set_title('z from normal')
      plt.show()
      predictions = self.decode(z)
      
      fig = plt.figure(figsize=(10, 10))
      for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.around(predictions[i, :, :, 0]*3))
        plt.axis('off')
      

      # tight_layout minimizes the overlap between 2 sub-plots
      #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
      plt.show()
      # add hist(z)

    def plot_latent_images(self, n=20, digit_size=45):
      """Plots n x n digit images decoded from the latent space."""

      norm = tfp.distributions.Normal(0, 1)
      grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
      grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
      image_width = digit_size*n
      image_height = image_width
      image = np.zeros((image_height, image_width))

      for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
          z = np.array([[xi, yi]])
          x_decoded = self.sample(z)
          digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
          image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.numpy()

      plt.figure(figsize=(10, 10))
      plt.imshow(np.around(image*3))
      plt.axis('Off')
      plt.show()



cvae = CVAE()


# In[]:

#cvae.train(epochs=100, batch_size=128, save_interval=1)



# Save models
#cvae.save('compiled_cvae_model_100')
#cvae.encoder.save('cvae_encoder_1000' )
#cvae.decoder.save('cvae_decoder_1000')


# Load models
#loaded_cvae = tf.keras.models.load_model('compiled_cvae_model_100')
cvae.encoder = tf.keras.models.load_model('cvae_encoder_1000')
cvae.decoder = tf.keras.models.load_model('cvae_decoder_1000')    

# In[]:

#cvae.save_imgs(0, seis[1])

#cvae.plot_latent_images()

# In[]:

plt.plot(cvae.loss_function)
plt.title("VAE Training Loss")
plt.ylabel("Computed loss")
plt.xlabel("Epoch")
plt.savefig("100_training_loss.png", dpi=1200)
plt.show()

cvae.total_training_time

# In[]

rows_labels = ['Realization 1', 'Realization 2', 'Realization 3', 'Realization 4']
columns_labels = ['Seismic facies', 'Latent space', 'Reconstructed facies']


fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 16))


for i in range(4):  # Rows
    for j in range(3):  # Columns
        # Plot seismic facies
        if j == 0:
            img = axs[i, j].imshow(seis[i])
            axs[i, j].set_ylabel(rows_labels[i], fontsize=14)
            axs[0, j].set_title(columns_labels[j],fontsize=14) 
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            fig.colorbar(img, ax=axs[i, j])

        # Plot latent space
        elif j == 1:
            layer = np.expand_dims(seis[i], 0)
            mean, logvar = cvae.encode(layer)
            z = cvae.reparameterize(mean, logvar)
            axs[i, j].hist(z.numpy().flatten())
            axs[0, j].set_title(columns_labels[j],fontsize=14)

        # Plot reconstructed facies
        else:
            predict = cvae.decode(z)
            img = axs[i, j].imshow(predict[0])
            axs[0, j].set_title(columns_labels[j],fontsize=14)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            fig.colorbar(img, ax=axs[i, j])

plt.tight_layout()
#plt.savefig("VAE_comparison_100.png", dpi=1200)
plt.show()

# In[]

cvae.latent_dim_shape



# In[ ]:


cvae.encoder.summary()


# In[ ]:


cvae.decoder.summary()



# In[ ]:

#plotting the histograms of realizations

rows_labels = ['Realization 1', 'Realization 2', 'Realization 3', 'Realization 4']
columns_labels = ['Input facies', 'Input histogram', 'Output facies', 'Output histogram']


fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 16))


for i in range(4):  # Rows
    for j in range(4):  # Columns
        # Plot seismic facies
        if j == 0:
            img = axs[i, j].imshow(seis[i])
            axs[i, j].set_ylabel(rows_labels[i], fontsize=14)
            axs[0, j].set_title(columns_labels[j],fontsize=14) 
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            fig.colorbar(img, ax=axs[i, j])

        # input histograms
        elif j == 1:
            axs[i, j].hist(seis[i].ravel())
            axs[i,j].set_xlabel('Value')
            axs[i,j].set_ylabel('Frequency')
            axs[0, j].set_title(columns_labels[j],fontsize=14)

        # Plot output facies
        elif j == 2:
            layer = np.expand_dims(seis[i], 0)
            mean, logvar = cvae.encode(layer)
            z = cvae.reparameterize(mean, logvar)
            predict = cvae.decode(z)
            img = axs[i, j].imshow(predict[0])
            axs[0, j].set_title(columns_labels[j],fontsize=14)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            fig.colorbar(img, ax=axs[i, j])
            
        else:
            axs[i, j].hist(predict[0].ravel())
            axs[i,j].set_xlabel('Value')
            axs[i,j].set_ylabel('Frequency')
            axs[0, j].set_title(columns_labels[j],fontsize=14)
            
#plt.tight_layout()
#plt.savefig("Histogram_comparison.png", dpi=1200)
plt.show()


# In[ ]:

# testing images

def test_images (X_test):
    
    mean, logvar = cvae.encode(X_test)
    
    z = cvae.reparameterize(mean, logvar)

    x_logit = cvae.decode(z)
    n=45
    plt.imshow(x_logit.reshape(4000, n,n)[0])
    plt.axis('off')
    plt.colorbar()
    plt.show()
    plt.close()
    return

test_images(X_test)

# In[]:

#applying the generative feature of VAE


def generate ():
    samples_init = tf.random.normal([1, cvae.latent_dim, cvae.latent_dim, 1])
    gan_facies = tf.experimental.numpy.around(cvae.sample(eps=samples_init)*3)
    plt.axis('off')
    n=45
    plt.imshow(gan_facies.reshape(200, n,n)[0])
    plt.colorbar(shrink=0.5)

    return 

fig, axs = plt.subplots(1, 4, figsize=(15,5))
for i in range(4):
    plt.sca(axs[i])
    plt.title("Generated realization {}".format(i+1))
    generate()
plt.tight_layout()
#plt.savefig("generative_1000", dpi=1200)
plt.show()

# In[ ]:

os.chdir(r"C:\Users\omars\OneDrive\Desktop\Master's Thesis\Automation")



# ES-MDA

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:35:46 2022
@author: Reza


Your code to read production data from simulation results
"""


#------------------------------------------------INTRODUCTIONS----------------------------------------------------------#
#Import libraries

np.random.seed(40)

# The list of variables to read.
# Well names
wellVars = ["PLT-S", "PLT-N", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "N1", "N2", "N3", "N4", "N5", "PLT-E"]
# Field variables
fieldVars = ["FOPR", "FWPR", "FWIR", "FWCT", "FPR", "FGPR", "FWPT", "FOPT"]
variables = wellVars + fieldVars
#--------------------------------------------------FUNCTIONS-------------------------------------------------------------#
#Calculating the observation based on the parameter. This is function d = g(m)
#the non-linear functions obsData = gFunctions(parameters, dLength)
#In the reservoir, the function is much more complex and needs a reservoir simulator to solve it (Eclipse, Intersect, etc.)
# sensitiveParams = np.load("SensitiveParameters.npy")
def func(x):
    if x.strip() == '':
        x = 0
    elif "*" in x.strip():
        x = 3
    return x

def gFunctions(parameters, dLength, ref = False):
    ''' Here, you can make your own functions!'''
    tNav_file = r"C:\Users\omars\OneDrive\Desktop\Master's Thesis\Field_tNav"
    
    time.sleep(1)
        
    if (ref):
        filename = "result.RSM"
        
    else:
        for l in range(len(parameters)):
            recons = cvae.decode(parameters[l])
            temp = "result_TEMP_{}".format(l+1)
            
            if os.path.exists(temp):
                try:
                    shutil.rmtree(temp)
                    shutil.copytree(tNav_file, temp)
                except:
                    if os.path.exists(temp+"\\RESULTS\Field\result.RSM"):
                        os.remove(temp+"\\RESULTS\Field\result.RSM")
                    shutil.copytree(tNav_file , temp, dirs_exist_ok = True)
            else:
                shutil.copytree(tNav_file, temp, dirs_exist_ok = True)
            
            with open(temp+"\\INCLUDE\Field_PERMX.inc", "w") as f:
                f.write("PERMX\n")
                r=0
                # Convert TensorFlow tensor to NumPy array
                recons_np = recons.numpy()
                
                #for r in range(recons.shape[0]):
                for i in range(45):
                        row_values = [str(np.clip(value*100 + 10, 10, 100)).strip('[]') for value in recons_np[r, i]]
                        # Join row values with spaces and write to file
                        f.write(" ".join(row_values) + "\n")

                        #r+=1 #dashed out because it will be out of range
                f.write("/")

            os.chdir(MAINDIR+"\\"+temp)
            
            # Runs the simulation model
            executable_path =r"C:\Users\omars\AppData\Local\Programs\RFD\tNavigator\23.4\tNavigator.exe"
            data_file = "Field.data"
            process = subprocess.Popen([executable_path,data_file])
            time.sleep(60)
            process.terminate()
            
            #os.system('$run.bat')
            os.chdir(MAINDIR)
        Running = np.ones((len(parameters),))
        Done = np.zeros((len(parameters),))
        while np.any(Running != Done):
            for q in range(len(parameters)):
                temp = "result_TEMP_{}".format(q+1)
                if os.path.exists(MAINDIR+"\\"+temp+"\\RESULTS\\Field\\result.RSM"):
                    Done[q] = 1
    time.sleep(1)
    num_of_total_vars = 4 # change as you want
    predData=np.zeros((dLength,num_of_total_vars, len(parameters)))
    for i in range(len(parameters)):
        
        time.sleep(1)
        temp = "result_TEMP_{}".format(i+1)
        if ref:
            file = open(MAINDIR+"\\result.RSM", "r") #I have to copy the RSM file only the first time into MAINDIR
        else:
            file = open(MAINDIR+"\\"+temp+"\\RESULTS\\Field\\result.RSM", "r")
        content = file.readlines()
        file.close()
        content.append("")
        counter = 0
        feature=0
        for line in content:
            for var in variables:

                if var in line:

                    c = 0
                    arr = np.array([x.strip() for x in content[counter].split("\t")])
                    indices = np.where(arr==var)[0]
                    for d in range(len(indices)):
                        powers = np.zeros((20,))
                        idx=indices[d]-1
                        if "*" in content[counter+2]:
                            powers = content[counter+2].split("\t")
                            powers = [func(x) for x in powers]
                            powers.pop(0)
                        elif "*" in content[counter-1]:
                            powers = content[counter-1].split("\t")
                            powers = [func(x) for x in powers]
                            powers.pop(0)

                        loop = True
                        ProdData = []
                        num = 1
                        a = 45
                        while(loop):
                            if var in fieldVars:
                                if content[counter+3].strip()!="" and "*" in content[counter+2]:
                                    num = 4
                                    a=46
                                else:
                                    num = 3
                            innerCounter = 0
                            for j in range(counter+num, len(content)):
                                if (content[j].strip()==""):
                                    continue
                                #if (np.mod(innerCounter,2)!=0):
                                ProdData.append(float(content[j].split()[idx])*np.power(10, int(powers[idx])))
                                innerCounter += 1
                                if (j - counter >= a or (content[j+1].strip()=="" and j > counter+num+10) ):
                                    break
                            loop = False
#                         print(ProdData)
                        #predData[0:len(ProdData),feature,i] = np.array(ProdData)
                        predData[0:len(ProdData)-1,feature,i] = np.array(ProdData[1:31])
                        feature += 1

            counter += 1
    if predData.shape[2] == 1:
        predData = np.squeeze(predData, axis=2)
    return predData
 


# In[101]:


start = time.time()
#Constants
MAINDIR = r"C:\Users\omars\OneDrive\Desktop\Master's Thesis\Automation"
mLength = 200 #the length of the parameter m = Latent space size
dLength = 30 #the length of the data d = number of timesteps
nEnsemble = 50 #the number of ensembles = number of realizations
# changed from 100 to 50 


alpha_max = 1000.
maxIter = 6

#Declaring variables
num_of_total_vars = 4
mInit = np.zeros((nEnsemble, mLength)) #Initial ensemble
mAnswer = np.zeros(mLength) #True parameter values
mPrior = np.zeros((nEnsemble, mLength)) #Prior ensemble
mPred = np.zeros((nEnsemble, mLength)) #Predicted ensemble
models = np.zeros((nEnsemble, mLength, maxIter+1)) #Predicted ensemble
mAverage = np.zeros(mLength)
dAverage = np.zeros(dLength*num_of_total_vars)
dPrior = np.zeros((dLength*num_of_total_vars, nEnsemble)) #Prior ensemble
error = np.zeros((maxIter+1,))
tElapsed = np.zeros((maxIter+1,))

d = np.zeros((dLength*num_of_total_vars, nEnsemble, maxIter + 1)) #Forecasted data
obsData = np.zeros((dLength, nEnsemble)) #Observed data --> true data + measurement noise
alpha = np.zeros(maxIter)
z = np.zeros((dLength, nEnsemble))

deltaM = np.zeros((nEnsemble, mLength))
deltaD = np.zeros((dLength*num_of_total_vars, nEnsemble))
ddMD = np.zeros(nEnsemble)
ddDD = np.zeros(nEnsemble)


#for p in range(maxIter):
#    alpha[p] = (2**(maxIter-p))

# constant alpha
for p in range(maxIter):
    alpha[p] = (1/maxIter)

#Generating the initial parameter (base case)
''' Here, you can initialize your own model parameter'''
mean, logvar = cvae.encode(seis[:50]) #changed from 100 to 50

mPrior = cvae.reparameterize(mean, logvar)
seis_logit = cvae.decode(mPrior)
#_, _, mPrior = cvae.encoder.predict(seis)
models[:,:,0] = mPrior


# In[102]:


#----------------------------------------------THE 'ANSWERS'-------------------------------------------------------------#

#This parts consists of parameters that are considered the truth value of the model. 
#Consequently, by plugging it into the gFunction() we would get the true observed data (data without noise) 

#mAnswer=[9]
dAnswer = gFunctions(mAnswer, dLength, True)

dAnswer = dAnswer[:,:,0] #had to decrease the dimensions becasue all values are the same
max_=np.max(dAnswer, axis=0)
dAnswer = dAnswer/max_


np.random.seed(40)
Zd = np.random.normal(0, 1, (dAnswer.shape[0]*dAnswer.shape[1],1))
err = np.zeros((dAnswer.shape[0], dAnswer.shape[1], nEnsemble))
for j in range(dAnswer.shape[1]):
    err[:,j, :] = np.random.normal(0, np.min([1, np.std(dAnswer[:,j])]), (dAnswer.shape[0], nEnsemble))
error = np.zeros((err.shape[0]*err.shape[1],nEnsemble))
for i in range(nEnsemble):
    error[:, i] = err[:,:,i].ravel(order="F")

eAverage = np.zeros((error.shape[1],)) 
for i in range(error.shape[1]):
    summationE = np.sum(error[:,i])
    eAverage[i] = (1/nEnsemble)*summationE


deltaE = np.zeros((error.shape[0], nEnsemble))
ddE = 0.
for i in range(nEnsemble):
    deltaE[:,i] = error[:,i] - eAverage[i]
    
    ddE += np.outer(deltaE[:,i], deltaE[:,i])
CD = ddE / (nEnsemble-1.)
e = np.dot(np.diag(CD), Zd)

dUnc = dAnswer.ravel(order="F") + e # Uncertain d, with measurment uncertainty .. un -changed the dimensions to match shapes

params = {"CD":CD, "e":e, "err":err, "error":error, "Zd":Zd}
with open("StatisticalParams.bin", "wb") as f:
    pickle.dump(params, f)


# In[ ]:


#------------------------------------------------POPULATING ENSEMBLE-----------------------------------------------------#
#Populate ensemble based on mean and standard deviation (we assume normal distribution for the noise in measurement)
    
mInit = mPrior #Initial ensemble
NC = 2

#Calculate prediction
for j in range(int(nEnsemble/NC)):
    print("Realization NO.: {:0.0f} - {:.0f} in Iteration No.: {:0.0f}".format(NC*j+1,NC*j+NC,0))
    data = gFunctions(mPrior[NC*j:(j+1)*NC,:], dLength, False)
    #data = data[:,sensitiveParams,:]
    r = 0
    for h in range(NC*j,(j+1)*NC):
        data[:,:,r] = (data[:,:,r])/(max_)
        dPrior[:,h] = data[:,:,r].ravel(order="F")
        d[:,h, 0] = dPrior[:,h].copy()
        r += 1
#Calculate Average and Covariance M (CM)
for i in range(mLength):
    summationM = np.sum(mInit[:,i])
    mAverage[i] = (1/nEnsemble)*summationM


ddM = 0.

for j in range(nEnsemble):
    deltaM[j,:] = mInit[j,:] - mAverage[:]

    #This should be a matrix
    ddM += np.outer(deltaM[j,:],deltaM[j,:])

covarianceM = ddM / (nEnsemble - 1.)

Error = np.zeros((maxIter+1,))
for i in range(nEnsemble):
    ee = (dUnc.reshape(-1,1)-dPrior[:,i].reshape(-1,1))
    X = (np.eye(CD.shape[0], CD.shape[1])*CD)
    Error[0] += 0.5 * np.dot(np.dot(ee.T, np.linalg.inv((X))), ee)
Error[0] /= nEnsemble
tElapsed[0] = time.time()-start

#------------------------------------------MAIN LOOP STARTS HERE---------------------------------------------------------#

deltaD = np.zeros((dLength*num_of_total_vars, nEnsemble))
for p in range(maxIter):
    #Get data

    #Calculate Average and Covariance MD and Covariance DD
    for i in range(mLength):
        summationM = np.sum(mPrior[:,i])
        mAverage[i] = (1/nEnsemble)*summationM
    
    #for i in range(dLength): changed because the loop does not read after 30
    for i in range(dLength*num_of_total_vars):
        summationD = np.sum(dPrior[i,:])
        dAverage[i] = (1/nEnsemble)*summationD
    
    ddMD = 0.
    ddDD = 0.
    for j in range(nEnsemble):
        deltaM[j,:] = mPrior[j,:] - mAverage[:]
        deltaD[:,j] = dPrior[:,j] - dAverage[:]
        
        #This should be a matrix
        ddMD += np.outer(deltaM[j,:],deltaD[:,j])
        ddDD += np.outer(deltaD[:,j],deltaD[:,j])

    covarianceMD = ddMD / (nEnsemble - 1.)
    covarianceDD = ddDD / (nEnsemble - 1.)
     
    dUncT = dAnswer.ravel(order="F") + np.dot(np.sqrt(alpha[p])*np.diag(CD), Zd) # Uncertain d, with measurment uncertainty
    #Main update equation
    for j in range(nEnsemble):
        X = (np.eye(CD.shape[0], CD.shape[1])*CD)
        dummyMat = np.matmul(covarianceMD,np.linalg.inv(covarianceDD + alpha[p]*X)) 
        dummyVec = dUncT - dPrior[:,j]
        mPred[j,:] = mPrior[j,:] + np.matmul(dummyMat,dummyVec)

    
    #Calculate new forecast based on the predicted parameters
    for j in range(int(nEnsemble/NC)):
        print("Realization NO.: {:0.0f} - {:.0f} in Iteration No.: {:0.0f}".format(NC*j+1,NC*j+NC,p+1))
        data = gFunctions(mPrior[NC*j:(j+1)*NC,:], dLength, False)

        r = 0
        for h in range(NC*j,(j+1)*NC):
            data[:,:,r] = (data[:,:,r])/(max_)
            #data[:,:,r] = (data[:,:,r]-mean_)/(std_)
            dPrior[:,h] = data[:,:,r].ravel(order="F")
            d[:,h,p+1] = dPrior[:,h].copy()
            r += 1
    
    #Calculate Average and Covariance M (CM)
    for i in range(mLength):
        summationM = np.sum(mPrior[:,i])
        mAverage[i] = (1/nEnsemble)*summationM
    ddM = 0.

    for j in range(nEnsemble):
        deltaM[j,:] = mPrior[j,:] - mAverage[:]

        #This should be a matrix
        ddM += np.outer(deltaM[j,:],deltaM[j,:])

    covarianceM = ddM / (nEnsemble - 1.) # Covariance of prior model (refer to Emerick 2012, page 6 of PDF, above Eq. 11)
    
    for i in range(nEnsemble):
        ee = (dUncT.reshape(-1,1)-dPrior[:,i].reshape(-1,1))
        em = (mPred[i,:]- mPrior[i,:]).reshape(-1,1)
        X = (np.eye(CD.shape[0], CD.shape[1])*CD)
        Error[p+1] += 0.5 * np.dot(np.dot(em.T, np.linalg.inv(covarianceM)), em) + 0.5 * np.dot(np.dot(ee.T, np.linalg.inv(X*alpha[p])), ee)

        
    Error[p+1] /= nEnsemble
    #Update the prior parameter for next iteration
    mPrior = mPred.copy()
    tElapsed[p+1] = time.time() - start
    #Plotting for change of average of the parameters
    meanP = np.mean(mPred, axis=1)
    models[:,:,p+1] = mPred.copy()
    results = {
        "dPrior": dPrior,
        "d": d,
        "mInit": mInit,
        "dAnswer": dAnswer,
        "Error" : Error,
        "mPrior": mPrior,
        "tElapsed": tElapsed,
        "models": models
    }
    with open("Results_CnonvLSTM_NEW.bin", "wb") as f:
        pickle.dump(results, f)
    plt.figure(3)
    plt.plot(meanP, label='iter = %i' %(p+1))
    plt.legend()

    plt.title('(Averaged) model parameters')
    plt.xlabel('i-th parameter')
    plt.ylabel('m[i]')

    plt.show()

#-------------------------------------------------OUTPUT-----------------------------------------------------------------#

#Plot of the ensemble of the parameters
plt.figure(1)
plt.plot(mPrior, 'g-') #m changed to mPrior
plt.plot(mPred, 'b-')
plt.plot(mAnswer, 'r-')

plt.title('Model parameters')
plt.xlabel('i-th parameter')
plt.ylabel('m[i]')

plt.show()

#Plot of the ensemble of the data
plt.figure(2)
#plt.plot(d, 'g-') #shape is greater than 2D
plt.plot(dPrior, 'b-')
plt.plot(dAnswer, 'r-')

plt.title('Data')
plt.xlabel('i-th data')
plt.ylabel('d[i]')
plt.show()

print('green = initial ensemble')
print('blue = ensemble at last iteration')
print('red = the answer')
end = time.time()
print("Elapsed time: {:0.2f}".format(end-start))

# In[]

os.chdir(r"C:\Users\omars\OneDrive\Desktop\Master's Thesis\constant alpha")

# production graphs

RSM_variables = ['FWPR', 'FPR', 'FWPT', 'FOPT' ]
units = ["STB/DAY", "PSIA", "STB", "STB" ]

FWPR = d[:30,:,:]*max_[0]
FPR= d[30:60,:,:]*max_[1]
FWPT= d[60:90,:,:]*max_[2]
FOPT= d[90:120,:,:]*max_[3]

dAnswer = dAnswer * max_

for i, (variable, unit) in enumerate(zip([FWPR,  FPR, FWPT, FOPT], units)):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{RSM_variables[i]} Production Plots', fontsize=16)
    handles = [] 
    labels = []  
    
    for iteration in range(maxIter):
        ax = axes[iteration // 3, iteration % 3]  # Calculate subplot index
        ax.plot(variable[:, :, iteration], 'grey', alpha=0.4,  label='Realizations')
        #ax.plot(variable[:, :, iteration].mean(axis=1), 'b',  label='Average value')
        ax.plot(dAnswer[:,i], 'r',linestyle='dashdot',  label='Observed values')
        ax.plot(np.percentile(variable[:, :, iteration], 50, axis=1), linestyle='dashed', color='b', label='50th percentile' )
        ax.set_title(f'Iteration {iteration+1}')
        ax.set_xlabel('Days')
        ax.set_ylabel(unit)
        #ax.legend()
        
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',ncol=3, frameon=False)
    plt.savefig(f'{RSM_variables[i]}_figure.png', dpi = 1200)
    #plt.tight_layout()
    plt.show()



# In[]

#error boxplots

for i, (variable, unit) in enumerate(zip([FWPR,  FPR, FWPT, FOPT], units)):
    fig = plt.figure(figsize=(8, 6))
    plt.title(f'{RSM_variables[i]} error boxplot', fontsize=16)
    handles = [] 
    labels = []  
    
    for iteration in range(maxIter):
        mse = [mean_squared_error(dAnswer[:, i], variable[:, j, iteration]) for j in range(nEnsemble)]
        plt.boxplot(mse, positions=[iteration+1], widths=0.6)
        #handles.append(plt.Line2D([], [], linestyle='-', linewidth=1.2, color='black'))
        labels.append(f'Iteration {iteration+1}')
        
    plt.xticks(range(1, maxIter+1), labels)
    plt.xlabel('Iteration')
    plt.ylabel(unit)
    plt.savefig(f'{RSM_variables[i]}_boxplot.png', dpi=1200)
    plt.tight_layout()
    plt.show()

# In[]

# decoding the realizations 
number_of_iterations = 6
ith_realization = 8

decoded=np.zeros((number_of_iterations, 45,45,1))

results = models[ith_realization,:,:]
for z in range(number_of_iterations):
    decoded[z] = cvae.decode(results[:,z])
    decoded[z] = np.clip(decoded[z] *100 + 10, 10,100) #converting facies to permeability


fig, axs = plt.subplots(2, 3, figsize=(10, 8))

for i, ax in enumerate(axs.flat):
    im = ax.imshow(decoded[i], vmin=np.min(decoded), vmax=np.max(decoded))  # Set vmin and vmax
    ax.set_title(f'Assimilation step {i+1}')
    ax.axis('off')

# Add a color bar to the side of the subplots
cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])  # Adjust position
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Permeability')


plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the layout to accommodate the color bar
fig.suptitle(f'Realization {ith_realization}', fontsize=16)
plt.savefig(f"decoded_perm{ith_realization}.png", dpi= 1200)
plt.show()

# In[]

plt.imshow(seis[8])