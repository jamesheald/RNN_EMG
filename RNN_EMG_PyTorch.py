# train a vanilla RNN using PyTorch

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import torch.distributions as dist
import pickle

class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,n_layers,dropout_prob,output_dim):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_dim,hidden_dim,n_layers,batch_first=True,dropout=dropout_prob,bias=True)
        self.fc = nn.Linear(hidden_dim,output_dim)

        # initialize parameters
        scale_weights = 1
        for layer in range(n_layers):
            if layer == 0:
                exec(f"self.rnn.weight_ih_l{layer} = torch.nn.Parameter(torch.randn(hidden_dim,input_dim)/np.sqrt(hidden_dim)*scale_weights)")
            else:
                exec(f"self.rnn.weight_ih_l{layer} = torch.nn.Parameter(torch.randn(hidden_dim,hidden_dim)/np.sqrt(hidden_dim)*scale_weights)")
            exec(f"self.rnn.weight_hh_l{layer} = torch.nn.Parameter(torch.randn(hidden_dim,hidden_dim)/np.sqrt(hidden_dim)*scale_weights)")
            exec(f"self.rnn.bias_ih_l{layer} = torch.nn.Parameter(torch.randn(hidden_dim)/np.sqrt(hidden_dim)*scale_weights)")
            exec(f"self.rnn.bias_hh_l{layer} = torch.nn.Parameter(torch.randn(hidden_dim)/np.sqrt(hidden_dim)*scale_weights)")
        self.fc.weight = torch.nn.Parameter(torch.randn(output_dim,hidden_dim)/np.sqrt(output_dim)*scale_weights)
        self.fc.bias = torch.nn.Parameter(torch.randn(output_dim)/np.sqrt(output_dim)*scale_weights)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self,x):

        # initialize hidden states
        h0 = torch.zeros(self.n_layers,x.size(0),self.hidden_dim,requires_grad=True) # n_layers x batch_size x hidden_dim

        # forward propagate
        out,h = self.rnn(x,h0.detach()) # x is batch_size x sequence_length x input_dim, out is batch_size x sequence_length x hidden_dim

        # linear readout
        out = self.fc(out) # input is batch_size x sequence_length x hidden_dim, out is batch_size x sequence_length x output_dim

        return out

class Optimization:
    def __init__(self,model,optimizer,loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_losses = []
        self.validation_losses = []

    def train_step(self,x,y):
        self.model.train() # set to train mode

        y_hat = self.model(x)

        y,y_hat = remove_NaNs(y.float(),y_hat)

        loss = self.loss_function(y,y_hat)

        loss.backward()

        self.optimizer.step()     
        self.optimizer.zero_grad()
        
        return loss.item()           

def remove_NaNs(y,y_hat):
    idx = torch.where(~torch.isnan(y))
    return y[idx],y_hat[idx]

def train_model(self,train_data,validation_data,model_path,batch_size=64,n_epochs=50,input_dim=1,sequence_length=1):

    for epoch in range(n_epochs):
        batch_train_losses = []
        for x_train, y_train in train_data:
            x_train = x_train.view([batch_size,sequence_length,input_dim]).to(device) # change dimensions from batch_size x input_dim to batch_size x sequence length (1) x input_dim (required by RNN)
            y_train = y_train.to(device)
            loss = self.train_step(x_train,y_train)
            batch_train_losses.append(loss)
        train_loss = np.mean(batch_train_losses)
        self.train_losses.append(train_loss)

        with torch.no_grad():
            batch_validation_losses = []
            for x_validation, y_validation in validation_data:
                x_validation = x_validation.view([batch_size,sequence_length,input_dim]).to(device) # change dimensions from batch_size x input_dim to batch_size x sequence length (1) x input_dim (required by RNN)
                y_validation = y_validation.to(device)
                self.model.eval()
                y_hat = self.model(x_validation)
                y_validation,y_hat = remove_NaNs(y_validation,y_hat)
                loss = self.loss_function(y_validation,y_hat).item()
                batch_validation_losses.append(loss)
            validation_loss = np.mean(batch_validation_losses)
            self.validation_losses.append(validation_loss)

        if epoch <= 10 or epoch % 50 == 0:
            print(f"[{epoch}/{n_epochs}] training loss: {train_loss:.4f} \t validation loss: {validation_loss:.4f}")

    torch.save(self.model.state_dict(),model_path)

def plot_losses(self):
    plt.figure(1)
    plt.plot(self.train_losses,label='training loss')
    plt.plot(self.validation_losses,label='validation loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Losses')
    plt.ion()
    plt.show()

def evaluate(self,test_data,sequence_length,batch_size=1,input_dim=1):
    with torch.no_grad():
        predictions = []
        true_values = []
        for x_test, y_test in test_data:
            x_test = x_test.view([batch_size,sequence_length,input_dim]).to(device) # change dimensions from batch_size x input_dim to batch_size x sequence length (1) x input_dim (required by RNN)
            y_test = y_test.to(device)
            self.model.eval()
            y_hat = self.model(x_test)
            predictions.append(y_hat.to(device).detach().numpy())
            true_values.append(y_test.to(device).detach().numpy())
    return predictions,true_values # number of batches x batch_size x sequence_length x output_dim

def format_predictions(predictions,true_values,scaler,output_dim):
    pred_shape = np.shape(predictions)
    pred = np.reshape(predictions,(int(np.prod(pred_shape)/output_dim),output_dim))
    pred = scaler.inverse_transform(pred)
    pred = np.reshape(pred,pred_shape)
    true_shape = np.shape(true_values)
    true = np.reshape(true_values,(int(np.prod(true_shape)/output_dim),output_dim))
    true = scaler.inverse_transform(true)
    true = np.reshape(true,true_shape)
    return pred,true

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

emg = io.loadmat("time_normalized_EMG_pre_RNN.mat")
emg_full = emg['EMG_time_normalized']

emg = np.zeros((100,6,4))
for muscle in range(6):
    for direction in range(4):
        emg_full[:,muscle,direction] = gaussian_filter1d(emg_full[:,muscle,direction],50) # filter
        emg[:,muscle,direction] = signal.decimate(emg_full[:,muscle,direction],14) # downsample
        emg[:,muscle,direction] -= emg[0,muscle,direction]

n_muscles = 6
n_conditions = 4
movement_delay = 50
spike_length = 100

max_hold_on_time = 100
hold_decay_time = 100
max_sequence_length = max_hold_on_time + movement_delay + spike_length
n_episodes_per_condition = 5
n_episodes = n_conditions*n_episodes_per_condition
input_dim = 5
condition_specific_input = np.linspace(1,-1,n_conditions)
batch_size = n_conditions

hold_on_time = []
hold_decay = 1/(1 + np.exp(torch.linspace(-7,7,hold_decay_time)))
inputs = torch.zeros([n_episodes,max_sequence_length,input_dim])
n = dist.Normal(0,1)
x = torch.linspace(-5,5,spike_length)
EMG = np.zeros([n_episodes,max_sequence_length,n_muscles])
muscle_amplitude = np.random.rand(n_muscles,n_conditions)*0.75+0.25
for episode in range(n_episodes):
    hold_on_time.append(np.random.randint(max_hold_on_time)+1)
    hold_on = torch.ones(hold_on_time[-1])
    hold_off = torch.zeros(max_sequence_length-hold_decay_time-hold_on_time[-1])
    inputs[episode,:,0] = torch.cat((hold_on,hold_decay,hold_off),0)
    condition = episode % n_conditions
    # inputs[episode,:,1] = inputs[episode,:,0]*condition_specific_input[condition]
    inputs[episode,:,condition+1] = inputs[episode,:,0] # one-hot encoding
    for muscle in range(n_muscles):
        t1 = hold_on_time[-1]+movement_delay
        t2 = spike_length
        t3 = max_hold_on_time-hold_on_time[-1]
        EMG[episode,0:t1,muscle] = np.NaN
        EMG[episode,(t1):(t1+t2),muscle] = emg[:,muscle,condition]
        EMG[episode,(t1+t2):(t1+t2+t3),muscle] = np.NaN

scaler = get_scaler('standard')
EMG_shape = np.shape(EMG)
EMG = EMG.reshape(int(np.prod(EMG_shape)/n_muscles),n_muscles)
EMG = scaler.fit_transform(EMG)
EMG = torch.from_numpy(EMG)
EMG = np.reshape(EMG,EMG_shape)

train = TensorDataset(inputs,EMG)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True) # number of batches is determined by the amount of data and batch sizes (number of rows of inputs or EMG divided by batch_size and then floored)
val_loader = train_loader
test_loader = train_loader

n_layers = 1
hidden_dim = 300
n_epochs = 500
dropout_prob = 0.2
learning_rate = 1e-3
weight_decay = 1e-6
model_path = "/Users/James/pytorch/RNN_muscles.pt"

network = RNN(input_dim,hidden_dim,n_layers,dropout_prob,output_dim=n_muscles)
optimizer = Adam(network.parameters(),lr=learning_rate,weight_decay=weight_decay)
loss_function = nn.MSELoss(reduction='mean')

opt = Optimization(network,optimizer,loss_function)
train_model(opt,train_data=train_loader,validation_data=val_loader,model_path=model_path,batch_size=batch_size,n_epochs=n_epochs,input_dim=input_dim,sequence_length=max_sequence_length)
plot_losses(opt)
predictions, true_values = evaluate(opt,test_data=test_loader,sequence_length=max_sequence_length,batch_size=batch_size,input_dim=input_dim)
predictions, true_values = format_predictions(predictions,true_values,scaler,output_dim=n_muscles)

n_iterations = 1 # n_iterations = np.shape(true_values)[0]
fig,ax = plt.subplots(n_iterations,batch_size)
cmap = plt.get_cmap("tab10") # https://matplotlib.org/stable/tutorials/colors/colormaps.html
titles = ['backward movement','leftward movement','forward movement','rightward movement']
muscle_names = ['pectoralis major','posterior deltoid','biceps brachii','triceps lateral head','brachioradialis','triceps long head']
y_max = np.max((np.nanmax(true_values),np.nanmax(predictions)))+0.5
y_min = np.min((np.nanmin(true_values),np.nanmin(predictions)))-0.5
legend_elements = [Line2D([0], [0], color='k',alpha=0.4, lw=3, label='actual'),Line2D([0], [0], linestyle='--',color='k', label='model')]
for example in range(batch_size):
    for muscle in range(n_muscles):
        ax[example].plot(true_values[iteration,example,:,muscle],'-',color=cmap(muscle),label=muscle_names[muscle],alpha=0.4,linewidth=3)
        ax[example].plot(predictions[iteration,example,:,muscle],'--',color=cmap(muscle))
        ax[example].set_ylim([y_min,y_max])
        ax[example].set_xlabel('time')
        ax[example].set_title(titles[example])
        ax[example].spines['right'].set_visible(False)
        ax[example].spines['top'].set_visible(False)
    if example == 0:
        ax[example].set_ylabel('EMG (a.u.)')
    if example == 0:
        legend1 = ax[example].legend(frameon=False,loc='upper left')
        ax[example].legend(handles=legend_elements,frameon=False,loc='lower right')
        ax[example].add_artist(legend1)
plt.ion()
plt.show()
