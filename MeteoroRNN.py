import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import sys


learningRate=0.0001
epoch=20
sequence_length=7
hiddenDim=5
minClip=-0.5
maxClip=0.5
Output_activation_function="tanh" 
Hidden_layer_activation="tanh"



class MeteoroRNN:
    def __init__(self,learningRate,epoch,sequence_length,hiddenDim,minClip,maxClip,feature_size):
        
        
        #model arch params
        #self.numLayers= numLayers
        self.hiddenDim = hiddenDim
        
        #training params
        self.epoch = epoch
        self.sequence_length = sequence_length
        
        
        #backprop related params
        self.learningRate = learningRate
        self.minClip = minClip
        self.maxClip = maxClip
        
        self.input_layer_size = feature_size
        self.outputDim = 1 #if not isinstance(self.Y_Train[0], np.ndarray) else len(self.Y_Train[0])
        self.U = np.random.uniform(0, 1, (self.hiddenDim, self.input_layer_size))
        self.W = np.random.uniform(0, 1, (self.hiddenDim, self.hiddenDim))
        self.V = np.random.uniform(0, 1, (self.outputDim, self.hiddenDim))
        self.bh = np.zeros((self.hiddenDim, 1))  # Hidden layer bias
        self.by = np.zeros((self.outputDim, 1))  # Output layer bias
        
        self.BU = np.random.uniform(0, 1, (self.hiddenDim, self.input_layer_size))
        self.BW = np.random.uniform(0, 1, (self.hiddenDim, self.hiddenDim))
        self.BV = np.random.uniform(0, 1, (self.outputDim, self.hiddenDim))
        self.Bbh = np.zeros((self.hiddenDim, 1))  # Hidden layer bias
        self.Bby = np.zeros((self.outputDim, 1))  # Output layer bias
        
        np.random.seed(1200)
     
    def calculateLoss(self,inputData,OutputData):
        loss=0.0 
        i=0 
        T = self.sequence_length
        while i<inputData.shape[0]-T: 
           hidden_state_t_minus_1 = np.zeros((hiddenDim,1))  
           output=0.0     
           for t in range(T): 
              hiddenState=self.HiddenLayerActivation(np.dot(self.U, inputData[i+t].reshape(-1,1))+np.dot(self.W, hidden_state_t_minus_1.reshape(-1,1))+self.bh)
              output=self.OutputLayerActivation(np.dot(self.V, hiddenState)+self.by) 
              hidden_state_t_minus_1=hiddenState 
           loss+=(output-OutputData[i+T])**2 / 2
           i+=1
        return loss  

    def saveBestModel(self):
         self.BU = self.U
         self.BW = self.W
         self.BV = self.V
         self.Bbh = self.bh
         self.Bby = self.by
         
         
    #input should be in the format of month , day encoded ,that day's temperature regularized to -1,1 and output should be next days temperature
     #forward pass ***************check on proper looping stratergy*************
          #regularize input and output between -1 and 1
                #so hidden layer recurrent weights are from one hidden layer neuron to all the neurons 
                #use tanh for hidden layer
                #use sigmoid for output layer
                #each has an output
                #calculate dh
                #calculate dW
                #calculate dU
                #calculatte dV
                
          #training loss
          #validation loss
    def train(self,trainData,OutputTrainData,validationData,OutputValidationData):
        T = self.sequence_length
        loss = 0.0
        U = self.U
        W = self.W
        V = self.V
        bh = self.bh
        by = self.by
        hiddenDim=self.hiddenDim
        maxClip=self.maxClip
        minClip=self.minClip
        BestLoss=sys.maxsize
        trainLossData=[]
        validLossData=[]
        for epoch in range(self.epoch):
            i=0;
            trainLoss=self.calculateLoss(trainData,OutputTrainData)
            ValidationLoss=self.calculateLoss(validationData,OutputValidationData)
            if(trainLoss<BestLoss): # to handle fluctations due to exploding gradients
                self.saveBestModel()
            print("Epoch {}, trainLoss: {} ,validation loss{}".format(epoch+1, trainLoss,ValidationLoss))
            trainLossData.append(trainLoss.squeeze().flatten())
            validLossData.append(ValidationLoss.squeeze().flatten())
            while i<trainData.shape[0]-T:
                aggInput= np.zeros((T, hiddenDim)) 
                hiddenState= np.zeros((T, hiddenDim))
                aggOutput= np.zeros((T, 1))
                Output= np.zeros((T, 1))
                error = np.zeros((T, 1))
                    #init delta values of weghts
                deltaU = np.zeros(U.shape)    
                deltaV = np.zeros(V.shape)    
                deltaW = np.zeros(W.shape)    
                deltabh= np.zeros(bh.shape)      
                deltaby= np.zeros(by.shape)    
                    # forward pass - for each sequence
                    # 
                for t in range(T):
                    if t < 1:
                        hidden_state_t_minus_1 = np.zeros_like(hiddenState[0]).reshape(-1,1)
                    else:
                        hidden_state_t_minus_1 = hiddenState[t-1].reshape(-1,1)
                    #print("shape U:{} W:{} input:{}, hidden{}",aggInput[t].shape,W.shape,trainData[i+t].shape,hidden_state_t_minus_1.shape)
                    aggInput[t]=(np.dot(U, trainData[i+t].reshape(-1,1))+np.dot(W, hidden_state_t_minus_1.reshape(-1,1))+bh).flatten()  #U-hidden_dim x input_size ,Xtrain[i]-input_size x 1 W-hidden_dim x hidden_dim ,hidden_state-hidden_dim x 1
                    hiddenState[t]=self.HiddenLayerActivation(aggInput[t]) #hidden_dim x 1
                    aggOutput[t]=np.dot(V, hiddenState[t].reshape(-1,1))+by #V-1 x hidden_dim hidden hiddenState[t]- hiddem_dim x 1
                    Output[t]=self.OutputLayerActivation(aggOutput[t])  #1 x 1
                    error[t]=Output[t]-OutputTrainData[i+t+1] 
                #backward pass
                d_output_agg=self.OutputLayerDerivateActivation(aggOutput.reshape(-1,1)) #dOutput/daggOutput
                d_input_agg=self.HiddenLayerDerivateActivation(aggInput.reshape(-1,1))  #dhiddenState/daggInput
                dh_next=np.zeros((hiddenDim,1))  # dh_next- hidden_dim x 1
                for t in reversed(range(T)):
                   deltaV+= error[t]*d_output_agg[t]*(hiddenState[t].reshape(1,-1)) #1Xhidden_dim 
                   #dJ/dh calculation W-T * d_input_agg dot dh_next+ V-T *dj/dO * d_output_agg 
                   if(t<T-1):
                        dh=np.dot(W*d_input_agg[t+1].reshape(-1,1),dh_next)+ (V.reshape(-1,1)*d_output_agg[t]*error[t]) #dh-hidden_dim x 1 , W-hiddem_dim x hidden_dim , hidden
                   else:
                        dh=V.reshape(-1,1)*d_output_agg[t]*error[t]
                   #dj/dw=dJ/dh dot (h[t-1]*d_input_agg) 
     
                   deltaW+=np.transpose(np.dot(dh*d_input_agg[t].reshape(-1,1),hiddenState[t-1].reshape(1,-1)))
                   #dj/du=dj/dh dot (x[t]*d_input_agg )
                   deltaU+=np.array(np.dot(dh*d_input_agg[t],trainData[i+t].reshape(1,-1)),dtype=np.float64) #
                   #dj/dbh=dj/dh dot d_input_agg 
                   deltabh+=dh*d_input_agg[t]
                   #dj/dby=
                   deltaby+=error[t]*d_output_agg[t]
                   
                   dh_next=dh #
                   #self.max_clip(deltaW)
                   #self.max_clip(deltaU)
                   #self.max_clip(deltaV)
                   #self.max_clip(deltabh)
                   #self.max_clip(deltaby)
                   #self.min_clip(deltaW)
                   #self.min_clip(deltaU)
                   #self.min_clip(deltaV)
                   #self.min_clip(deltabh)
                   #self.min_clip(deltaby)
                i=i+1
                U -= self.learningRate * deltaU
                W -= self.learningRate * deltaW
                V -= self.learningRate * deltaV
                bh -= self.learningRate * deltabh
                by -= self.learningRate * deltaby
        self.U = self.U #loading best model
        self.W = self.W
        self.V = self.V
        self.bh = self.bh
        self.by = self.by
        default_index = range(1, len(trainLossData) + 1)
        plt.plot(default_index, trainLossData, label='trainLossData', marker='o')

        # Plotting the second set of values with default index
        plt.plot(default_index, validLossData, label='ValidationLoss', marker='x')
        
        plt.xlabel('Index')
        plt.ylabel('Loss')
        plt.title('Training vs Validation loss')
        # Adding legend
        plt.legend()
        # Displaying the plot
        plt.show()
    
    def predict(self,inputData,OutputData):
        loss=0.0 
        i=0 
        T = self.sequence_length
        hidden_state_t_minus_1 = np.zeros((hiddenDim,1))
        predictions =[]
        actualValues=[]
        while i<inputData.shape[0]-T:        
            output=0.0
            for t in range(T): 
                  hiddenState=self.HiddenLayerActivation(np.dot(self.U, inputData[i+t].reshape(-1,1))+np.dot(self.W, hidden_state_t_minus_1.reshape(-1,1))+self.bh)
                  output=self.OutputLayerActivation(np.dot(self.V, hiddenState)+self.by) 
                  hidden_state_t_minus_1=hiddenState
            predictions.append(output.squeeze().flatten() )
            actualValues.append(OutputData[i+T].squeeze().flatten())
            i+=1
        default_index = range(1, len(predictions) + 1)
        plt.plot(default_index, actualValues, label='Values 1', marker='o')

        # Plotting the second set of values with default index
        plt.plot(default_index, predictions, label='Values 2', marker='x')
        
        plt.xlabel('Index')
        plt.ylabel('Temperature')
        plt.title('Actual vs Predicted Values')
        # Adding legend
        plt.legend()
        # Displaying the plot
        plt.show()
    
    def HiddenLayerActivation(self,x):
        return self.ActivationMethod(x,Hidden_layer_activation)
    
    def HiddenLayerDerivateActivation(self,x):
        return self.ActivationMethodDerivative(x,Hidden_layer_activation)
        
    def OutputLayerActivation(self,x):
        return self.ActivationMethod(x,Output_activation_function)
    
    def OutputLayerDerivateActivation(self,x):
        return self.ActivationMethodDerivative(x,Output_activation_function)
        
    def ActivationMethod(self,x,method):
        y=np.array(x,dtype=np.float32)
        if method=="sigmoid":
           return self.tanh_function(y)
        elif method=="tanh":
           return self.tanh_function(y)
        else:
           return self.reLU(y)
           
    def ActivationMethodDerivative(self,x,method):
        y=np.array(x,dtype=np.float32)
        if method=="sigmoid":
           return self.tanh_derivative_function(y)
        elif method=="tanh":
           return self.tanh_derivative_function(y)
        else:
           return self.reLU_derivative(y)    
           
    #Activation Functions
    def sigmoid(self, x):
        exponent = np.exp(-x)
        denominator = 1 + exponent
        sigmoid_value = 1 / denominator
        return sigmoid_value

    def reLU(self, x):
        return np.maximum(0, x)

    def tanh_function(self, x):
        e_x = np.exp(np.array(x,dtype=np.float32))
        e_minus_x = np.exp(np.array(-x,dtype=np.float32))
        numerator = e_x - e_minus_x
        denominator = e_x + e_minus_x
        tanh_value = numerator / denominator
        return tanh_value

    #Activation Function Derivatives
    def sigmoid_derivative(self, x):
        sigmoid_x = self.sigmoid(x)
        sigmoid_derivative = sigmoid_x * (1 - sigmoid_x)
        return sigmoid_derivative

    def reLU_derivative(self, x):
        return (x > 0).astype(float)

    def tanh_derivative_function(self, x):
        tanh_x = self.tanh_function(x)
        tanh_derivative = 1 - np.power(tanh_x, 2)
        return tanh_derivative
            
    def max_clip(delta_val):
        if delta_val.max() > self.maxClip:   
                delta_val[delta_val > self.maxClip] = self.maxClip

    def min_clip(delta_val):
        if delta_val.min() < self.minClip:
                delta_val[delta_val < self.minClip] = self.minClip
                
                
 
class DataPreprocessor:
    def __init__(self):
        self.city_encoder = None
        self.day_encoder = None
        self.year_encoder = None
        if Output_activation_function=="tanh":
            self.year_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.temp_scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.year_scaler = MinMaxScaler(feature_range=(0, 1))
            self.temp_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fit(self, data):
        # One-hot encoding for categorical columns
        self.city_encoder = pd.get_dummies(data['City'], prefix='City', drop_first=True)
        self.day_encoder = pd.get_dummies(data['Day'], prefix='Day', drop_first=True)
        self.month_encoder = pd.get_dummies(data['Month'], prefix='Month', drop_first=True)

        # Fit scaler for temperature normalization
        self.year_scaler.fit(data[['Year']])
        self.temp_scaler.fit(data[['AvgTemperature']])
        
    def transform(self, data):
        # Apply one-hot encoding
        city_encoded = self.city_encoder.reindex(columns=self.city_encoder.columns, fill_value=0)
        day_encoded = self.day_encoder.reindex(columns=self.day_encoder.columns, fill_value=0)
        month_encoded = self.month_encoder.reindex(columns=self.month_encoder.columns, fill_value=0)

        # Normalize temperature
        normalized_temp = self.temp_scaler.transform(data[['AvgTemperature']])
        
        # Normalize year
        normalized_year = self.year_scaler.transform(data[['Year']])

        # Concatenate encoded features and normalized temperature
        processed_data = pd.concat([ month_encoded, pd.DataFrame(normalized_temp, columns=['Norm_AvgTemperature'])], axis=1)

        return processed_data

def ModifyData(path):
    data = pd.read_csv(path)
    mean_temp = data[data['AvgTemperature']>=0]['AvgTemperature'].mean()
    data.loc[data['AvgTemperature']<0, 'AvgTemperature'] = mean_temp
    preprocessor = DataPreprocessor()
    preprocessor.fit(data)
    return preprocessor.transform(data)


trainData=ModifyData(r"C:\Users\gonap\OneDrive\Desktop\train.csv")
validationData=ModifyData(r"C:\Users\gonap\OneDrive\Desktop\test.csv")


meteoroRNN=MeteoroRNN(learningRate,epoch,sequence_length,hiddenDim,minClip,maxClip,trainData.shape[1])
meteoroRNN.train(trainData.values,trainData['Norm_AvgTemperature'].values,validationData.values,validationData['Norm_AvgTemperature'].values)
meteoroRNN.predict(validationData.values,validationData['Norm_AvgTemperature'].values)



    

