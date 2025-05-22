'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas 
import torch
import numpy as np
###################### Additional Imports ####################
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################




def data_preprocessing(task_1a_dataframe):
    df=task_1a_dataframe
    scaler=StandardScaler()
    label_encoder = preprocessing.LabelEncoder()
    for column in df.columns:

        if df[column].dtype == 'O':
            df[column] = label_encoder.fit_transform(df[column])
    for column in df.columns:
     if column != 'TargetColumnToExclude':  # You can exclude specific columns if needed
        df[column] = scaler.fit_transform(df[[column]])
        
    return task_1a_dataframe
    

        

	

def identify_features_and_targets(encoded_dataframe):

    features_and_targets = []
    target = encoded_dataframe["LeaveOrNot"]
    unselected_features = ["LeaveOrNot"]
    features = encoded_dataframe.copy()
    features.drop(unselected_features, axis=1, inplace=True)
    features_and_targets = [features, target]

    return features_and_targets


def load_as_tensors(features_and_targets):
    ''' 
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to iterable dataset
    object.

    Input Arguments:
    ---
    `features_and targets` : [ list ]
                                                    python list in which the first item is the 
                                                    selected features and second item is the target label

    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                                                                    Items:
                                                                                    [0]: X_train_tensor: Training features loaded into Pytorch array
                                                                                    [1]: X_test_tensor: Feature tensors in validation data
                                                                                    [2]: y_train_tensor: Training labels as Pytorch tensor
                                                                                    [3]: y_test_tensor: Target labels as tensor in validation data
                                                                                    [4]: Iterable dataset object and iterating over it in 
                                                                                             batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

    #################	ADD YOUR CODE HERE	##################
    tensors_and_iterable_training_data = []
    features_and_targets[0] = features_and_targets[0].values
    features_and_targets[1] = features_and_targets[1].values
    x = features_and_targets[0]
    y = features_and_targets[1]
    x_train, x_test, y_train, y_test =train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    X_train_tensor = torch.Tensor(x_train)
    X_test_tensor = torch.Tensor(x_test)
    Y_train_tensor = torch.FloatTensor(y_train).reshape(-1,1)
    Y_test_tensor = torch.FloatTensor(y_test).reshape(-1,1)
    iterable_dataset=TensorDataset(X_train_tensor)
    tensors_and_iterable_training_data = [
        X_train_tensor,
        X_test_tensor,
        Y_train_tensor,
        Y_test_tensor,
        iterable_dataset
    ]
    


    ##########################################################

    return tensors_and_iterable_training_data


class Salary_Predictor(torch.nn.Module):
	
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		
	
	
        #######	ADD YOUR CODE HERE	#######
		self.layer1 = torch.nn.Linear(in_features=8,out_features=12)
		self.layer2 = torch.nn.Linear(in_features=12,out_features=8)
		self.out_layer1 = torch.nn.Linear(in_features=8,out_features=1)
		
	def forward(self, x):
		x = torch.nn.functional.sigmoid(self.layer1(x))
		x = torch.nn.functional.sigmoid(self.layer2(x))
		x = torch.nn.functional.sigmoid(self.out_layer1(x))
		
		
		
		predicted_output = x
		return predicted_output



def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures 
    how well the predictions of a model match the actual target values 
    in training data.

    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    #################	ADD YOUR CODE HERE	##################
    loss_function = torch.nn.BCELoss()
    ##########################################################

    return loss_function

loss_function = model_loss_function()


def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible 
    for updating the parameters (weights and biases) in a way that 
    minimizes the loss function.

    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class

    Returns:
    ---
    `optimizer`: Pre-defined optimizer from Pytorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    #################	ADD YOUR CODE HERE	##################
    optimizer = torch.optim.Adam(model.parameters(),lr=0.334)

    ##########################################################

    return optimizer



def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call
    ---
    number_of_epochs = model_number_of_epochs()
    '''

    #################	ADD YOUR CODE HERE	##################
    number_of_epochs = 100
    ##########################################################

    return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                                                                     and iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    '''
    #################	ADD YOUR CODE HERE	##################
    X_train = tensors_and_iterable_training_data[0]
    Y_train = tensors_and_iterable_training_data[2]
    losses = []
    for i in range(number_of_epochs):
        y_pred = model(X_train)
        loss = loss_function(y_pred, Y_train)
        losses.append(loss.detach().numpy())

        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # plt.plot(range(number_of_epochs), losses)
    # plt.ylabel("loss/error")
    # plt.xlabel("Epoch")
    # trained_model = torch.save(model.forward, "/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task 1/Task_1A/model.pt")
    trained_model = model
    ##########################################################

    return trained_model


def validation_function(trained_model, tensors_and_iterable_training_data):
	X_test = tensors_and_iterable_training_data[1]
	Y_test = tensors_and_iterable_training_data[3]
	
	loss_function = torch.nn.BCELoss()
	model=trained_model
	with torch.no_grad():
		y_eval = model.forward(X_test)
		loss = loss_function(y_eval, Y_test)  # Find the loss or error
		model_accuracy = loss*100
		return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")