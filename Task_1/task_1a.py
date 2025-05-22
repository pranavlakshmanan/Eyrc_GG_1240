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
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
import pandas
import torch.nn
#import numpy
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################


##############################################################

def data_preprocessing(task_1a_dataframe):
    ''' 
    Purpose:
    ---
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that 
    there are features in the csv file whose values are textual (eg: Industry, 
    Education Level etc)These features might be required for training the model
    but can not be given directly as strings for training. Hence this function 
    should return encoded dataframe in which all the textual features are 
    numerically labeled.

    Input Arguments:
    ---
    `task_1a_dataframe`: [Dataframe]
                                              Pandas dataframe read from the provided dataset 	

    Returns:
    ---
    `encoded_dataframe` : [ Dataframe ]
                                              Pandas dataframe that has all the features mapped to 
                                              numbers starting from zero

    Example call:
    ---
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    '''

    #################	ADD YOUR CODE HERE	##################
    raw_df = task_1a_dataframe
    Gender = {0: "Female", 1: "Male"}
    raw_df["Gender"].replace(Gender[0], 0, inplace=True)
    raw_df["Gender"].replace(Gender[1], 1, inplace=True)
    # Converting EverBeched to boolean 0 or 1
    EverBenched = {0: "No", 1: "Yes"}
    raw_df["EverBenched"].replace(EverBenched[0], 0, inplace=True)
    raw_df["EverBenched"].replace(EverBenched[1], 1, inplace=True)
    # Converting 4 digit Year to 2 digit
    raw_df["JoiningYear"] = raw_df["JoiningYear"] % 100
    # Assigning Bangalore as 0, New Delhi as 1, Pune as 2
    City = {0: "Bangalore", 1: "New Delhi", 2: "Pune"}
    raw_df["City"].replace(City[0], 0, inplace=True)
    raw_df["City"].replace(City[1], 1, inplace=True)
    raw_df["City"].replace(City[2], 2, inplace=True)

    #encoder = preprocessing.OneHotEncoder(dtype=int, sparse_output=False)
    # encoded_cities = pandas.DataFrame(
    #    encoder.fit_transform(raw_df[["City"]]),
    #    columns=encoder.get_feature_names_out(["City"]),
    # )

    # Assigning for Education
    Education = {0: "Bachelors", 1: "Masters", 2: "PHD"}
    raw_df["Education"].replace(Education[0], 0, inplace=True)
    raw_df["Education"].replace(Education[1], 1, inplace=True)
    raw_df["Education"].replace(Education[2], 2, inplace=True)

    # encoded_edu = pandas.DataFrame(
    #    encoder.fit_transform(raw_df[["Education"]]),
    #    columns=encoder.get_feature_names_out(["Education"]),
    # )
    leaveornot = {0: "0", 1: "1"}
    raw_df["LeaveOrNot"].replace(leaveornot[0], 0, inplace=True)
    raw_df["LeaveOrNot"].replace(leaveornot[1], 1, inplace=True)
    #raw_df.drop(["City", "Education"], axis=1, inplace=True)
    #raw_df = pandas.concat((raw_df, encoded_cities, encoded_edu), axis=1)
    # print(raw_df)
    # label_encoder = preprocessing.LabelEncoder()
    # for column in task_1a_dataframe.columns:

    #     if raw_df[column].dtype == 'O':
    #         raw_df[column] = label_encoder.fit_transform(raw_df[column])

    encoded_dataframe = raw_df
    ##########################################################

    return encoded_dataframe


def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second 
    item is the target label

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                                            Pandas dataframe that has all the features mapped to 
                                            numbers starting from zero

    Returns:
    ---
    `features_and_targets` : [ list ]
                                                    python list in which the first item is the 
                                                    selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    '''

    #################	ADD YOUR CODE HERE	##################
    features_and_targets = []
    target = encoded_dataframe["LeaveOrNot"]
    unselected_features = ["LeaveOrNot"]
    features = encoded_dataframe.copy()
    features.drop(unselected_features, axis=1, inplace=True)
    features_and_targets = [features, target]
    # print("features", features.shape)
    # print("Targets", target.shape)
    ##########################################################

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
    Iterable_dataset = []
    features_and_targets[0] = features_and_targets[0].values
    features_and_targets[1] = features_and_targets[1].values
    tensors_and_iterable_training_data = []
    x = features_and_targets[0]
    y = features_and_targets[1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.5, shuffle=True
    )
    X_train_tensor = torch.Tensor(x_train)
    X_test_tensor =  torch.Tensor(x_test)
    Y_train_tensor = torch.LongTensor(y_train)
    Y_test_tensor =  torch.LongTensor(y_test)
    tensors_and_iterable_training_data = [
        X_train_tensor,
        X_test_tensor,
        Y_train_tensor,
        Y_test_tensor,
        Iterable_dataset
    ]
    ##########################################################

    return tensors_and_iterable_training_data


class Salary_Predictor(torch.nn.Module):
    '''
    Purpose:
    ---
    The architecture and behavior of your neural network model will be
    defined within this class that inherits from nn.Module. Here you
    also need to specify how the input data is processed through the layers. 
    It defines the sequence of operations that transform the input data into 
    the predicted output. When an instance of this class is created and data
    is passed through it, the `forward` method is automatically called, and 
    the output is the prediction of the model based on the input data.

    Returns:
    ---
    `predicted_output` : Predicted output for the given input data
    '''

    def __init__(self):
        super(Salary_Predictor, self).__init__()
        """
		Define the type and number of layers
		"""
        #######	ADD YOUR CODE HERE	#######
        neurons = 12
        # in_features = 12
        in_features = 8
        layer1_neurons = neurons
        layer2_neurons = neurons
        out_features = 1
        self.layer1 = torch.nn.Linear(in_features, layer1_neurons, bias=False)
        self.layer2 = torch.nn.Linear(layer1_neurons, layer2_neurons)
        self.out = torch.nn.Linear(layer2_neurons, out_features)
        ###################################

    def forward(self, x):
        """
        Define the activation functions
        """
        #######	ADD YOUR CODE HERE	#######
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        #predicted_output = torch.nn.functional.softmax(self.out(x),dtype=float)
        predicted_output = self.out(x)
        ###################################
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
    loss_function = torch.nn.CrossEntropyLoss()
    ##########################################################

    return loss_function


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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''

    #################	ADD YOUR CODE HERE	##################
    number_of_epochs = 120
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
    torch.set_printoptions(profile="full")
    X_train = tensors_and_iterable_training_data[0]
    Y_train = tensors_and_iterable_training_data[2]
    print(X_train.shape)
    losses = []
    for i in range(number_of_epochs):
        optimizer.zero_grad()
        y_pred = model.forward(X_train)
        loss = loss_function(y_pred, Y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())
        if i % 10 == 0:
            print(f"Epoch: {i} and loss: {100-loss}")

    plt.plot(range(number_of_epochs), losses)
    plt.ylabel("loss/error")
    plt.xlabel("Epoch")
    trained_model = model
    #print("Trained Model", trained_model)

    ##########################################################

    return trained_model


def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                                                                     and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''
    #################	ADD YOUR CODE HERE	##################
    correct = 0
    total = 0
    X_test = tensors_and_iterable_training_data[1]
    Y_test = tensors_and_iterable_training_data[3]
    #loss_function = model_loss_function()
    trained_model.eval()
    with torch.no_grad():  # Basically turn off back propogation
        # X_test are features from our test set, y_eval will be predictions
        y_eval = trained_model.forward(X_test)
        loss = loss_function(y_eval, Y_test)  # Find the loss or error
    model_accuracy = 100-loss
    #     _, predicted = torch.max(y_eval.data, 1)
    #     total += Y_test.size(0)
    #     correct += (predicted == Y_test).sum().item()

    # model_accuracy = 100.0 * correct / total
    ##########################################################

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
    task_1a_dataframe = pandas.read_csv(
        "/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task 1/Task_1A/task_1a_dataset.csv")

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
    model_accuracy = validation_function(
        trained_model, tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(
        model, (x)), "task_1a_trained_model.pth")
