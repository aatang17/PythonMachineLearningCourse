#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#Single hinge loiss

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    feature_vector = np.array(feature_vector)
    theta , theta_0 = np.array(theta), np.array(theta_0)

    loss = label * (np.dot(feature_vector, theta) + theta_0)
    return max(0, 1- loss)
    raise NotImplementedError


# In[3]:


#Complete Hinge Loss

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    feature_matrix = np.array(feature_matrix)
    theta = np.array(theta)
    
    total_hinge_loss = 0
    for i in range(len(feature_matrix)):
        loss = labels[i] * (np.dot(feature_matrix[i], theta ) + theta_0)
        total_loss = max(0, 1 - loss)
        total_hinge_loss += total_loss
    return total_hinge_loss / len(feature_matrix)
    raise NotImplementedError


# In[4]:


#Single step perceptron

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    feature_vector = np.array(feature_vector)
    current_theta = np.array(current_theta)
    total = label * (np.dot(current_theta, feature_vector) + current_theta_0)
    if total <= 0:
        updated_theta = current_theta  + (label * feature_vector )
        updated_theta_one = current_theta_0 + label
        return (updated_theta, updated_theta_one)
    return (current_theta, current_theta_0)
    raise NotImplementedError


# In[5]:


#Full Perceptron

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    nsamples = feature_matrix.shape[0]
    feature = feature_matrix.shape[1]
    theta = np.zeros(feature)
    theta_one = 0.0
   
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            theta, theta_one = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_one)
            
    return (theta, theta_one)
    raise NotImplementedError


# In[6]:


#Average Perceptron

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    feature = feature_matrix.shape[1]
    theta = np.zeros(feature)
    theta_one = 0.0
    theta_one_sum = 0.0
    theta_sum = np.zeros(feature)
   
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            theta, theta_one = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_one)
            theta_sum += theta
            theta_one_sum +=theta_one
            
    return ((theta_sum/ (T * feature_matrix.shape[0])), (theta_one_sum/(T*feature_matrix.shape[0])))
    raise NotImplementedError


# In[7]:


#Single Step Pegasos

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <=1:
        return ((1- (L *eta))* current_theta + (eta* label * feature_vector), (current_theta_0) + (eta * label))
        
    return ((1- (L*eta)) * current_theta, current_theta_0)
    raise NotImplementedError


# In[ ]:


#Full Pegasos

def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    feature = feature_matrix.shape[1]
    theta = np.zeros(feature)
    theta_one = 0.0
    c = 0
   
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            c +=1
            eta = 1/np.sqrt(c)
            theta, theta_one = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_one)
            
    return (theta, theta_one)
    raise NotImplementedError

