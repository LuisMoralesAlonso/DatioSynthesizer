from enum import Enum

class DataType(Enum):
    INTEGER = 'Integer'
    FLOAT = 'Float'
    STRING = 'String'
    DATETIME = 'DateTime'
    SOCIAL_SECURITY_NUMBER = 'SocialSecurityNumber'

#Categorical threshold
threshold_of_categorical_variable=20

#Bins length for numerical variables
histogram_size=20

#A parameter in differential privacy
epsilon = 0.01

#Number of parents in Bayesian Network
k=0

#Aditional Null values
null_values=None