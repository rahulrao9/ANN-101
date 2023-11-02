
# to run use - python3 decisiontree_test.py --decisiontree_main

import torch

"""Calculate the entropy of the entire dataset"""
# input:tensor
# output:int/float
def get_entropy_of_dataset(tensor:torch.Tensor):
    target_column = tensor[:,-1]
    outputs, value_counts = target_column.unique(return_counts=True)
    no_instances = target_column.size(0)
    probability = value_counts.float() / no_instances
    entropy = -torch.sum(probability * torch.log2(probability))
    
    return entropy.item()


"""Return avg_info of the attribute provided as parameter"""
# input:tensor,attribute number 
# output:int/float
def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):

    target_column = tensor[:, attribute]
    instance, value_counts = target_column.unique(return_counts=True)
    no_instance = target_column.size(0)
    avg_conditional_entropy = 0

    for value in instance:
        subset_indices = (target_column == value)
        subset_data = tensor[subset_indices]
        
        conditional_entropy = get_entropy_of_dataset(subset_data)
        avg_conditional_entropy += (subset_data.size(0) / no_instance) * conditional_entropy # summation {|Sv|*Entropy(Sv)} / |S|
    
    return avg_conditional_entropy


"""Return Information Gain of the attribute provided as parameter"""
# input:tensor,attribute number
# output:int/float
def get_information_gain(tensor:torch.Tensor, attribute:int):
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_attribute_entropy = get_avg_info_of_attribute(tensor,attribute)
    info_gain = dataset_entropy - avg_attribute_entropy
    return info_gain



# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    columns = tensor.size(1)
    ig_dict = {}
    max_ig = 0
    return_tuple = (ig_dict,max_ig)

    for column in range(columns):
        
        ig = get_information_gain(tensor,column)
        ig_dict[column] = ig
        
        if ig > max_ig:
            max_ig = ig

    return return_tuple        

