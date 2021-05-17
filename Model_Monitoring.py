#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import collections
import pandas as pd

class Stability_Metrics:
    

    def __init__(self, df_dev, df_rev, feat_col_names, feat_data_types, output_col_names, buckettype='quantiles', buckets=10, axis=0):
        self.df_dev = df_dev
        self.df_rev = df_rev
        self.feat_col_names = feat_col_names
        self.feat_data_types = feat_data_types
        self.output_col_names = output_col_names
        self.buckettype = buckettype
        self.buckets = buckets
        self.axis = axis
        print("No. of feature columns in the dataset: "+str(len(self.feat_col_names)))
        print("No. of output columns in the dataset: "+str(len(self.output_col_names)))
        self.data_conversion()
        self.calculate_metrics()

    def data_conversion(self):  
        """ Converts the development and review datasets to numpy arrays required for calculations """
        categorical_feature_names = []
        numerical_feature_names = []
        
        try:
            # Create a mapping between the feature data types and their column names 
            key_feat = dict(zip(self.feat_col_names, self.feat_data_types)) 
            
            # Ensure columns with NaN values above/equal to a threshold are removed (default = 1)
            thresh = 0.75
            for name in self.feat_col_names: 
                if((self.df_dev[name].isnull().sum() >= thresh*len(self.df_dev)) | (self.df_rev[name].isnull().sum() >= thresh*len(self.df_rev))):
                    key_feat.pop(name)
                else:
                    pass
            self.feat_col_names = []
            self.feat_data_types = []
            items = key_feat.items() 
            for item in items: 
                self.feat_col_names.append(item[0]), self.feat_data_types.append(item[1]) 

            # Create separate lists of numerical (int64 and float64) and categorical (string) column names
            for i in self.feat_col_names:
                if ((key_feat[i] == 'int64') | (key_feat[i] == 'float64')):
                    numerical_feature_names.append(i)
                elif ((key_feat[i] == 'object') | (key_feat[i] == 'bool') | (key_feat[i] == 'O')):
                    categorical_feature_names.append(i)
            
            # Convert user-specified numerical columns to numeric data type (to ensure consistency)
            for num in numerical_feature_names:
                self.df_dev[num] = pd.to_numeric(self.df_dev[num], errors = 'coerce')
                self.df_rev[num] = pd.to_numeric(self.df_rev[num], errors = 'coerce')
            
            # Convert user-specified non-numerical columns to categorical data type (to ensure consistency)
            for cat in categorical_feature_names:
                self.df_dev[cat] = self.df_dev[cat].astype('category')
                self.df_rev[cat] = self.df_rev[cat].astype('category')
            
            # Create temporary numpy arrays containing values corresponding to numerical and categorical columns (dev and rev)
            dev_feat_numerical_temp = self.df_dev[numerical_feature_names].values
            dev_feat_categorical_temp = self.df_dev[categorical_feature_names].values
            rev_feat_numerical_temp = self.df_rev[numerical_feature_names].values
            rev_feat_categorical_temp = self.df_rev[categorical_feature_names].values
            
            # Create temporary numpy arrays containing values corresponding to output columns (they will be numerical)
            for i in self.output_col_names:
                dev_output_temp = self.df_dev[self.output_col_names].values 
                rev_output_temp = self.df_rev[self.output_col_names].values
           
            # Creating a concatenated numpy array 
            self.feat_col_names_new = numerical_feature_names + categorical_feature_names
            self.all_col_names_new = numerical_feature_names + categorical_feature_names + self.output_col_names
            np_concat_dev_temp = np.concatenate((dev_feat_numerical_temp, dev_feat_categorical_temp, dev_output_temp), axis=1)
            np_concat_rev_temp = np.concatenate((rev_feat_numerical_temp, rev_feat_categorical_temp, rev_output_temp), axis=1)           

            # Converting to a dataframe to drop rows containing any NaN 
            df_concat_dev_temp = pd.DataFrame(np_concat_dev_temp)
            df_concat_rev_temp = pd.DataFrame(np_concat_rev_temp)
            df_concat_dev_temp = df_concat_dev_temp.dropna()
            df_concat_rev_temp = df_concat_rev_temp.dropna()
            
            len_num = len(numerical_feature_names)
            len_cat = len(categorical_feature_names)
            len_output = len(self.output_col_names)
            
            # Creating final numpy arrays (to be used for stability metrics calculation)
            self.dev_feat_numerical = df_concat_dev_temp.values[:,:len_num]
            self.rev_feat_numerical = df_concat_rev_temp.values[:,:len_num]
            self.dev_feat_categorical = df_concat_dev_temp.values[:,len_num:len_num+len_cat]
            self.rev_feat_categorical = df_concat_rev_temp.values[:,len_num:len_num+len_cat]
            self.dev_output = df_concat_dev_temp.values[:,len_num+len_cat:len_num+len_cat+len_output]
            self.rev_output = df_concat_rev_temp.values[:,len_num+len_cat:len_num+len_cat+len_output]
            print("\nData Conversion and Cleaning Successful!!\n")
            
        except Exception as e:
            print("An exception occurred during data conversion:\n")
            print("Conversion not successful:", e)
            print("Please check if the inputs are specified in the correct format")

    def calculate_metrics(self):
        '''Calculates the PSI, CSI and DI metrics and returns a tuple of 3 dictionaries containing PSI, CSI and DI for feature and output columns'''
        
        def calculate_psi(dev_output, rev_output):
            '''Calculate the PSI (population stability index) across multiple columns (regression/multi-class/multilabel output)
            Input variables:
               dev_output: numpy array of original response values (t = 0)
               rev_output: numpy array of new response values, same size as dev (t = t)
            Returns:
               psi_values: ndarray of psi values for each class/column
            '''
            psi_values = []
            
            def psi_single_column(dev_column, rev_column):
                '''Calculate the PSI for a single class/columns
                Input variables:
                   dev_column: numpy array of original values
                   rev_column: numpy array of new values, same size as dev
                Returns:
                   psi_value: calculated PSI value
                '''

                def scaling (input, min, max):
                    input += -(min(input))
                    input /= max(input) / (max - min)
                    input += min
                    return input


                breakpoints = np.arange(0, self.buckets + 1) / (self.buckets) * 100

                if self.buckettype == 'bins':
                    breakpoints = scaling(breakpoints, np.min(dev_column), np.max(dev_column))
                elif self.buckettype == 'quantiles':
                    breakpoints = np.stack([np.percentile(dev_column, b) for b in breakpoints])
                
                dev_percents = np.histogram(dev_column, breakpoints)[0] / len(dev_column)      
                rev_percents = np.histogram(rev_column, breakpoints)[0] / len(rev_column)
                
                def sub_psi(dev_perc, rev_perc):
                    '''Calculate the actual PSI value from comparing the values.
                       Update the value to a very small number if equal to zero
                    '''
                    if dev_perc == 0:
                        dev_perc = 0.00001
                    if rev_perc == 0:
                        rev_perc = 0.00001

                    value = (dev_perc - rev_perc) * np.log(np.abs(dev_perc / rev_perc))
                    return(value)
                psi_value = sum(sub_psi(dev_percents[i], rev_percents[i]) for i in range(0, len(dev_percents)))

                return(psi_value)

            if self.axis == 0:
                transpose = 1    
            else:
                transpose = 0

            if len(dev_output.shape) == 1:
                psi_values = np.empty(len(dev_output.shape))
            else:
                psi_values = np.empty(dev_output.shape[transpose])

            for i in range(0, len(psi_values)):
                if len(psi_values) == 1:
                    psi_values = []
                    psi_values.append(psi_single_column(dev_output, rev_output))
                elif self.axis == 0:
                    psi_values[i] = psi_single_column(dev_output[:,i], rev_output[:,i])
                elif self.axis == 1:
                    psi_values[i] = psi_single_column(dev_output[i,:], rev_output[i,:])
            return(psi_values)
        
        def calculate_csi(dev_feat_numerical, rev_feat_numerical, dev_feat_categorical, rev_feat_categorical):
            '''Calculate the CSI (chraceteristic stability index) across multiple features
            Input variables:
               dev_feat_numerical: numpy array of original numerical feature values (t = 0)
               rev_feat_numerical: numpy array of new numerical feature values (t = t)
               dev_feat_categorical: numpy array of original categorical feature values (t = 0)
               rev_feat_categorical: numpy array of new categorical feature values (t = t)

            Returns:
               csi_values: list of csi values for each feature column
            '''
            
            def calculate_csi_categorical(dev_feat_categorical, rev_feat_categorical):
                '''Calculate the CSI (chraceteristic stability index) for categorical features
                Input variables:
                   dev_feat_categorical: numpy array of original categorical feature values (t = 0)
                   rev_feat_categorical: numpy array of new categorical feature values (t = t)

                Returns:
                   csi_categorical: list of csi values for each categorical feature column
                '''
                
                def csi_single_column_categorical(dev_column, rev_column):
                    '''Calculate the CSI (chraceteristic stability index) for a single categorical feature
                    Input variables:
                       dev_column: 1-D numpy array of original categorical feature value (t = 0)
                       rev_column: 1-D numpy array of new categorical feature value (t = t)

                    Returns:
                       csi_value: csi value for a single categorical feature column
                    '''
                    
                    # Lists containing the fraction/percentage of common categorical variable in dev and rev datasets 
                    dist_dev = []
                    dist_rev = []
                    
                    # Dictionary containing unique elements and their counts
                    dict_dev = collections.Counter(dev_column)
                    dict_rev = collections.Counter(rev_column)
                    
                    # For common variables in dev and rev, calculate CSI value for a single column
                    for item in dict_dev.keys():
                        if item in dict_rev.keys():
                            dist_dev.append(dict_dev[item]*1.0/len(dev_column))
                            dist_rev.append(dict_rev[item]*1.0/len(rev_column))
                    csi_value = sum(1.0*(dist_rev[j]-dist_dev[j])*np.log(dist_rev[j]/dist_dev[j]) for j in range(0,len(dist_dev)))
                    return(csi_value)
                
                # Use above helper function to get CSI values for all categorical feature columns and return them as a list
                csi_categorical = []
                for i in range(0, dev_feat_categorical.shape[1]):
                    csi_categorical.append(csi_single_column_categorical(dev_feat_categorical[:,i],rev_feat_categorical[:,i]))
                    
                return(csi_categorical)
            
            # For numerical columns, the same PSI function, defined above, can be used
            if(dev_feat_numerical.shape[1]!=1):
                csi_numerical = []
                csi_numerical = calculate_psi(dev_feat_numerical, rev_feat_numerical).tolist()
            else:
                csi_numerical = calculate_psi(dev_feat_numerical, rev_feat_numerical)
                
            # For categorical columns, using the helper function
            csi_categorical = calculate_csi_categorical(dev_feat_categorical, rev_feat_categorical)
            csi_values = csi_numerical + csi_categorical
            return(csi_values)
            
        def calculate_di(dev_feat_numerical, rev_feat_numerical, dev_feat_categorical, rev_feat_categorical): 
            '''Calculate the DI (chraceteristic stability index) across multiple features
            Input variables:
               dev_feat_numerical: numpy array of original numerical feature values (t = 0)
               rev_feat_numerical: numpy array of new numerical feature values (t = t)
               dev_feat_categorical: numpy array of original categorical feature values (t = 0)
               rev_feat_categorical: numpy array of new categorical feature values (t = t)

            Returns:
               di_values: list of di values for each feature column
            '''            
            
            def calculate_di_numerical(dev_feat_numerical, rev_feat_numerical):
                '''Calculate the DI (divergence index) across numerical features
                Input variables:
                   dev_feat_numerical: numpy array of original numerical feature values (t = 0)
                   rev_feat_numerical: numpy array of new numerical feature values (t = t)

                Returns:
                   di_numerical: list of di values for numerical feature columns
                '''            
                def di_single_column_numerical(dev_column, rev_column):
                    tol = 0.00001
                    if(dev_column.mean()!=0):
                        di_value = abs(1.0*(rev_column.mean() - dev_column.mean())/dev_column.mean()) 
                    else:
                        di_value = abs(1.0*(rev_column.mean() - dev_column.mean())/tol)
                    return(di_value)
                
                di_numerical = []
                for i in range(0, dev_feat_numerical.shape[1]):
                    di_numerical.append(di_single_column_numerical(dev_feat_numerical[:,i],rev_feat_numerical[:,i]))

                return(di_numerical)
                
            def calculate_di_categorical(dev_feat_categorical, rev_feat_categorical):
                '''Calculate the DI (divergence index) across categorical features
                Input variables:
                   dev_feat_categorical: numpy array of original numerical feature values (t = 0)
                   rev_feat_categorical: numpy array of new numerical feature values (t = t)

                Returns:
                   di_categorical: list of di values for categorical feature columns
                '''            
                def di_single_column_categorical(dev_column, rev_column):
                    
                    # Lists containing the fraction/percentage of common categorical variable in dev and rev datasets 
                    dist_dev = [] 
                    dist_rev = []
                    
                    # Dictionary containing unique elements and their counts
                    dict_dev = collections.Counter(dev_column)
                    dict_rev = collections.Counter(rev_column)
                    # For common variables in dev and rev, calculate CSI value for a single column
                    intersection = [value for value in dict_dev.keys() if value in dict_rev.keys()] 
                    union = list(set().union(list(dict_dev.keys()), list(dict_rev.keys()))) 
                    di_value = 1 - (len(intersection)/len(union))*1.0
                    return(di_value)
                
                di_categorical = []

                for i in range(0, dev_feat_categorical.shape[1]):
                    di_categorical.append(di_single_column_categorical(dev_feat_categorical[:,i],rev_feat_categorical[:,i]))
                
                return(di_categorical)
    
            di_numerical = calculate_di_numerical(dev_feat_numerical, rev_feat_numerical)
            di_categorical = calculate_di_categorical(dev_feat_categorical, rev_feat_categorical)
            di_values = di_numerical + di_categorical
            return(di_values)
        
        try:
            psi_values = calculate_psi(self.dev_output, self.rev_output)
            csi_values = calculate_csi(self.dev_feat_numerical, self.rev_feat_numerical, self.dev_feat_categorical, self.rev_feat_categorical)
            di_values = calculate_di(self.dev_feat_numerical, self.rev_feat_numerical, self.dev_feat_categorical, self.rev_feat_categorical)
            psi = {self.output_col_names[i]: psi_values[i] for i in range(len(self.output_col_names))} 
            csi = {self.feat_col_names_new[i]: csi_values[i] for i in range(len(self.feat_col_names_new))}
            di = {self.feat_col_names_new[i]: di_values[i] for i in range(len(self.feat_col_names_new))}
            print("\nThe computed Stability Metrics are listed below:")
            print("\nPSI values for " + str(len(psi_values)) + " output columns: ", psi)
            print("\nCSI values for " + str(len(csi_values)) + " feature columns: ", csi)
            print("\nDI values for " + str(len(csi_values)) + " feature columns: ", di)
            return (psi,csi,di)
        
        except Exception as e:
            print("\nAn exception occurred during calculation of Stability Metrics:")
            print(e)


# In[ ]:




