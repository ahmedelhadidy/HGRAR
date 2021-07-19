import numpy as np

class OneHotEncoder:
    def __init__(self, classes, off_on_values=[0,1]):
        '''
        One=Hot encoder utility
        :param classes: the unique list of all classes/ labels
        '''
        self.__classes = classes
        assert np.array(off_on_values).shape
        self.__encoding_values = np.array(off_on_values)

    def encode( self, values ):
        '''
        convert list of labels to one hot encoding according to the classes order given in the constructor
        :param values: iterator of values to encode
        :return: darray of shape (len(values), len(classes)) each row encoding each item from values
        '''
        encoded_values = np.full(fill_value=self.__encoding_values[0], shape=(len(values), len(self.__classes)))
        for row_index, value in enumerate(values):
            encoded_values[row_index][self.__classes.index(value)]=self.__encoding_values[1]
        return encoded_values

    def classes( self ):
        return list(self.__classes)

    def decode( self, encoded_rows ):
        '''
        decode list of one-hot encoded rows to its corresponding classes
        :param encoded_rows: list of one-hot encoded
        :return: list of classes
        '''
        ret = []
        for ind,row in enumerate(encoded_rows):
            cls_index = np.where(row == self.__encoding_values[1])[0][0]
            ret.insert(ind,self.__classes[cls_index])
        return ret