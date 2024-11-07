import numpy as np
import pandas


###### Now lets write distribution functions #######
class score():
    def __init__(self,series, primary,rate):
        """
        Initializes score class with a pandas series as argument. Converts every value into
        float and stores number of classes and classes in sorted fashion.

        Args:
            series : pandas series

        Returns:
            _type_: _description_
        """
        assert isinstance(self.series, pandas.core.series.Series),'InputError: Input is not pandas series'
        self.primary = primary
        self.rate  = rate
        self.series = series
        self.classes = (series.apply(lambda x: float(x))).unique()
        self.classes.sort()
        self.num_classes = len(self.classes)
        self.finaldistribution = self.__FinalRateDistribution__()
        return None
        

    def __RatingDistribution__(self, rate):
        assert rate in range(1,11), f'InputError: {rate} is not within the range of 1 to 10'
        """
        The rating distribution uses the idea of 10 points distribution 
        around primary choice with a max_limit of 2 for non chosen
        Return np.array
        """
        if rate == 10:
            return np.array([10])
        if rate == 1:
            return np.ones(10)
        arr = [rate]
        flag = 1
        while sum(arr)<10:
            if flag == 1:
                if arr[0] >=2 or (arr[0] == rate):
                    ### insert and give 1
                    arr.insert(1)
                elif arr[0]==1:
                    arr[0]+=1
            if flag == -1:    
                if arr[-1]== rate or (arr[-1]>=2):
                    ##append and give 1
                    arr.append(1)
                elif arr[-1] == 1:
                    arr[-1]+=1
            flag *= -1
        
        return np.array(arr)
    
    def __FinalRateDistribution__(self):
        """
        Still need to discuss,
        if law of rating 10 is required, and if does, where to distribute
        edge score to -- should it be reversed or added to the tails

        Args:
            primary (int): Must be within num_classes's index
            num_classes (int): must be greater than 1
            rate (int): Must be within 1-10
        """

        distribution = self.__RatingDistribution__(self.rate)
        d = distribution
        a = np.append(d,np.zeros(9))
        a = np.append(np.zeros(9),a)  
        return  a[len(a)//2 -self.primary: len(a)//2 -self.primary + self.num_classes]       #slicing the distribution based on primary

    



############## Assigning score to each row of data ##############################
    def ApplyScore(self):
        assert isinstance(self.series, pandas.core.series.Series),'InputError: Input is not pandas series'
        np_series_value = self.series.values
        np_series_score = np.array([self.finaldistribution[i] for i in np_series_value])
        return np_series_score

        

