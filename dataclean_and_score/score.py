import numpy as np
from utils.dataclean_jagath import DataClean

instance  = DataClean(path)
clean_df = instance.clean_df



###### Now lets write distribution functions #######3
class score():
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
                    ### insert amd give 1
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
    
    def FinalRateDistribution(self,primary,num_classes,rate):
        """
        Still need to discuss,
        if law of rating 10 is required, and if does, where to distribute
        edge score to -- should it be reversed or added to the tails

        Args:
            primary (int): Must be within num_classes's index
            num_classes (int): must be greater than 1
            rate (int): Must be within 1-10
        """
        z = np.zeros(num_classes)
        z(primary) = 1
        distribution = self.__RatingDistribution__(rate)
        d = distribution
        ### padded_d = np.zeros(num_classes)+ d+ np.zeros(num_classes)
        ### 



