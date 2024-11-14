import numpy as np
import pandas as pd

class ScoreDistribution:
    def __init__(self, series, primary, rate):
        """
        Initializes ScoreDistribution class with a pandas series, primary choice, and rating.
        Converts every value in the series to float and stores the unique sorted classes.

        Args:
            series (pandas.Series): Input data series
            primary (int): Index of the primary choice class
            rate (int): Rating value between 1 and 10

        Raises:
            AssertionError: If the input is not a pandas series or the rate is not between 1 and 10
        """
        assert isinstance(series, pd.core.series.Series), 'InputError: Input is not a pandas series'
        assert 1 <= rate <= 10, f'InputError: {rate} is not within the range of 1 to 10'

        self.series = series
        self.primary = primary
        self.rate = rate
        self.classes = (series.apply(float)).unique()
        self.classes.sort()
        self.num_classes = int(max(max(self.classes),self.primary)+1)
        self.final_distribution = self._get_final_rate_distribution()

    def _get_rating_distribution(self):
        """
        Calculates the rating distribution using the idea of a 10-point distribution
        around the primary choice, with a maximum limit of 2 for non-chosen classes.

        Args:
            rate (int): Rating value between 1 and 10

        Returns:
            numpy.ndarray: Rating distribution array
        """
        if self.rate == 10:
            return np.array([10])
        if self.rate == 1:
            return np.ones(10)

        distribution = [self.rate]
        flag = 1
        while sum(distribution) < 10:
            if flag == 1:
                if distribution[0] == 2 or (distribution[0] == self.rate):    ## dis[0]!=rate
                    distribution.insert(0, 1)
                elif distribution[0] == 1:
                    distribution[0] += 1
                elif distribution[0]==self.rate:
                    distribution.insert(1,1)
            elif flag == -1:
                if distribution[-1] == self.rate or (distribution[-1] >= 2):
                    distribution.append(1)
                elif distribution[-1] == 1:
                    distribution[-1] += 1
            flag *= -1

        return np.array(distribution)
    
    def _get_final_rate_distribution(self):
        """
        Calculates the final rate distribution by slicing the rating distribution
        based on the primary choice index.

        Returns:
            numpy.ndarray: Final rate distribution array
        """
        distribution = self._get_rating_distribution()
        final_distribution = np.append(np.zeros(self.num_classes), distribution)
        final_distribution = np.append(final_distribution, np.zeros(self.num_classes))
        return final_distribution[len(final_distribution) // 2 - self.primary: len(final_distribution) // 2 - self.primary + self.num_classes+1]
    
    def apply_score(self):
        """
        Applies the final rate distribution to the input data series.

        Returns:
            numpy.ndarray: Array of scores
        """
        assert isinstance(self.series, pd.core.series.Series), 'InputError: Input is not a pandas series'
        np_series_value = self.series.values
        np_series_score = np.array([self.final_distribution[int(i)] for i in np_series_value])
        #print(f'np_array of scores {np_series_score} for classes {self.classes}')
        return np_series_score