import numpy as np
import pandas as pd

from scipy import stats

class Stats():

    def __init__(self):
        self.n = 0
        self.mu = 0
        self.std = 0
        self.M = 0
        self.alpha = 0.05
        self.tail_num = 2        

    def __calculate_zscore__(self):
        se = self.std / np.sqrt(self.n)
        return round((self.M - self.mu) / se, 2)

    def __calculate_norm_portion__(self):
        return round(stats.norm.ppf(1 - self.alpha/self.tail_num), 3)

    def ztest_1samp_from_stats(self):

        z, cr = self.__calculate_zscore__(), self.__calculate_norm_portion__()

        if self.tail_num == 2:

            rejection_decision = (z > cr) | (z < -1 * cr)
            region = f'z > {cr} or z < -{cr}'
            criteria = f'two tail, alpha {self.alpha}'

        elif self.tail_num == 1:

            if z > 0:

                rejection_decision = (z > cr)
                region = f'z > {cr}'

            else:

                rejection_decision = (z < -1 * cr)
                region = f'z < -{cr}'

            criteria = f'one tail, alpha {self.alpha}'

        else:
            print('Should use tail_num 1 or 2.')
            return None

        print(f'[{criteria}] z_statistic:{z}, critical_region:{region}\n=> null hypothesis rejection [{rejection_decision}]')

        
    def __calculate_t_statistic__(self):
        se = self.std / np.sqrt(self.n)
        return round((self.M - self.mu) / se, 2)

    def __calculate_t_portion__(self):
        df = self.n - 1
        return round(stats.t.ppf(1 - self.alpha/self.tail_num, df=df), 3)

    def r_squared_from_stats(self):
        t = self.__calculate_t_statistic__()
        return t ** 2 / (t ** 2 + self.n - 1)

    def ttest_1samp_from_stats(self):

        t, cr = self.__calculate_t_statistic__(), self.__calculate_t_portion__()

        if self.tail_num == 2:

            rejection_decision = (t > cr) | (t < -1 * cr)
            region = f't > {cr} or t < -{cr}'
            criteria = f'two tail, alpha {self.alpha}'

        elif self.tail_num == 1:

            if t > 0:

                rejection_decision = (t > cr)
                region = f't > {cr}'

            else:

                rejection_decision = (t < -1 * cr)
                region = f't < -{cr}'

            criteria = f'one tail, alpha {self.alpha}'

        else:
            print('Should use tail_num 1 or 2.')
            return None

        print(f'[{criteria}] t_statistic:{t}, critical_region:{region}\n=> null hypothesis rejection [{rejection_decision}]')
        
        
    def cohens_d_from_stats(self):
        return round(abs((self.M - self.mu) / self.std), 2)

#     def statistical_power_from_stats():
#         se = self.std / np.sqrt(self.n)
#         z = ((self.mu + 1.96 * se) - self.M) / se
#         return round(1 - stats.norm.cdf(z), 4)