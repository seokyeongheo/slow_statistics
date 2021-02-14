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

    # z-test
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

    # simple t-test
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

    def confidence_interval_from_stats(self):
        ci_start = self.M - self.__calculate_t_portion__() * (self.std / np.sqrt(self.n))
        ci_end = self.M + self.__calculate_t_portion__() * (self.std / np.sqrt(self.n))
        print(f'[confidence interval] {ci_start} ~ {ci_end}')
        
    # independent-measure t-test
    def __calculate_t_statistic_ind__(self, M1, M2, SS1, SS2, n1, n2):
    
        pool_var = (SS1 + SS2) / (n1 - 1 + n2 - 1)
        std_error = np.sqrt((pool_var / n1) + (pool_var / n2))
        M_d = M1 - M2
        t_statistic = M_d / std_error

        return t_statistic

    def __calculate_t_portion_ind__(self, n1, n2):

        df = n1 + n2 - 2
        t_portion = round(stats.t.ppf(1 - self.alpha/self.tail_num, df=df), 3)

        return t_portion

    def ttest_ind_from_stats(self, M1, M2, SS1, SS2, n1, n2):

        t, cr = self.__calculate_t_statistic_ind__(M1, M2, SS1, SS2, n1, n2), self.__calculate_t_portion_ind__(n1, n2)

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

    def cohens_d_ind_from_stats(self, M1, M2, SS1, SS2, n1, n2):

        M_d = M1 - M2
        pool_var = (SS1 + SS2) / (n1 - 1 + n2 - 1)
        estimated_d = round(M_d / np.sqrt(pool_var), 3)

        return estimated_d

    def r_squared_ind_from_stat(self, M1, M2, SS1, SS2, n1, n2):

        t_statistic = self.__calculate_t_statistic_ind__(M1, M2, SS1, SS2, n1, n2)
        r_squared = round(t_statistic**2 / (t_statistic**2 + n1 + n2 - 2), 4)

        return r_squared
    
    def confidence_interval_ind_from_stats(self, M1, M2, SS1, SS2, n1, n2):
    
        M_d = M1 - M2
    
        pool_var = (SS1 + SS2) / (n1 - 1 + n2 - 1)
        std_error = np.sqrt((pool_var / n1) + (pool_var / n2))
        
        ci_start = round(M_d - self.__calculate_t_portion_ind__(n1, n2) * std_error, 4)
        ci_end = round(M_d + self.__calculate_t_portion_ind__(n1, n2) * std_error, 4)

        print(f'[confidence interval] {ci_start} ~ {ci_end}')
        
    # repeated-measure t-test
    def __calculate_t_statistic_rel__(self, M_d, SS, n):

        std_error = np.sqrt((SS / (n - 1)) / n)
        t_statistic = (M_d - 0) / std_error

        return t_statistic

    def __calculate_t_portion_rel__(self, n):

        df = n - 1
        t_portion = round(stats.t.ppf(1 - self.alpha/self.tail_num, df=df), 3)

        return t_portion

    def ttest_rel_from_stats(self, M_d, SS, n):

        t, cr = self.__calculate_t_statistic_rel__(M_d, SS, n), self.__calculate_t_portion_rel__(n)

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

    def cohens_d_rel_from_stats(self, M_d, SS, n):

        s = np.sqrt(SS / (n - 1))
        estimated_d = round(M_d / s, 3)

        return estimated_d

    def r_squared_rel_from_stats(self, M_d, SS, n):

        t_statistic = self.__calculate_t_statistic_rel__(M_d, SS, n)
        r_squared = round(t_statistic**2 / (t_statistic**2 + n - 1), 4)

        return r_squared

    def confidence_interval_rel_from_stats(self, M_d, SS, n):

        std_error = np.sqrt((SS / (n - 1)) / n)
        ci_start = round(M_d - self.__calculate_t_portion_rel__(n) * std_error, 4)
        ci_end = round(M_d + self.__calculate_t_portion_rel__(n) * std_error, 4)

        print(f'[confidence interval] {ci_start} ~ {ci_end}')