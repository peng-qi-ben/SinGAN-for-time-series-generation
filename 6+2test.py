"""
test the model using 8 indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from Stat_Calculator_1 import *
from stat_present_2 import *
from dtw_present_3 import *

import warnings

warnings.filterwarnings("ignore")

'''
Parameters
'''
type = 'HS300_daily_new'  # 
param_singan = 'wgan,epoch=1500'
DTW_run = 1  # if calculating DTW


if __name__ == '__main__':
    # Obtain the real yield series
    param_stat = Param_Stat()
    # Folder
    folder = 'daily_6+2_test/%s/%s' % (type, param_singan)

    # Get the real sequence
    price_real = pd.read_excel('Input/Excel/%s.xlsx' % type,
                               index_col=0)
    price_real_ = price_real.values
    ret_real = np.log(price_real_[1:] / price_real_[:-1])

    '''
    The first 6 indicators are drawn
    '''
    # True series (first six indicators)
    print('start calculating real series')
    res_real = calc_all_stats(ret_real, param_stat)
    plot_stat_properties(ret_real, param_stat, res_real, folder, suffix='real(6)')

    # Fake series (first six indicators)
    print('start calculating singan series')
    ret_singan = pd.read_excel('%s/singan_ret_fake.xlsx' % folder, index_col=0).values

    res_singan, res_singan_mean = batch_calc_all_stats(ret_singan, param_stat)
    plot_stat_properties(ret_singan, param_stat, res_singan_mean, folder, suffix='singan(6)')
    print('start calculating fake series')

    '''
    The first 6 indicators output to excel + Hurst + Variance ratio
    '''
    np.random.seed(123)
    train_data = ret_real
    # True hurst value
    real_hurst = nolds.hurst_rs(train_data)
    print()
    print('real_hurst=', real_hurst)

    # Results of SinGAN
    k_list = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    print('start calculating singan series')

    # The variance ratio of SinGAN
    singan_average_result, singan_all_series_result = estimate(ret_singan, k_list)  # %生成序列的方差比率统计量结果
    real_vr_stat = estimate(train_data.reshape(1, len(train_data)), k_list)[0]  # %真实序列的方差比率统计量结果
    plot_variance_ratio(singan_all_series_result, real_vr_stat, k_list, folder)

    # % SinGAN's hurst index
    singan_hurst = []
    for i in range(0, ret_singan.shape[0]):
        singan_hurst.append(nolds.hurst_rs(ret_singan[i]))
        print('finished [%d/%d]' % (i + 1, ret_singan.shape[0]))

    mean = np.mean(singan_hurst)  # 
    num = len(np.array(singan_hurst)[np.array(singan_hurst) >= 0.5]) / len(singan_hurst)  # %The percentage of sequences with hurst > 0.5 (exhibiting long memory)
    S_n = (1 / (len(singan_hurst) - 1)) * np.sum([(x - mean) ** 2 for x in singan_hurst])  # %Calculate the unbiased sample standard deviation
    U = np.sqrt(len(singan_hurst)) * (mean - 0.5) / S_n  # %The mean test statistic, which is a normal mean test under large sample conditions
    print()
    print('real_Hurst=', real_hurst)
    print('singan', mean, num, S_n, U)

    # --------------- The following code is used to output the value of each statistical indicator
    # % Statistical values of the six indicators of the true series and SinGAN
    # % The sixth indicator is counted separately
    real_gain_loss = np.argmax(res_real.prob_gain) - np.argmax(res_real.prob_loss)
    singan_gain_loss = np.mean([np.argmax(x) for x in res_singan.prob_gain]) - np.mean(
        [np.argmax(x) for x in res_singan.prob_loss])

    stat_real = [np.mean(res_real.acf[1:11]), res_real.alpha, res_real.beta, np.mean(res_real.lead_lag_corr[1:11]),
                 res_real.d_rho[1], real_gain_loss]
    stat_singan = [np.mean(res_singan_mean.acf[1:11]), res_singan_mean.alpha, res_singan_mean.beta,
                   np.mean(res_singan_mean.lead_lag_corr[1:11]), res_singan_mean.d_rho[1], singan_gain_loss]
    name_1 = ['Autocorrelation', 'Fat-tailed', 'Volatility clustering', 'Leverage effect', 'Coarse-fine volatility', 'Gain/loss asymmetry']
    name_2 = ['Mean value of the first 10 orders of autocorrelation coefficient', 'Fitting the power-law decay coefficient alpha', 'the power-law decay coefficient beta',
              'Mean value of the first 10 orders of correlation coefficient', 'Difference of lagged positive and negative 1st order correlation coefficients', 'The difference between the peak of the distribution of the number of days required for positive and negative theta for profit and loss']

    stat_df = pd.DataFrame({'Indicators': name_1, 'Statistical quantities': name_2, 'Real_Sequence': stat_real, '%sgenerated sequence' % ('singan'): stat_singan})
    stat_df.to_excel('%s/stat_value.xlsx' % (folder))

    # % Variance ratio test statistic values for true series, singan
    vr_df = pd.DataFrame(np.array([real_vr_stat, singan_average_result]),
                         index=['Real_Sequence', 'singan'], columns=k_list)
    vr_df.to_excel('%s/variance_ratio.xlsx' % (folder))
    # % Hurst index interval distribution for true series, singan
    min_value = np.min(singan_hurst)
    max_value = np.max(singan_hurst)
    left_side = np.arange(np.floor(min_value * 10) / 10, np.ceil(max_value * 10) / 10, 0.03)
    right_side = left_side + 0.03
    hurst_df = pd.DataFrame({'min': left_side, 'max': right_side})
    hurst_df['interval'] = ['[' + str(x) + ',' + str(y) + ')' for (x, y) in zip(left_side, right_side)]
    hurst_df['%s_hurst' % ('singan')] = [len([d for d in singan_hurst if (x <= d < y)]) for (x, y) in
                                         zip(left_side, right_side)]
    hurst_df.to_excel('%s/hurst.xlsx' % (folder))

    '''
    DTW
    '''
    if DTW_run == 1:
        series_index_1 = np.random.randint(0, ret_singan.shape[0], 1000)
        series_index_2 = np.random.randint(0, ret_singan.shape[0], 1000)

        distance_singan = []
        for i in range(len(series_index_1)):
            distance_singan.append(
                fastdtw(ret_singan[series_index_1[i], :], ret_singan[series_index_2[i], :], dist=euclidean)[0])
            print('DTW {}/{} finished!'.format(i + 1, len(series_index_1)))

        print('singan DTW mean=', np.mean(distance_singan))

        # Output to excel, for the convenience of plotting, here are the interval statistics for the respective DTW
        n = 2
        min_value = np.min(distance_singan)
        max_value = np.max(distance_singan)
        if int(min_value) % 2 == 1:
            min_value = min_value - 1
        left_side = np.arange(int(min_value) - n, int(max_value) + n, n)
        right_side = left_side + n
        df = pd.DataFrame({'min': left_side, 'max': right_side})
        df['interval'] = ['[' + str(x) + ',' + str(y) + ')' for (x, y) in zip(left_side, right_side)]
        df['distance_%s' % ('singan')] = [len([d for d in distance_singan if (x <= d < y)]) for (x, y) in
                                          zip(left_side, right_side)]

        df.to_excel('%s/DTW.xlsx' % folder)

        print('-----------------------------------------------------------------------------')
        print('real_Hurst=', real_hurst)
        print('singan', mean, num, S_n, U)
        print('-----------------------------------------------------------------------------')
        print('singan DTW mean=', np.mean(distance_singan))
