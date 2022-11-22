"""
SinGAN：Graphing and data processing

Draw the price graph of the sequence generated by SinGAN
and prepare the data for the test

The results of daily frequency data are in the folder of "daily_6+2_test".
The results of monthly frequency data are in the folder "monthly_fourier"

Where the monthly frequency data output price series
Daily frequency data output price series and yield series, where the yield series is used for the next 6+2 test (calculated with 6+2test)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from Stat_Calculator_1 import *


plt.rcParams['axes.unicode_minus'] = False 


def get_price(r, price_real):
    price = (np.exp(r.cumsum(axis=1)) * float(price_real.iloc[0])).T
    price.loc[-1] = float(price_real.iloc[0])
    price = price.sort_index()
    return price


# Draw the original price chart
def plot_price_real(price_real, type):
    # plt.figure()
    # plt.plot(price_real.index.astype(str), price_real.values)
    price_real.plot(label='real')
    plt.title('%s real' % (type))
    plt.show()


# Draw the generated sequence diagram
def plot_price_fake(price_fake, type, index_show):
    # plt.figure()
    # plt.plot(price_real.index.astype(str), price_real.values, label='real')
    price_real.plot(label='real')
    for i in index_show:
        # plt.plot(price_fake.index, price_fake.iloc[:, i].values, label='SinGAN_fake')
        price_fake.iloc[:, i].plot(label='fake')
    plt.title('%s SinGAN' % (type))
    plt.legend()
    plt.show()


# ignore warning
warnings.filterwarnings("ignore")

'''
parameters
'''
use_ret = 1  # Yield = 1; Price series = 0
is_day = 0  # Daily frequency=1; Monthly frequency=0
type = 'crb'  
param_singan = 'min=15,max=320,epoch=1000,factor=0.700000,scale=20'  # SinGAN parameters can be pasted directly from the output folder by finding the corresponding folder name directly.

index_show = [1,2,3]  # Which price series are displayed

if __name__ == '__main__':
    '''
    Draw price sequence diagram
    '''
    # Read the real sequence
    price_real = pd.read_excel('Input/Excel/%s.xlsx' % type,
                               index_col=0,
                               )
    print(price_real.dtypes)
    # Draw the diagram of the real sequence
    plot_price_real(price_real, type)
    # If using the yield series
    if use_ret == 1:
        # Read the generated sequence
        ret_fake = pd.read_excel(
            'Output/RandomSamples/%s/gen_start_scale=0/%s/result_r.xlsx' % (type, param_singan),
            index_col=0)
        # Get Price Sequence
        price_fake = get_price(ret_fake, price_real)
        price_fake.index = price_real.index
        # Draw to generate a sequence price chart
        plot_price_fake(price_fake, type, index_show)

    # Price data is read directly from the normalized price
    else:
        # If using price series
        # Convert to a drawing-friendly format
        price_fake = pd.read_excel(
            'Output/RandomSamples/%s/gen_start_scale=0/%s/result_pirce_z.xlsx' % (type, param_singan),
            index_col=0)
        price_fake = price_fake.T
        price_fake.index = price_real.index

        price_real = (price_real - np.mean(price_real)) / (np.std(price_real))
        # Draw to generate a sequence price chart
        plot_price_fake(price_fake, type, index_show)

    '''
    Preparation of test data
    '''
    # For the monthly frequency data, the prices are exported directly to excel in order to see the results of the Fourier Tranformation
    if is_day == 0:
        try:
            os.makedirs('monthly_fourier/%s/%s' % (type, param_singan))
        except:
            pass
        price_fake.to_excel('monthly_fourier/%s/%s/singan_price_fake.xlsx' % (type, param_singan))
        print('Output complete')

    # For daily frequency data, ret series are also output for testing
    else:
        print('Daily frequency data output price and yield series, where the 6+2 indicator test is performed with the yield series')
        try:
            os.makedirs('daily_6+2_test/%s/%s' % (type, param_singan))
        except:
            pass
        print('Output yield series')
        ret_fake.to_excel('daily_6+2_test/%s/%s/singan_ret_fake.xlsx' % (type, param_singan))
        print('Output price series')
        price_fake.to_excel('daily_6+2_test/%s/%s/singan_price_fake.xlsx' % (type, param_singan))



        folder = 'daily_6+2_test/%s/%s' % (type,param_singan)
        # Obtain the real yield series
        param_stat = Param_Stat()

        # Get the real sequence
        price_real_ = price_real.values
        ret_real = np.log(price_real_[1:] / price_real_[:-1])
        # plot
        # Real Sequence
        print('start calculating real series')
        series = ret_real
        res_real = calc_all_stats(series, param_stat)
        plot_stat_properties(series, param_stat, res_real, folder , suffix = 'real')

        # Fake Sequence
        print('start calculating fake series')