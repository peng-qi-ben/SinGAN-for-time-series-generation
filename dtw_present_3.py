# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# import param_wdcgan_multi as param
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
#from Stat_Calculator import *


# %% 读取虚假价格时间序列，转换为对数收益率
def read_fake_series(path_series, sheet_name, fill_start=True):
    df_fake = pd.read_excel(path_series, sheet_name=sheet_name, index_col=0).values
    ret_fake = np.array(df_fake)
    # 第1列以1补齐
    if fill_start:
        df_fake = np.hstack([np.ones([df_fake.shape[0], 1]), df_fake])
    #ret_fake = np.log(df_fake[:, 1:] / df_fake[:, :-1])
    print('finish reading %s' % sheet_name)
    return ret_fake

# if __name__=='__main__':

#     index_name = ["科创50"]
#     # index_name = ["000300.SH", "CBA00601.CS", "AU9999.SGE"]
    
#     ls_gan_type = ['wdcgan','dcgan','wgan']
#     ls_gan_type = ['singan']
    
#     for gan_type in ls_gan_type:

#         # %% 遍历各资产
#         for i_z, this_index in enumerate(index_name):
#             print('start reading excel')
#             _path_series = '../Data/' + this_index + '.xlsx'
#             # ret_singan = read_fake_series('../Result/fake_data_%s.xlsx'%(gan_type),
#             #                             'Sheet' + str(i_z + 1))[0:1000, :]
#             ret_singan = pd.read_excel('../Data/r_singan_科创50_20_400_2000_0.8_50.xlsx').values

#             # % 随机抽取1000组序列对
#             series_index_1 = np.random.randint(0, ret_singan.shape[0], 1000)
#             series_index_2 = np.random.randint(0, ret_singan.shape[0], 1000)
    
#             # % wdcgan_new生成序列和wdcgan生成序列各自1000条序列对的DTW值。下面这段代码运行时间较久
#             distance_singan = []
#             for i in range(len(series_index_1)):
#                 distance_singan.append(fastdtw(ret_singan[series_index_1[i], :], ret_singan[series_index_2[i], :], dist=euclidean)[0])
#                 print('DTW {}/{} finished!'.format(i+1, len(series_index_1)))
    
#             # % 展示DTW均值
#             print(gan_type, np.mean(distance_singan))
    
            # # % 输出到excel，为绘图方便，这里对各自的DTW进行区间统计
            # n = 2
            # min_value = np.min(distance_singan)
            # max_value = np.max(distance_singan)
            # if int(min_value)%2 == 1:
            #     min_value = min_value - 1
            # left_side = np.arange(int(min_value) - n, int(max_value) + n, n)
            # right_side = left_side + n
            # df = pd.DataFrame({'min': left_side, 'max': right_side})
            # df['interval'] = ['[' + str(x) + ',' + str(y) + ')' for (x, y) in zip(left_side, right_side)]
            # df['distance_%s'%(gan_type)] = [len([d for d in distance_singan if (x <= d < y)]) for (x, y) in zip(left_side, right_side)]
            #
            # # df.to_excel('../Result/stat_%s/DTW_'%(gan_type) + str(i_z) + '.xlsx')
            # df.to_excel('../Result/stat_singan_科创50_wgan_2000/DTW_0.xlsx')
    
