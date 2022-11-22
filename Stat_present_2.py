# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# import param_singan_multi as param
import nolds
from Stat_Calculator_1 import *
# from Gan_Simulator import extract_train_data, read_raw_data
import matplotlib.pyplot as plt


# Variance ratio
def _estimate(log_ret, k, const_arr):
    """
    输入一条对数收益率序列及滞后阶数，计算k阶方差比
    Reference: https://mingze-gao.com/measures/lomackinlay1988/
    :param log_ret: list or array. 对数收益率序列
    :param k: int. 滞后阶数
    :param const_arr: list or array. 计算方差比时使用到的一组常数
    :example
    log_ret = train_data
    k = 10
    const_arr = np.arange(k-1, 0, step=-1, dtype=np.int)
    estimate(log_ret, k, const_arr)
    """
    log_prices = np.cumsum(log_ret)
    T = len(log_ret)
    mu = np.mean(log_ret)
    sqr_demeaned_x = np.square(log_ret - mu)  # %定义一个后续反复用到的变量

    # % 1st order log return variance
    var_1 = np.sum(sqr_demeaned_x) / (T - 1)

    # % kth order log return variance
    rets_k = (log_prices - np.roll(log_prices, k))[k:]
    m = k * (T - k + 1) * (1 - k / T)
    var_k = 1/m * np.sum(np.square(rets_k - k * mu))

    vr = var_k / var_1  # %Variance ratio
    a_arr = np.square(const_arr * 2 / k)

    # % Calculating the variance ratio test statistic
    b_arr = np.empty(k-1, dtype=np.float64)
    for j in range(1, k):
        b_arr[j-1] = np.sum((sqr_demeaned_x * np.roll(sqr_demeaned_x, j))[j+1:])

    delta_arr = b_arr / np.square(np.sum(sqr_demeaned_x))
    assert len(delta_arr) == len(a_arr) == k - 1
    phi = np.sum(a_arr * delta_arr)
    vr_stat_heteroscedasticity = (vr - 1) / np.sqrt(phi)

    return vr_stat_heteroscedasticity


def estimate(data, k_list):
    """
    A fast estimation of Variance Ratio test statistics as in Lo and MacKinlay (1988)
    :param data: ndArray. 1000条生成序列或者1条真实序列，但是要是2维数组的形式传入
    :param k_list: list. 滞后阶数构成的list
    :return: tuple. 返回所有序列的平均各阶检验统计量和每条序列的各阶检验统计量
    :example:
    import pandas as pd
    _path_series = param.path_results + 'results_' + param.gan_type + '_' + param.path_suffix + '.xlsx'
    data = read_fake_series(_path_series, 'fake_data_' + param.gan_type)  # %读取生成序列：对数收益率序列
    k_list = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200]
    average_result, all_series_result = estimate(data, k_list)
    real_vr_stat = estimate(train_data.reshape(1, len(train_data)))[0]
    """
    all_series_result = np.zeros((data.shape[0], len(k_list)))
    for i in range(data.shape[0]):  
        single_series_result = [] 

        for k in k_list: 
            const_arr = np.arange(k - 1, 0, step=-1, dtype=np.int)
            single_series_result.append(_estimate(data[i, ], k, const_arr))
        all_series_result[i, ] = single_series_result

    average_result = np.mean(all_series_result, axis=0)
    return average_result, all_series_result


def plot_variance_ratio(all_series_result, real_vr_stat, k_list, folder):
    df = pd.DataFrame(all_series_result, columns=k_list)

    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Red')
    df.plot.box(title="Statistics of Variance Ratio Test", color=color, sym='r+')
    plt.grid(linestyle="--", alpha=0.3)
    plt.plot([None] + real_vr_stat.tolist(), linestyle='--')
    plt.savefig('%s/variance_ratio.png'%folder)


if __name__ == '__main__':
    np.random.seed(123)
    warnings.filterwarnings("ignore")
    # % param
    param_stat = Param_Stat()

    index_name = ["SP500_monthly(95)_new"]
    #index_name = ["000300.SH", "CBA00601.CS", "AU9999.SGE"]
    
    ls_gan_type = ['singan','dcgan','wgan']
    ls_gan_type = ['singan']

    for gan_type in ls_gan_type:
            
        # %% 遍历各资产
        for i_z, this_index in enumerate(index_name):
            # if i_z!=1:
            #     continue
            # 读取真实序列
            print('start reading excel')
            _path_series = '../Data/' + this_index + '.xlsx'
            ret_real = read_real_series(_path_series)
    
            train_data = ret_real
            # % 真实序列前六项指标
            res_real = calc_all_stats(train_data, param_stat)
            # % 真实序列的hurst指数
            real_hurst = nolds.hurst_rs(train_data)
            print('read_hurst=',real_hurst)
    
            k_list = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
        
            # % ------------ sinGAN生成序列的结果
            print("start reading excel")
            # ret_singan = read_fake_series('../Result/fake_data_%s.xlsx'%(gan_type),
            #                               'Sheet' + str(i_z + 1))[0:param_stat.num_choose, :]
            ret_singan = pd.read_excel('../Data/r_wgan_科创50_2000.xlsx').values
            ret_singan = ret_singan[:, 1:]
            print('start calculating %s series'%(gan_type))
            series = ret_singan
            res_singan, res_singan_mean = batch_calc_all_stats(series, param_stat)
    
            # % singan的方差比
            singan_average_result, singan_all_series_result = estimate(ret_singan, k_list)  # %生成序列的方差比率统计量结果
            real_vr_stat = estimate(train_data.reshape(1, len(train_data)), k_list)[0]  # %真实序列的方差比率统计量结果
            plot_variance_ratio(singan_all_series_result, real_vr_stat, k_list, '../Result/stat_%s_科创50_wgan_2000/'%(gan_type))
    
            # % singan的hurst指数
            singan_hurst = []
            for i in range(0, ret_singan.shape[0]):
                singan_hurst.append(nolds.hurst_rs(ret_singan[i]))
                print('finished [%d/%d]' % (i + 1, ret_singan.shape[0]))
    
            mean = np.mean(singan_hurst)  # %均值
            num = len(np.array(singan_hurst)[np.array(singan_hurst) >= 0.5]) / len(singan_hurst)  # %hurst>0.5(表现出长记忆性)的序列占比
            S_n = (1 / (len(singan_hurst) - 1)) * np.sum([(x - mean) ** 2 for x in singan_hurst])  # %计算无偏样本标准差
            U = np.sqrt(len(singan_hurst)) * (mean - 0.5) / S_n  # %均值检验统计量,大样本条件下即为正态均值检验
            print()
            print(gan_type,mean,num,S_n,U)
    
            # --------------- 以下代码用来输出各统计指标的数值
            # % 真实序列、singan的六项指标统计值
            # % 第六项指标单独算
            real_gain_loss = np.argmax(res_real.prob_gain) - np.argmax(res_real.prob_loss)
            singan_gain_loss = np.mean([np.argmax(x) for x in res_singan.prob_gain]) - np.mean([np.argmax(x) for x in res_singan.prob_loss])
    
            stat_real = [np.mean(res_real.acf[1:11]), res_real.alpha, res_real.beta, np.mean(res_real.lead_lag_corr[1:11]),
                         res_real.d_rho[1], real_gain_loss]
            stat_singan = [np.mean(res_singan_mean.acf[1:11]), res_singan_mean.alpha, res_singan_mean.beta,
                        np.mean(res_singan_mean.lead_lag_corr[1:11]),res_singan_mean.d_rho[1], singan_gain_loss]
            name_1 = ['自相关性', '厚尾分布', '波动率聚集', '杠杆效应', '粗细波动率相关', '盈亏不对称性']
            name_2 = ['前10阶自相关系数均值', '拟合幂律衰减系数alpha', '拟合幂律衰减系数beta',
                      '前10阶相关系数均值', '滞后正负1阶相关系数之差', '盈亏正负theta所需天数分布峰值之差']
    
            stat_df = pd.DataFrame({'评价指标': name_1, '统计量': name_2, '真实序列': stat_real,'%s生成序列'%(gan_type): stat_singan})
            stat_df.to_excel('../Result/stat_%s_科创50_wgan_2000/stat_value_%s.xlsx'%(gan_type,i_z),sheet_name=this_index)
    
            # % 真实序列、singan、dcgan和wgan的方差比率检验统计值
            vr_df = pd.DataFrame(np.array([real_vr_stat, singan_average_result]),
                                 index=['真实序列', gan_type], columns=k_list)
            vr_df.to_excel('../Result/stat_%s_科创50_wgan_2000/variance_ratio_%s.xlsx'%(gan_type,i_z),sheet_name=this_index)
    
            # % 真实序列、singan、dcgan和wgan的Hurst指数区间分布
            min_value = np.min(singan_hurst)
            max_value = np.max(singan_hurst)
            left_side = np.arange(np.floor(min_value * 10) / 10, np.ceil(max_value * 10) / 10, 0.03)
            right_side = left_side + 0.03
            hurst_df = pd.DataFrame({'min': left_side, 'max': right_side})
            hurst_df['interval'] = ['[' + str(x) + ',' + str(y) + ')' for (x, y) in zip(left_side, right_side)]
            hurst_df['%s_hurst'%(gan_type)] = [len([d for d in singan_hurst if (x <= d < y)]) for (x, y) in zip(left_side, right_side)]
    
            hurst_df.to_excel('../Result/stat_%s_科创50_wgan_2000/hurst_%s.xlsx'%(gan_type,i_z),sheet_name=this_index)
            print('read_hurst=', real_hurst)
