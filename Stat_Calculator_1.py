# -*- coding: utf-8 -*-
"""
Calculate statistical indicator functions

refer to Takahashi(2019)：
    Linear unpredictbility:   Corr(r(t),r(t+k)) ≈ 0,for k>=1
    Fat-tailed distribution:  pdf of the price return P(r) ∝ r^(-α)
    Volatility clustering:    Corr(|r(t)|,|r(t+k)|) ∝ k^(-β)
    Leverage effect:          L(k) = (E[r(t)|r(t+k)|^2] - E[r(t)]*E[|r(t)|^2]) / E[|r(t)|^2]^2
    Coarse-fine volatility:   vc(t)=|Σ r(t-i)|,  vf=Σ|r(t-i)|,
                              ρ(k)=Corr(vc(t+k),vf(t)),  Δρ(k)=ρ(k)-ρ(-k)
    Gain/loss asymmetry:      T_wait(Θ) = inf{t'|log(p(t+t'))-log(p(t))>=Θ}, Θ>0
                              T_wait(Θ) = inf{t'|log(p(t+t'))-log(p(t))<=Θ}, Θ<0  

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from sklearn import preprocessing
import powerlaw
from scipy.optimize import curve_fit


#%% Parameters
class Param_Stat():
    # 1 Autocorrelation coefficient 3 Volatility clustering 4 Leverage effect
    max_lag = 200
    # 5 Coarse-fine volatility
    tau = 5 # monthly 3 daily 5
    max_lag_cf = 20
    # 6 Gain/loss asymmetry
    theta = 0.1 # monthly 0.25 daily 0.1
    max_ts = 1000
    # debug
    num_choose = 1000 # sample size
    

#%% result
class Res_Stat():
    def __init__(self):
        # 1 Autocorrelation coefficient
        self.acf = []
        # 2 Fat-tailed distribution
        self.pos_tail, self.alpha = [], []
        # 3 Volatility clustering
        self.acf_abs, self.beta = [], []
        # 4 Leverage effect
        self.lead_lag_corr = []
        # 5 Coarse-fine volatility
        self.rho, self.d_rho = [], []
        # 6 Gain/loss asymmetry
        self.prob_gain, self.prob_loss = [], []
    
    
#%% Read real price time series and convert to price
def read_real_series(path_series):
    df_real = pd.read_excel(path_series,index_col=0).values
    #df_real = pd.read_excel(path_series, index_col=0).values
    temp_data = df_real[1:,0] / df_real[:-1,0]
    ret_real = np.log(temp_data.astype(float))
    print('finish reading real_data')
    return ret_real


#%% Read fake price time series and convert to price
def read_fake_series(path_series,sheet_name,fill_start=True):
    df_fake = pd.read_excel(path_series,sheet_name=sheet_name,index_col=0)

    if fill_start:
        df_fake = np.hstack([np.ones([df_fake.shape[0],1]),df_fake])

    ret_fake = np.log(df_fake[:,1:] / df_fake[:,:-1])
    print('finish reading %s'%(sheet_name))
    return ret_fake


#%% 1. Linear-unpredictability
def calc_ar(series,max_lag=120):
    """
    Linear-unpredictability
    计算k阶自相关系数
    Corr(r(t),r(t+k)) ≈ 0,for k>=1
    """
    series = series.reshape((-1))
    acf = smt.stattools.acf(series,nlags=max_lag,fft=False)
    return acf


#%% 2. Fat-tailed distribution
# 标准化收益率幂律衰减系数
def calc_fat_tail(series):
    """
    Fat-tailed distribution
    pdf of the price return P(r) ∝ r^(-α)
    The exponent typically ranges 3 ≤ α ≤ 5
    """
    series = series.reshape((-1))
    # 将对数收益率标准化
    norm_series = preprocessing.scale(series)
    # 取正项尾部数据
    pos_tail = norm_series[norm_series>0]
    # 用指数幂拟合正项尾部数据，P(r) ∝ r^(-α),计算α
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(pos_tail,verbose=False)
    alpha = fit.power_law.alpha
    #if alpha>=3 and alpha<=10:
    # 返回正收益率和alpha
    return pos_tail, alpha
 

#%% func_power
def func_power(x, a, b):
    y = a / np.power(x,b)
    return y
    
def func_log_power(lnx,lna,b):
    lny = lna - b * lnx
    return lny
    
    
#%% 3. Volatility clustering
def calc_vol_clust(series,max_lag=120):
    """
    Volatility clustering
    计算绝对值对数收益率的自相关函数
    Corr(|r(t)|,|r(t+k)|) ∝ k^(-β)
    The exponent β ranges between 0.1 and 0.5
    """
    series = series.reshape((-1))
    abs_series = np.abs(series)
    # (|r(t)|,|r(t+k)|)
    acf_abs = smt.stattools.acf(abs_series,nlags=max_lag,fft=False)
    # Corr(|r(t)|,|r(t+k)|) ∝ k^(-β),拟合β的值
    xdata = np.arange(1,max_lag+1)
    ydata = acf_abs[1:]
    try:
        popt, ppp = curve_fit(func_power, xdata, ydata,
                              p0=[ydata[0],0.1])
        # popt, ppp = curve_fit(func_log_power, np.log(xdata), np.log(ydata),
        #                       p0=[np.log(ydata[0]),0.1])
        beta = popt[1]
    except RuntimeError:
        beta = np.nan
    # 返回收益率绝对值的自相关系数和beta值
    return acf_abs, beta
 
   
#%% 4. Leverage effect
# 当前收益series和未来波动率series**2的滞后相关
def calc_lever_effect(series,max_lag=120):
    """
    Leverage effect
    lead–lag correlation function：
    L(k) = (E[r(t)|r(t+k)|^2] - E[r(t)]*E[|r(t)|^2]) / E[|r(t)|^2]^2
    """
    series = series.reshape((-1))
    # 计算L(k)
    ret_vol_corr = np.zeros(max_lag)
    # 原文公式有误，参考Qiu(2006)文献公式
    denom = (series**2).mean()**2
    for k in range(1,max_lag+1):
        series_a = series[:-k]
        series_b = series[k:]
        nom_a = (series_a * (series_b**2)).mean()
        nom_b = series_a.mean() * (series_a**2).mean()  
        ret_vol_corr[k-1] = (nom_a - nom_b) / denom
    # 返回收益率和波动率滞后相关
    return ret_vol_corr
 

#%% 两条时间序列领先/滞后相关
def lead_lag_corr(series_a,series_b,k):
    '''
    corr(a+k,b)
    '''
    if k>0:
        r = np.corrcoef(series_a[k:],series_b[:-k])[0,1]
    elif k==0:
        r = np.corrcoef(series_a,series_b)[0,1]
    else:
        abs_k = -k
        r = np.corrcoef(series_a[:-abs_k],series_b[abs_k:])[0,1]
    return r


#%% 5. Coarse-fine volatility
# 细波动率和滞后粗波动率的相关
# 若k期和-k期相关系数不对称，即差值为负数，表明细波动率可以预测未来粗波动率
def calc_coarse_fine_vol(series,tau=5,max_lag=20):
    """
    Coarse-fine volatility correlation
    coarse volatility: vc(t)=|Σ r(t-i)| 
    fine volatility:   vf=Σ|r(t-i)|
    ρ(k)=Corr(vc(t+k),vf(t)),  Δρ(k)=ρ(k)-ρ(-k)
    τ = 5 is chosen as it stands for day-week time-scales
    """
    series = series.reshape((-1))
    # vc是过去tau日收益率之和的绝对值（粗）
    vc = np.array([np.abs(series[i-tau:i].sum()) for i in range(tau,len(series))]) # 长度为l-τ
    # vf是过去tau日收益率的绝对值之和（细）
    vf = np.array([np.abs(series[i-tau:i]).sum() for i in range(tau,len(series))]) # 长度为l-τ
    # ρ(k)计算-20~20阶滞后
    lag_rho = np.arange(-max_lag,max_lag+1)
    rho = np.zeros((len(lag_rho)))
    for i, k in enumerate(lag_rho):
        rho[i] = lead_lag_corr(vc,vf,k)
    # Δρ(k)计算0~20阶滞后
    lag_d_rho = np.arange(0,max_lag+1)
    d_rho = np.zeros((len(lag_d_rho)))
    for i, k in enumerate(lag_d_rho):
        d_rho[i] = rho[max_lag+i] - rho[max_lag-i]
    # 返回相关系数
    return rho, d_rho


#%% 6. Gain/loss asymmetry
# 实现涨跌幅theta所需天数的分布，涨得慢，跌得快
def calc_gain_loss_asym(series,theta=0.1,max_ts=120):
    """
    Gain/loss asymmetry
    计算Gain/loss asymmetry
    T_wait(Θ) = inf{t'|log(p(t+t'))-log(p(t))>=Θ}, Θ>0
    T_wait(Θ) = inf{t'|log(p(t+t'))-log(p(t))<=Θ}, Θ<0
    参数Θ的值设为theta
    最大时间间隔t'不超过max_ts
    """
    series = series.reshape((-1))
    # 遍历每个点，找到每个点的最小时间间隔
    T_wait_gain = []
    T_wait_loss = []
    for i in range(len(series)):
        cum_ret = series[i:].cumsum()
        # Θ>0时，T_wait(Θ) = inf{t'|log(p(t+t'))-log(p(t))>=Θ}
        temp_idx = np.where(cum_ret>theta)[0]
        if len(temp_idx) > 0 and temp_idx[0]<=max_ts-1:
            T_wait_gain.append(temp_idx[0]+1)
        # Θ<0时，T_wait(Θ) = inf{t'|log(p(t+t'))-log(p(t))<=Θ}     
        temp_idx = np.where(cum_ret<-theta)[0]
        if len(temp_idx) > 0 and temp_idx[0]<=max_ts-1:
            T_wait_loss.append(temp_idx[0]+1)    


    # prob_gain表示各个取值的概率
    prob_gain = np.bincount(T_wait_gain,minlength=max_ts+1) / len(T_wait_gain)
    # prob_loss表示各个取值的概率
    prob_loss = np.bincount(T_wait_loss,minlength=max_ts+1) / len(T_wait_loss)
    # 返回实现涨跌幅theta所需天数的分布
    return prob_gain[1:], prob_loss[1:]


#%% 对单个序列计算全部统计指标
def calc_all_stats(series,param_stat):
    res = Res_Stat()
    res.acf = calc_ar(series,param_stat.max_lag)
    res.pos_tail, res.alpha = calc_fat_tail(series)
    res.acf_abs, res.beta = calc_vol_clust(series,param_stat.max_lag)
    res.lead_lag_corr = calc_lever_effect(series,param_stat.max_lag)
    res.rho, res.d_rho = calc_coarse_fine_vol(series,param_stat.tau,param_stat.max_lag_cf)
    res.prob_gain, res.prob_loss = calc_gain_loss_asym(series,param_stat.theta,param_stat.max_ts)
    return res


def batch_calc_all_stats(series,param_stat):
    res = Res_Stat()
    for i in range(series.shape[0]):
        acf = calc_ar(series[i,:],param_stat.max_lag)
        pos_tail, alpha = calc_fat_tail(series[i,:])
        acf_abs, beta = calc_vol_clust(series[i,:],param_stat.max_lag)
        lead_lag_corr = calc_lever_effect(series[i,:],param_stat.max_lag)
        rho, d_rho = calc_coarse_fine_vol(series[i,:],param_stat.tau,param_stat.max_lag_cf)
        prob_gain, prob_loss = calc_gain_loss_asym(series[i,:],param_stat.theta,param_stat.max_ts)
        # append to res
        res.acf.append(acf)
        res.pos_tail.extend(pos_tail)
        res.alpha.append(alpha)
        res.acf_abs.append(acf_abs)
        res.beta.append(beta)
        res.lead_lag_corr.append(lead_lag_corr)
        res.rho.append(rho)
        res.d_rho.append(d_rho)
        res.prob_gain.append(prob_gain)
        res.prob_loss.append(prob_loss)
        # print
        if (i+1)%100 == 0:
            print('Finish series [%d/%d]'%(i+1,series.shape[0]))

    res_mean = Res_Stat()
    res_mean.acf = np.nanmean(np.array(res.acf),axis=0)
    res_mean.pos_tail = np.array(res.pos_tail)
    res_mean.alpha = np.nanmean(res.alpha)
    print(np.nanmean(res.alpha))
    res_mean.acf_abs = np.nanmean(np.array(res.acf_abs),axis=0)
    res_mean.lead_lag_corr = np.nanmean(np.array(res.lead_lag_corr),axis=0)
    res_mean.rho = np.nanmean(np.array(res.rho),axis=0)
    res_mean.d_rho = np.nanmean(np.array(res.d_rho),axis=0)
    res_mean.prob_gain = np.nanmean(np.array(res.prob_gain),axis=0)
    res_mean.prob_loss = np.nanmean(np.array(res.prob_loss),axis=0)
    res.beta = np.array(res.beta)
    res.beta[res.beta>10] = np.nan
    res_mean.beta = np.nanmean(res.beta)
    return res, res_mean


#%% plot
def plot_stat_properties(series,param_stat,res,folder,suffix):
    plt.figure(figsize=(16,9))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    # 1. 
    plt.subplot(231)
    ax = plt.gca()
    ax.set_ylim([-1,1])
    ax.set_xscale('log')
    plt.plot(res.acf,'.',markersize=2)
    plt.xlabel('lag k')    
    plt.ylabel('Auto-correlation')
    plt.title('Linear unpredictability')
    
    # 2. 
    plt.subplot(232)
    powerlaw.plot_ccdf(res.pos_tail,color='b',marker='.',markersize=2, linewidth=0.01)
    plt.xlabel('normalized log-return')
    plt.ylabel('p(r>x)')
    plt.title('Fat-tailed distribution')
    # plt.show()


    # 3. 
    plt.subplot(233)    
    ax = plt.gca()
    ax.set_ylim([-0.5,0.5]) # ax.set_ylim([10**(-5),1])
    ax.set_xscale('log')
    #ax.set_yscale('log')  
    plt.plot(np.arange(param_stat.max_lag+1),res.acf_abs,'.',markersize=3)
    plt.xlabel('lag k')
    plt.ylabel('Auto-correlation')
    plt.title('Volatility clustering')
    
    # 4. 
    plt.subplot(234)
    plt.plot(np.arange(len(res.lead_lag_corr)),res.lead_lag_corr,linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--',linewidth=0.7,alpha=0.7)
    plt.xlabel('lag k')
    plt.ylabel('L(k)')
    plt.title('Leverage effect')
    
    # 5. 
    lag_rho = np.arange(-param_stat.max_lag_cf, param_stat.max_lag_cf+1)     
    lag_d_rho = np.arange(0, param_stat.max_lag_cf+1)
    plt.subplot(235)
    ax = plt.gca()
    ax.set_ylim([-0.1,1])
    
    plt.plot(lag_rho,res.rho,'.',markersize=2,color='b',label='ρ(k)') 
    # 黄色点表示Δρ(k)
    plt.plot(lag_d_rho,res.d_rho,'.',markersize=2,color='orange',label='Δρ(k)')
    plt.axhline(y=0, color='k', linestyle='--',linewidth=0.7,alpha=0.7)
    plt.xlabel('lag k')
    plt.ylabel('ρ(k)')
    plt.title('Coarse-fine volatility correlation')
    plt.legend()

    # 6. 
    plt.subplot(236)
    ax = plt.gca()
    ax.set_xlim([1,param_stat.max_ts])
    ax.set_xscale('log')
    # 红点表示T_wait(+Θ)
    plt.plot(np.arange(1,param_stat.max_ts+1),res.prob_gain,'.',markersize=2,color='r',label='T_wait(+Θ)')
    # 蓝点表示T_wait(-Θ)
    plt.plot(np.arange(1,param_stat.max_ts+1),res.prob_loss,'.',markersize=2,color='b',label='T_wait(-Θ)')
    plt.xlabel("time ticks t'")
    plt.ylabel('return time probability')
    plt.title('Gain/loss asymmetry')
    plt.legend()
    # plt.show()
    
    plt.savefig('%s/%s.png'%(folder,suffix), bbox_inches='tight')
    # plt.show()


# %% calc_single_stats
def calc_single_stats(res_real,res_focus,res_name):
    res_stats = pd.DataFrame(columns=['real',res_name])
    res_stats.loc['acf_top10_mean',:] = [np.nanmean(res_real.acf[1:11]),
                                         np.nanmean(np.nanmean(np.array(res_focus.acf)[:,1:11],axis=1),axis=0)]
    res_stats.loc['acf_top10_std',:] = [np.nan,
                                        np.nanstd(np.nanmean(np.array(res_focus.acf)[:,1:11],axis=1),axis=0)]
    res_stats.loc['alpha_mean',:] = [res_real.alpha,
                                     np.nanmean(res_focus.alpha)]
    res_stats.loc['alpha_std',:] = [np.nan,
                                     np.nanstd(res_focus.alpha)]    
    res_stats.loc['beta_mean',:] = [res_real.beta,
                                    np.nanmean(res_focus.beta)]
    res_stats.loc['beta_std',:] = [np.nan,
                                   np.nanstd(res_focus.beta)]
    res_stats.loc['lever_top10_mean',:] = [np.nanmean(res_real.lead_lag_corr[1:11]),
                                           np.nanmean(np.nanmean(np.array(res_focus.lead_lag_corr)[:,1:11],axis=1),axis=0)]
    res_stats.loc['lever_top10_std',:] = [np.nan,
                                          np.nanstd(np.nanmean(np.array(res_focus.lead_lag_corr)[:,1:11],axis=1),axis=0)]
    res_stats.loc['drho_k1_mean',:] = [res_real.d_rho[1],
                                       np.nanmean(np.array(res_focus.d_rho),axis=0)[1]]
    res_stats.loc['drho_k1_std',:] = [np.nan,
                                      np.nanstd(np.array(res_focus.d_rho),axis=0)[1]]
    res_stats.loc['gain-loss_mean',:] = [np.argmax(res_real.prob_gain)-np.argmax(res_real.prob_loss),
                                         np.nanmean(np.argmax(np.array(res_focus.prob_gain),axis=1)-np.argmax(np.array(res_focus.prob_loss),axis=1))]
    res_stats.loc['gain-loss_std',:] = [np.nan,
                                        np.nanstd(np.argmax(np.array(res_focus.prob_gain),axis=1)-np.argmax(np.array(res_focus.prob_loss),axis=1))]
    return res_stats


#%% calc_stats
def calc_stats(res_real,res_gan,res_bs,res_garch):
    res_stats = pd.DataFrame(columns=['real','wdcgan','dcgan','wgan'])
    res_stats.loc['acf_top10_mean',:] = [np.nanmean(res_real.acf[1:11]),
                                         np.nanmean(np.nanmean(np.array(res_gan.acf)[:,1:11],axis=1),axis=0),
                                         np.nanmean(np.nanmean(np.array(res_bs.acf)[:,1:11],axis=1),axis=0),
                                         np.nanmean(np.nanmean(np.array(res_garch.acf)[:,1:11],axis=1),axis=0)]

    res_stats.loc['acf_top10_std',:] = [np.nan,
                                        np.nanstd(np.nanmean(np.array(res_gan.acf)[:,1:11],axis=1),axis=0),
                                        np.nanstd(np.nanmean(np.array(res_bs.acf)[:,1:11],axis=1),axis=0),
                                        np.nanstd(np.nanmean(np.array(res_garch.acf)[:,1:11],axis=1),axis=0)]
    res_stats.loc['alpha_mean',:] = [res_real.alpha,
                                     np.nanmean(res_gan.alpha),
                                     np.nanmean(res_bs.alpha),
                                     np.nanmean(res_garch.alpha)]
    res_stats.loc['alpha_std',:] = [np.nan,
                                     np.nanstd(res_gan.alpha),
                                     np.nanstd(res_bs.alpha),
                                     np.nanstd(res_garch.alpha)]    
    res_stats.loc['beta_mean',:] = [res_real.beta,
                                    np.nanmean(res_gan.beta),
                                    np.nanmean(res_bs.beta),
                                    np.nanmean(res_garch.beta)]
    res_stats.loc['beta_std',:] = [np.nan,
                                   np.nanstd(res_gan.beta),
                                   np.nanstd(res_bs.beta),
                                   np.nanstd(res_garch.beta)]
    res_stats.loc['lever_top10_mean',:] = [np.nanmean(res_real.lead_lag_corr[1:11]),
                                           np.nanmean(np.nanmean(np.array(res_gan.lead_lag_corr)[:,1:11],axis=1),axis=0),
                                           np.nanmean(np.nanmean(np.array(res_bs.lead_lag_corr)[:,1:11],axis=1),axis=0),
                                           np.nanmean(np.nanmean(np.array(res_garch.lead_lag_corr)[:,1:11],axis=1),axis=0)]
    res_stats.loc['lever_top10_std',:] = [np.nan,
                                          np.nanstd(np.nanmean(np.array(res_gan.lead_lag_corr)[:,1:11],axis=1),axis=0),
                                          np.nanstd(np.nanmean(np.array(res_bs.lead_lag_corr)[:,1:11],axis=1),axis=0),
                                          np.nanstd(np.nanmean(np.array(res_garch.lead_lag_corr)[:,1:11],axis=1),axis=0)]
    res_stats.loc['drho_k1_mean',:] = [res_real.d_rho[1],
                                       np.nanmean(np.array(res_gan.d_rho),axis=0)[1],
                                       np.nanmean(np.array(res_bs.d_rho),axis=0)[1],
                                       np.nanmean(np.array(res_garch.d_rho),axis=0)[1]]
    res_stats.loc['drho_k1_std',:] = [np.nan,
                                      np.nanstd(np.array(res_gan.d_rho),axis=0)[1],
                                      np.nanstd(np.array(res_bs.d_rho),axis=0)[1],
                                      np.nanstd(np.array(res_garch.d_rho),axis=0)[1]]
    res_stats.loc['gain-loss_mean',:] = [np.argmax(res_real.prob_gain)-np.argmax(res_real.prob_loss),
                                         np.nanmean(np.argmax(np.array(res_gan.prob_gain),axis=1)-np.argmax(np.array(res_gan.prob_loss),axis=1)),
                                         np.nanmean(np.argmax(np.array(res_bs.prob_gain),axis=1)-np.argmax(np.array(res_bs.prob_loss),axis=1)),
                                         np.nanmean(np.argmax(np.array(res_garch.prob_gain),axis=1)-np.argmax(np.array(res_garch.prob_loss),axis=1))]
    res_stats.loc['gain-loss_std',:] = [np.nan,
                                        np.nanstd(np.argmax(np.array(res_gan.prob_gain),axis=1)-np.argmax(np.array(res_gan.prob_loss),axis=1)),
                                        np.nanstd(np.argmax(np.array(res_bs.prob_gain),axis=1)-np.argmax(np.array(res_bs.prob_loss),axis=1)),
                                        np.nanstd(np.argmax(np.array(res_garch.prob_gain),axis=1)-np.argmax(np.array(res_garch.prob_loss),axis=1))]

    return res_stats
    
    
#%% main fuction
if __name__ == '__main__':
    index_name = ["科创50"]
    #index_name = ["000300.SH", "CBA00601.CS", "AU9999.SGE"]
    
    warnings.filterwarnings("ignore")
    param_stat = Param_Stat()
    
    folder = '../Result/stat_single'
    
    #%% for all the asset
    for i, this_index in enumerate(index_name):
        #     continue
        print('start reading excel')
        _path_series = '../Data/' + this_index + '.xlsx'

        # real 
        ret_real = read_real_series(_path_series)

        print('start calculating real series')
        series = ret_real
        res_real = calc_all_stats(series,param_stat)
        plot_stat_properties(series,param_stat,res_real,folder)


        # singan series
        print('start calculating singan series')
        series = ret_sinGAN
        res_singan,res_singan_mean = batch_calc_all_stats(series,param_stat)
        plot_stat_properties(series,param_stat,res_singan_mean,folder,suffix=f'stat_{i}_singan')



        # output
        res_stats = calc_stats(res_real,res_singan,res_singan,res_singan)

        print(np.nanmean(res_real.acf[:10]),
              np.nanmean(np.array(res_singan_mean.acf)[:10]),
              # np.nanmean(np.array(res_dcgan_mean.acf)[:10]),
              # np.nanmean(np.array(res_wgan_mean.acf)[:10]))
              )
        print(np.nanstd(res_real.acf[:100]),
              np.nanstd(np.array(res_singan_mean.acf)[:100]),
              # np.nanstd(np.array(res_dcgan_mean.acf)[:100]),
              # np.nanstd(np.array(res_wgan_mean.acf)[:100]))
              )

        res_stats.to_excel('../Result/stat_single/results_stats_single_'+str(i)+'.xlsx')
