import pandas as pd
import numpy as np
import scipy.stats as stats
import wquantiles
import statsmodels.robust as robust
from matplotlib import pyplot as plt

# 기본적인 위치추정 기법

state = pd.read_csv('data/state.csv')
print(state['Population'].mean())                        # 평균
print(state['Population'].median())                      # 중간값 (평균에 비해 robust) : 50% 백분위수와 같음
print(stats.trim_mean(state['Population'], 0.1))         # 절사평균 : 상하위 n% 빼고 평균 (평균에 비해 robust)
print(np.average(state['Murder.Rate'], weights=state['Population']))            # 가중평균
print(wquantiles.median(state['Murder.Rate'], weights=state['Population']))     # 가중중간값

# 변이 추정 - 편차 관련

print(state['Population'].std())   # 보편적 변이 추정 표준 편차.(극단값에 robust 하지않음, 특이값에 민감) 분산은 제곱으로 표현되므로 동일한 스케일에 있는 표준편차가 해석에 유리
print(robust.scale.mad(state['Population']))    # 중위절대편차(MAD) : 극단값에 영향받지 않고, 특이값에 robust

# 변이 추정 - 백분위수 관련

print(state['Population'].quantile(0.75) - state['Population'].quantile(0.25))  # 사분위 범위(IQR) , 75번째 백분위수 - 25번째 백분위수
print(state['Population'].quantile([0.05, 0.25, 0.5, 0.75, 0.95]))   # 5, 25, 50, 75, 95 번째 백분위수( 5% 25% 50%, 75%, 95%)

# 데이터 분포 시각화

ax = (state['Population']/1_000_000).plot.box()     # boxplot 상자그림 (판다스의 데이터프레임에서 제공하는 메서드로 시각화 조정을 위해 축객체(ax)를 반환하는게 일반적)
ax.set_ylabel('Population (millions)')
plt.show()

binPopulation = pd.cut(state['Population'], 10)     # 도수분포표 (전체를 10등분으로 나누고 각 구간에 몇개의 개체가 있는지 보여주는 표)
print(binPopulation.value_counts())

ax = (state['Population']/1_000_000).plot.hist(figsize=(8, 8))      # 도수분포표를 히스토그램으로 시각화
ax.set_xlabel('Population (millions)')
plt.show()
