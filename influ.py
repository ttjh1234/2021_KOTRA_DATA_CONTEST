# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 19:31:24 2021

@author: 82104
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

xg=np.array([1,1,2,2,3,3,4,4,9])
yg=np.array([1,1.5,2,2.5,3,3.5,4,4.5,1])

xg_rev=np.array([1,1,2,2,3,3,4,4])
yg_rev=np.array([1,1.5,2,2.5,3,3.5,4,4.5])

sns.set_style('white')
fig=plt.figure(figsize=(15,5))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

sns.regplot(xg,yg,ci=None,ax=ax1)
sns.regplot(xg_rev,yg_rev,ci=None,ax=ax2)
ax1.set_title('Before Elimination Influential OBS')
ax2.set_title('After Elimination Influential OBS')

ax1.set_xlim(0,10)
ax2.set_xlim(0,10)
ax1.set_ylim(0,5)
ax2.set_ylim(0,5)