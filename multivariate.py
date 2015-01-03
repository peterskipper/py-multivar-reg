import pandas as pd
import scipy
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

loanData = pd.read_csv("LoanStats3c.csv", skiprows=1)

# Drop na rows from int_rate etc. (2-3 rows total)
loanData.dropna(subset=['int_rate', 'annual_inc', 'home_ownership'],
                inplace=True)
loanData['int_rate'] = map(lambda x: float(str(x)[:-1]), loanData['int_rate'])
loanData = pd.core.reshape.get_dummies(loanData, columns=['home_ownership'])

# Initial Model
y = np.matrix(loanData['int_rate']).transpose()
x1 = np.matrix(loanData['annual_inc']).transpose()
X = sm.add_constant(x1)
model = sm.OLS(y, X)
f1 = model.fit()
f1.summary()

# Fuller model
x2 = np.matrix(loanData['home_ownership_MORTGAGE']).transpose()
x3 = np.matrix(loanData['home_ownership_RENT']).transpose()
x4 = np.matrix(loanData['home_ownership_OWN']).transpose()
x = np.column_stack([x1, x2, x3, x4])
X = sm.add_constant(x)
model2 = sm.OLS(y, X)
f2 = model2.fit()
import pdb
pdb.set_trace()
f2.summary()

# Interactions
# R-style formulas
f3 = smf.ols(formula=('int_rate ~ annual_inc + home_ownership_MORTGAGE + '
                      'home_ownership_RENT + home_ownership_OWN + '
                      'annual_inc:home_ownership_MORTGAGE + '
                      'annual_inc:home_ownership_RENT '
                      '+ annual_inc:home_ownership_OWN'), data=loanData).fit()
f3.summary()
