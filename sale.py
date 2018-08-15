# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 20:10:50 2018

@author: todd
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.preprocessing import Imputer

from sklearn.model_selection  import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import Lasso

#读csv数据
train = pd.read_csv("D:/train.csv")
#print(train.dtypes)
#删除索引
train.drop(['Id'],axis=1, inplace=True)

#查找空值并补全
aa = train.isnull().sum()
aa[aa>0].sort_values(ascending=False)

#根据其他相关变量分组，在组内用中位数补全
train["LotAreaCut"] = pd.qcut(train.LotArea,10)
train['LotFrontage'] = train.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#用0补全
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    train[col].fillna(0, inplace=True)

#用None补全
cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    train[col].fillna("None", inplace=True)

#用众数补全
cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    train[col].fillna(train[col].mode()[0], inplace=True)

#数值类型转为字符转类型    
NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    train[col]=train[col].astype(str)

#有序离散变量的标签编码 
def map_values():
    train["oMSSubClass"] = train.MSSubClass.map({'180':1, 
                                        '30':2, '45':2, 
                                        '190':3, '50':3, '90':3, 
                                        '85':4, '40':4, '160':4, 
                                        '70':5, '20':5, '75':5, '80':5, '150':5,
                                        '120': 6, '60':6})
    
    train["oMSZoning"] = train.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})
    
    train["oNeighborhood"] = train.Neighborhood.map({'MeadowV':1,
                                               'IDOTRR':2, 'BrDale':2,
                                               'OldTown':3, 'Edwards':3, 'BrkSide':3,
                                               'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,
                                               'NPkVill':5, 'Mitchel':5,
                                               'SawyerW':6, 'Gilbert':6, 'NWAmes':6,
                                               'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,
                                               'Veenker':8, 'Somerst':8, 'Timber':8,
                                               'StoneBr':9,
                                               'NoRidge':10, 'NridgHt':10})
    
    train["oCondition1"] = train.Condition1.map({'Artery':1,
                                           'Feedr':2, 'RRAe':2,
                                           'Norm':3, 'RRAn':3,
                                           'PosN':4, 'RRNe':4,
                                           'PosA':5 ,'RRNn':5})
    
    train["oBldgType"] = train.BldgType.map({'2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})
    
    train["oHouseStyle"] = train.HouseStyle.map({'1.5Unf':1, 
                                           '1.5Fin':2, '2.5Unf':2, 'SFoyer':2, 
                                           '1Story':3, 'SLvl':3,
                                           '2Story':4, '2.5Fin':4})
    
    train["oExterior1st"] = train.Exterior1st.map({'BrkComm':1,
                                             'AsphShn':2, 'CBlock':2, 'AsbShng':2,
                                             'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                                             'BrkFace':4, 'Plywood':4,
                                             'VinylSd':5,
                                             'CemntBd':6,
                                             'Stone':7, 'ImStucc':7})
    
    train["oMasVnrType"] = train.MasVnrType.map({'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3})
    
    train["oExterQual"] = train.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    train["oFoundation"] = train.Foundation.map({'Slab':1, 
                                           'BrkTil':2, 'CBlock':2, 'Stone':2,
                                           'Wood':3, 'PConc':4})
    
    train["oBsmtQual"] = train.BsmtQual.map({'Fa':2, 'None':1, 'TA':3, 'Gd':4, 'Ex':5})
    
    train["oBsmtExposure"] = train.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})
    
    train["oHeating"] = train.Heating.map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5})
    
    train["oHeatingQC"] = train.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    train["oKitchenQual"] = train.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    train["oFunctional"] = train.Functional.map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})
    
    train["oFireplaceQu"] = train.FireplaceQu.map({'None':1, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    train["oGarageType"] = train.GarageType.map({'CarPort':1, 'None':1,
                                           'Detchd':2,
                                           '2Types':3, 'Basment':3,
                                           'Attchd':4, 'BuiltIn':5})
    
    train["oGarageFinish"] = train.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})
    
    train["oPavedDrive"] = train.PavedDrive.map({'N':1, 'P':2, 'Y':3})
    
    train["oSaleType"] = train.SaleType.map({'COD':1, 'ConLD':1, 'ConLI':1, 'ConLw':1, 'Oth':1, 'WD':1,
                                       'CWD':2, 'Con':3, 'New':3})
    
    train["oSaleCondition"] = train.SaleCondition.map({'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4})            
    return "Done!"

map_values()

# drop two unwanted columns
train.drop("LotAreaCut",axis=1,inplace=True)
train.drop(['SalePrice'],axis=1,inplace=True)

#有序离散变量的标签编码
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass   
    def fit(self,X,y=None):
        return self  
    def transform(self,X):
        lab=LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X
    
#对类别变量one-hot编码，对偏度大于0.5的数值型变量取对数
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self,skew=0.5):
        self.skew = skew  
    def fit(self,X,y=None):
        return self    
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X

# build pipeline
pipe = Pipeline([
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1)),
    ])

train1 = train.copy()

data_pipe = pipe.fit_transform(train1)

scaler = RobustScaler()

X = data_pipe

y= pd.read_csv("D:/train.csv").SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(y)

#寻优lasso参数alpha
class grid():
    def __init__(self,model):
        self.model = model
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        #print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))

#grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})

#建立lasso对象
lasso=Lasso(alpha=0.0006)

#拟合
lasso.fit(X_scaled,y_log)

#自变量系数
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=data_pipe.columns)

#过滤
print(FI_lasso[FI_lasso["Feature Importance"]>=0.05].sort_values("Feature Importance"))


#import pandas as pd
#df = pd.DataFrame(X_scaled)
#df_corr = df.corr()
#print(df_corr)
#import matplotlib.pyplot as mp, seaborn
#seaborn.heatmap(df_corr, center=0, annot=True)
#mp.show()

import pandas as pd
h = pd.read_csv("D:/train.csv")
df_corr = h.corr()

print(df_corr)
#df_corr.SalePrice.sort_values()
import matplotlib.pyplot as mp, seaborn
seaborn.heatmap(df_corr, center=0, annot=True)
mp.show()
