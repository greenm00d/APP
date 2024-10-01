import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV #LR
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn.svm import SVC #SVM
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #RF GBTD
from xgboost import XGBClassifier #XGBOOST
from lightgbm import LGBMClassifier #LGBM
from sklearn.tree import DecisionTreeClassifier#决策数
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
pd.options.display.max_columns = None #显示完整打印数据
pd.options.display.max_rows = None
warnings.filterwarnings("ignore") # 忽略警告
plt.rcParams['figure.dpi'] = 300
plt.rc('font',family='Times New Roman')

df = pd.read_excel('4.xlsx', engine='openpyxl',usecols=np.arange(0,150))

# 划分特征和目标变量
X = df.drop(['birth'], axis=1)
y = df['birth']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=3833, stratify=y)

xgb = XGBClassifier(
                        random_state=50,
                        max_depth=2,
                        learning_rate=0.2,
                        n_estimators=30,importance_type=["gain", "weight", "cover"])
xgb.fit(X_train, y_train)

best_model = xgb

import joblib
# 保存模型
joblib.dump(best_model , 'XGBoost.pkl')

#安装完成后这些包应该就能正常导入了
import streamlit as st
from streamlit_echarts import st_echarts
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts
import pandas as pd
import numpy as np

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature options
embryos = {    
    0: 'no (1)',    
    1: 'yes (2)'
}

# Define feature options
biochemical = {    
    0: 'no (1)',    
    1: 'yes (2)'
}

# Define feature names
feature_names = [    
    "age", "infertility_time", "menarche_age", "AMH", "gn_dose", "gn_days", "oocyte", "embryos", "biochemical"
]

# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# age: numerical input
infertility_time = st.number_input("infertility_time:", min_value=1, max_value=120, value=50)

# age: numerical input
menarche_age = st.number_input("menarche_age:", min_value=1, max_value=120, value=50)

# age: numerical input
AMH = st.number_input("AMH:", min_value=1, max_value=120, value=50)

# age: numerical input
gn_dose = st.number_input("gn_dose:", min_value=1, max_value=120, value=50)

# age: numerical input
gn_days = st.number_input("gn_days:", min_value=1, max_value=120, value=50)

# age: numerical input
oocyte = st.number_input("oocyte:", min_value=1, max_value=120, value=50)

# cp: categorical selection
embryos = st.selectbox("embryos:", options=list(embryos.keys()), format_func=lambda x: embryos[x])

# cp: categorical selection
biochemical = st.selectbox("biochemical:", options=list(biochemical.keys()), format_func=lambda x: biochemical[x])

# Process inputs and make predictions
feature_values = [age, infertility_time, menarche_age, AMH, gn_dose, gn_days, oocyte, embryos, biochemical]
features = np.array([feature_values])

if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:        
        advice = (            
            f"According to our model, you have a high risk of heart disease. "            
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "            
            "While this is just an estimate, it suggests that you may be at significant risk. "            
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "            
            "to ensure you receive an accurate diagnosis and necessary treatment."        
        )
    else:        
        advice = (            
            f"According to our model, you have a low risk of heart disease. "            
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "            
            "However, maintaining a healthy lifestyle is still very important. "            
            "I recommend regular check-ups to monitor your heart health, "            
            "and to seek medical advice promptly if you experience any symptoms."        
        )
    st.write(advice)
    
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    
    st.image("shap_force_plot.png")