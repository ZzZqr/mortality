import streamlit as st
import joblib
import numpy as np
import pandas as pd
# 定义模型文件的完整路径
model_path = 'model/xgb_model.joblib'

# 加载模型
loaded_model = joblib.load(model_path)
# 加载之前训练好的模型

# 创建Streamlit应用程序界面
st.title('Random Forest Model Deployment')
st.write('Enter some input features to make predictions:')

# 创建输入框用于用户输入特征
new_onset_shock = st.selectbox('new_onset_shock', options=["Yes", "No"])
ALT = st.number_input('ALT (U/L)', min_value=5.0, max_value=800.0, step=0.1)
SOFA = st.number_input('SOFA', min_value=0, max_value=9, step=1)
MDR = st.selectbox('MDR', options=["Yes", "No"])
pre_shock = st.selectbox('pre_shock', options=["Yes", "No"])
area_of_burn = st.number_input('area_of_burn (%)', min_value=20, max_value=100, step=1)
three = st.number_input('Ⅲ (%)', min_value=0, max_value=80, step=1)
sepsis = st.selectbox('sepsis', options=["Yes", "No"])
type_of_burn = st.selectbox('type_of_burn ', min_value=1, max_value=5, step=1)

if new_onset_shock == "Yes":
    new_onset_shock = 1
else:
    new_onset_shock = 0

if MDR == "Yes":
    MDR = 1
else:
    MDR = 0

if pre_shock == "Yes":
    pre_shock = 1
else:
    pre_shock = 0

if sepsis == "Yes":
    sepsis = 1
else:
    sepsis = 0

# 定义应用程序行为
if st.button('Predict'):
    # 将用户输入的特征转换为模型所需的输入格式
    input_features = np.array([[sepsis, type_of_burn, area_of_burn, three, pre_shock, new_onset_shock, MDR, ALT, SOFA]])
    df = pd.DataFrame(input_features, columns=["sepsis", "type_of_burn", "area_of_burn", "III", "pre_shock", "new_onset_shock", "MDR", "ALT", "SOFA"])
    # 使用加载的模型进行预测
    prediction = loaded_model.predict_proba(df)
    print(prediction)

    # 显示预测结果
    st.write(f'Mortality Probability Prediction:{prediction[0][0]}')
# # 运行Streamlit应用程序
# if __name__ == '__main__':
#     st.run()

