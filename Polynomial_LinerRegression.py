from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X = np.arange(0, 100, 1).reshape(-1, 1)
Y = 0.01 * X**2 + np.random.rand(100, 1) * 10

# 可视化原始数据
plt.scatter(X, Y)
plt.title("Original Data")
plt.show()

# 使用多项式特征
pf = PolynomialFeatures(degree=2)
X_poly = pf.fit_transform(X)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_poly, Y, test_size=0.3, random_state=23)

# 使用线性回归模型拟合数据
model = LinearRegression()
model.fit(x_train, y_train)

# 对预测的新数据进行多项式变换
X_pre = np.arange(0, 100, 1).reshape(-1, 1)
X_pre_poly = pf.transform(X_pre)

# 预测结果
Y_pre = model.predict(X_pre_poly)

# 可视化预测结果
plt.scatter(X, Y, label="Original Data")
plt.plot(X_pre, Y_pre, label="Predictions", color='red')
plt.title("Polynomial Regression Predictions")
plt.legend()
plt.show()
