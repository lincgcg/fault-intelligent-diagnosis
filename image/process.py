import matplotlib.pyplot as plt

# 数据
data_A = [1, 2, 3, 4]
data_B = [2, 4, 6, 1]

# 创建图像
plt.figure(figsize=(10, 6))

# 绘制数据
plt.plot(data_A, data_B, 'o-')

# # 添加标签
# plt.xlabel('Data A')
# plt.ylabel('Data B')

# 显示图像
plt.show()