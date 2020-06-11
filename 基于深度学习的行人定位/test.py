import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
M, input_size, hidden_size, output_size = 4, 4, 100, 1

# 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
# 我们不需要求这几个 Variable 的导数
df_train = pd.read_csv('train.csv')
X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
y_train = df_train[['zloc']].values
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
y_train = scalar.fit_transform(y_train)
xx = torch.tensor(X_train,dtype=torch.float32)
yy = torch.tensor(y_train,dtype=torch.float32)
x = Variable(xx)
y = Variable(yy)
print(x)
# 使用 nn 包的 Sequential 来快速构建模型，Sequential可以看成一个组件的容器。
# 它涵盖神经网络中的很多层，并将这些层组合在一起构成一个模型.
# 之后，我们输入的数据会按照这个Sequential的流程进行数据的传输，最后一层就是输出层。
# 默认会帮我们进行参数初始化
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
)

# 定义损失函数
loss_fn = nn.MSELoss(reduction='sum')

## 设置超参数 ##
learning_rate = 1e-7
EPOCH = 500

# 使用optim包来定义优化算法，可以自动的帮我们对模型的参数进行梯度更新。这里我们使用的是随机梯度下降法。
# 第一个传入的参数是告诉优化器，我们需要进行梯度更新的Variable 是哪些，
# 第二个参数就是学习速率了。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.99))
## 开始训练 ##
for t in range(EPOCH):

    # 向前传播
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 显示损失
    if (t + 1) % 50 == 0:
        print(loss.item())

    # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
    optimizer.zero_grad()

    # 计算梯度
    loss.backward()

    # 更新梯度
    optimizer.step()
#PATH = "./test.pkl"
#测试准确率
model.eval()
#t_x = Variable(torch.randn(1, input_size))
y_pred_t = model(x)
y_pred_t = scalar.inverse_transform(y_pred_t)
y_r = scalar.inverse_transform(y)
#loss2 = loss_fn(y_pred, y_r)
loss_fn2 = nn.L1Loss(reduction='mean')
loss2 = loss_fn2(y_pred_t, y_r)
df_test = pd.read_csv('data/test.csv')
X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
y_test = df_test[['zloc']].values

# standardized data
scalar2 = StandardScaler()
X_test = scalar2.fit_transform(X_test)
x_t = torch.tensor(X_test,dtype=torch.float32)
y_testt = torch.tensor(y_test,dtype=torch.float32)
x_tt = Variable(x_t)
y_testt = Variable(y_testt)
y_pred_tt = model(x_tt)
y_pred_test = scalar.inverse_transform(y_pred_tt.detach().numpy())
loss3 = loss_fn2(y_pred_test, y_testt)
print("loss2: "+loss2)
print("loss3:"+loss3)
'''y_test = scalar.fit_transform(y_test)
y_pred = scalar.inverse_transform(y_pred)
test_xx = torch.tensor(X_train,dtype=torch.float32)
test_yy = torch.tensor(y_train,dtype=torch.float32)
#test_x = Variable(test_xx)
#test_y = Variable(test_yy )
print(model.forward(t_x))'''
#torch.save(model, PATH)