import numpy as np
from sklearn import datasets

class SVM:
    def __init__(self,X,Y,eps=1e-3,TAU = 1e-12,gamma = 1.0,C = 1.0,max_iter = 1000000000):
        self.alpha = None  ###初始化为0
        self.G = None      ###初始化为-1
        self.Q = None
        self.L = None
        self.alpha_Y = None
        self.b = None
        self.eps = eps
        self.TAU = TAU
        self.Y = Y
        self.X = X
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter

    ###模型初始化
    def Initial(self):
        self.L = self.Y.shape[0]
        self.alpha = np.zeros(self.L)
        self.G = np.zeros(self.L)
        self.Q = np.zeros((self.L,self.L))
        if self.gamma == 1.0:
            self.gamma = 1.0/self.X.shape[1]
        for i in range(self.L):
            self.G[i] -= 1
            for j in range(self.L):
                self.Q[i][j] = self.Y[i]*self.Y[j]*np.exp(- 1 / self.gamma * np.sum((self.X[i] - self.X[j]) ** 2))

    ###SMO中i，j选择
    def select_sub(self):
        m = -float('inf')  ##m(alpha)
        M = -float('inf')  ##M(alpha)
        m_idx = -1
        M_idx = -1
        sub_min = float('inf')  ##sub问题最小值

        ####select m_idx
        for i in range(self.L):  ####数据size
            if self.Y[i] == 1:
                if self.alpha[i] < self.C:
                    if -self.G[i] >= m:
                        m = -self.G[i]
                        m_idx = i
            else:
                if self.alpha[i] > 0:
                    if self.G[i] >= m:
                        m = self.G[i]
                        m_idx = i

        ####select M_idx
        for i in range(self.L):
            if self.Y[i] == 1:
                if self.alpha[i] > 0:
                    grad_diff = m + self.G[i]
                    if self.G[i] >= M:
                        M = self.G[i]
                    if grad_diff > 0:
                        quad_coef = self.Q[m_idx][m_idx] + self.Q[i][i] - 2.0 * self.Y[m_idx] * self.Q[m_idx][i]
                        if quad_coef > 0:
                            sub_value = - grad_diff ** 2 / quad_coef
                        else:
                            sub_value = - grad_diff ** 2 / self.TAU
                        if sub_value <= sub_min:
                            M_idx = i
                            sub_min = sub_value
            ####y != 1
            else:
                if self.alpha[i] < self.C:
                    grad_diff = m - self.G[i]
                    if -self.G[i] >= M:
                        M = -self.G[i]
                    if grad_diff > 0:
                        quad_coef = self.Q[m_idx][m_idx] + self.Q[i][i] + 2.0 * self.Y[m_idx] * self.Q[m_idx][i]
                        if quad_coef > 0:
                            sub_value = - grad_diff ** 2 / quad_coef
                        else:
                            sub_value = - grad_diff ** 2 / self.TAU
                        if sub_value <= sub_min:
                            M_idx = i
                            sub_min = sub_value

            if m + M < self.eps or M_idx == -1:
                return -1, -1
            return m_idx, M_idx

    ###alpha更新
    def update(self):
        for i in range(self.max_iter):
            m_idx, M_idx = self.select_sub()
            if m_idx == -1 and M_idx == -1:
                return
            if self.Y[m_idx] != self.Y[M_idx]:
                quad_coef = self.Q[m_idx][m_idx] + self.Q[M_idx][M_idx] + 2.0 * self.Q[m_idx][M_idx]
                if quad_coef <= 0:
                    quad_coef = self.TAU
                delta = (-self.G[m_idx] - self.G[M_idx]) / quad_coef
                diff = self.alpha[m_idx] - self.alpha[M_idx]

                ###记录更新前的alpha
                old_alpha = np.zeros((1, self.L))
                old_alpha[0] = self.alpha

                ###alpha  update
                self.alpha[m_idx] += delta
                self.alpha[M_idx] += delta

                ###clipping
                if diff > 0:
                    ###region 3
                    if self.alpha[M_idx] < 0:
                        self.alpha[M_idx] = 0
                        self.alpha[m_idx] = diff
                    ###region 1
                    elif self.alpha[m_idx] > self.C:
                        self.alpha[m_idx] = self.C
                        self.alpha[M_idx] = self.C - diff
                else:
                    ###region 4
                    if self.alpha[m_idx] < 0:
                        self.alpha[m_idx] = 0
                        self.alpha[M_idx] = -diff
                    ###region 2
                    elif self.alpha[M_idx] > self.C:
                        self.alpha[M_idx] = self.C
                        self.alpha[m_idx] = self.C + diff
            else:
                quad_coef = self.Q[m_idx][m_idx] + self.Q[M_idx][M_idx] - 2.0 * self.Q[m_idx][M_idx]
                if quad_coef <= 0:
                    quad_coef = self.TAU
                delta = (self.G[m_idx] - self.G[M_idx]) / quad_coef
                total = self.alpha[m_idx] + self.alpha[M_idx]
                self.alpha[m_idx] -= delta
                self.alpha[M_idx] += delta

                if total > self.C:
                    ###region 2
                    if self.alpha[M_idx] > self.C:
                        self.alpha[M_idx] = self.C
                        self.alpha[m_idx] = total - self.C
                    ###region 1
                    elif self.alpha[m_idx] > self.C:
                        self.alpha[m_idx] = self.C
                        self.alpha[M_idx] = total - self.C
                else:
                    ###region 4
                    if self.alpha[m_idx] < 0:
                        self.alpha[m_idx] = 0
                        self.alpha[M_idx] = total
                    ###region 3
                    elif self.alpha[M_idx] < 0:
                        self.alpha[M_idx] = 0
                        self.alpha[m_idx] = total

            delta_alpha_m = self.alpha[m_idx] - old_alpha[0][m_idx]
            delta_alpha_M = self.alpha[M_idx] - old_alpha[0][M_idx]

            for i in range(self.L):
                self.G[i] += self.Q[m_idx][i] * delta_alpha_m + self.Q[M_idx][i] * delta_alpha_M

    ####计算y_i*alpha_i 和 b值
    def calculate(self):
        self.alpha_Y = self.alpha*self.Y
        num = 0
        total = 0
        ub = float('inf')
        lb = -float('inf')
        Y_G = self.Y * self.G
        for i in range(self.L):
            if self.alpha[i] == self.C:
                if self.Y[i] == -1:
                    ub = min(ub, Y_G[i])
                else:
                    lb = max(lb, Y_G[i])
            elif self.alpha[i] == 0.0:
                if self.Y[i] == 1:
                    ub = min(ub, Y_G[i])
                else:
                    lb = max(lb, Y_G[i])

            else: ### self.alpha < self.C and self.alpha > 0:
                num += 1
                total += -Y_G[i]

        if num > 0:
            self.b = total/num
        else:
            self.b = -(ub + lb)/2

    ###模型训练
    def fit(self):
        self.Initial()
        self.update()
        self.calculate()

    ###模型预测
    def predict(self,x):
        len = x.shape[0]
        value = np.zeros(len)
        value += self.b
        for i in range(len):
            for j in range(self.L):
                value[i] += self.alpha_Y[j] * np.exp(- 1 / self.gamma * np.sum((x[i] - self.X[j]) ** 2))

        for i in range(len):
            if value[i] > 0:
                value[i] = 1
            else:
                value[i] = -1
        return value

# iris = datasets.load_iris()
# X = iris["data"][:, (2,3)]
# Y = (iris["target"] == 2).astype( np.float64 )
# for i in range(len(Y)):
#     if Y[i] == 0:
#         Y[i] = -1
# S1 = SVM(X,Y)
# S1.fit()
# res = S1.predict(np.array([[5.5, 1.7]]))
# print(res)
