import random
import math
class BiasSVD:
    def __init__(self, data, k, learning_rate = 0.1, lmd = 0.1, max_iter = 100) -> None:
        self.k = k 
        self.P = dict()
        self.Q = dict()
        self.bu = dict()
        self.bi = dict()
        self.lr = learning_rate
        self.lmd = lmd # the regularization coefficient
        self.max_iter = max_iter
        self.rating_matrix = data

        for user, items in self.rating_matrix.items():
                # 初始化矩阵P和Q, 随机数需要和1/sqrt(F)成正比
                self.P[user] = [random.random() / math.sqrt(self.k) for x in range(0, k)]
                self.bu[user] = 0
                for item, rating in items.items():
                    if item not in self.Q:
                        self.Q[item] = [random.random() / math.sqrt(self.k) for x in range(0, k)]
                        self.bi[item] = 0
    def predict(self, user, item):
        return sum(self.P[user][f] * self.Q[item][f] for f in range(0, self.k)) + self.bu[user] + self.bi[
            item] + self.mu
    def train(self):
         cnt, mu_sum = 0, 0
         for user, items in self.rating_matrix.items():
            for item, rui in items.items():
                mu_sum, cnt = mu_sum + rui, cnt + 1
         self.mu = mu_sum / cnt
         for _ in range(self.max_iter):
            for user, item in self.rating_matrix.items():
                 for item, r_ui in item.items():
                    r_hat = self.predict(user, item)
                    e_ui = r_ui - r_hat

                    # update the parameters
                    self.bu[user] += self.lr * (e_ui - self.lmd * self.bu[user])
                    self.bi[item] += self.lr * (e_ui - self.lmd * self.bi[item])
                    for f in range(self.k):
                            self.P[user][f] = self.lr* (e_ui * self.Q[item][f] - self.lmd * self.P[user][f])
                            self.Q[item][f] = self.lr* (e_ui * self.P[user][f] - self.lmd * self.Q[item][f])
            self.lr *=0.1
def loadData():
    rating_data={1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
           2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
           3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
           4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
           5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
          }
    return rating_data

rating_data = loadData()
model = BiasSVD(rating_data, 10)
model.train()
print(model.predict(1, 'E'))