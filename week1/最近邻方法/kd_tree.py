import numpy as np


class KDNode(object):
    def __init__(self, value, split, left, right):
        self.value = value
        self.split = split
        self.right = right
        self.left = left


class MyKDTree(object):
    def __init__(self, data):
        self.data = data
        k = len(data[0])
        indexs = np.array(range(len(data)))  # 添加一列下标，用于搜索时返回下标
        data_with_index = np.column_stack((self.data, indexs))

        def CreateNode(split, data_set):
            if data_set == []:
                return None
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2  # 整除2
            median = data_set[split_pos]
            split_next = (split + 1) % k

            return KDNode(median, split, CreateNode(split_next, data_set[: split_pos]),
                          CreateNode(split_next, data_set[split_pos + 1:]))

        data = list(data_with_index)
        self.root = CreateNode(0, data)

    def search(self, x, count=1):
        nearest = []
        for i in range(count):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)


        def recurve(node):
            if node is not None:
                axis = node.split
                daxis = x[axis] - node.value[axis]
                if daxis < 0:
                    recurve(node.left)
                else:
                    recurve(node.right)

                dist = np.sqrt(np.sum((np.array(x) - node.value[:-1]) ** 2))
                for i, d in enumerate(self.nearest):
                    if d[0] < 0 or dist < d[0]:  # 如果当前nearest内i处未标记（-1），或者新点与x距离更近
                        self.nearest = np.insert(self.nearest, i, [dist, node.value[-1]], axis=0)  # 插入比i处距离更小的
                        self.nearest = self.nearest[:-1]
                        break
                # 找到nearest集合里距离最大值的位置，为-1值的个数
                n = list(self.nearest[:, 0]).count(-1)
                # 切分轴的距离比nearest中最大的小（存在相交）
                if self.nearest[-n - 1, 0] > abs(daxis):
                    if daxis < 0:  # 相交，x[axis]< node.data[axis]时，去右边（左边已经遍历了）
                        recurve(node.right)
                    else:  # x[axis]> node.data[axis]时，去左边，（右边已经遍历了）
                        recurve(node.left)

        recurve(self.root)
        ret_d = np.hsplit(self.nearest, 2)[0].flatten()
        ret_i = np.hsplit(self.nearest, 2)[1].astype(int).flatten()

        return ret_d, ret_i

    def query(self, X, k=1):
        if k > len(self.data):
            k = len(self.data)
        indices = []
        distance = []
        for x in X:
            d, i = self.search(x, k)
            distance.append(d)
            indices.append(i)
        return np.array(distance), np.array(indices)


if __name__ == '__main__':
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd = MyKDTree(data)

    # [3, 4.5]最近的3个点
    n = kd.search( [3, 4.5], 3)
    print(n)

