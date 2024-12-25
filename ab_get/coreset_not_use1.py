import numpy
from gurobipy import *
import pickle
import numpy.matlib
import time
import pickle
import bisect


def solve_fac_loc(xx, yy, subset, n, budget):
    # 创建一个Gurobi优化模型，目标是最小化k-Center问题
    model = Model("k-center")

    # 定义变量
    x = {}  # 连接变量：x[i,j]表示i和j之间是否连接
    y = {}  # 设施选择变量：y[i]表示是否选择i作为设施
    z = {}  # 辅助变量：z[i]表示是否点i是一个损失点

    # 定义z_i变量：是否点i为损失点（值为0或1），目标是最小化z的总和
    for i in range(n):
        z[i] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(i))

    m = len(xx)  # m是数据点的数量
    # 为每个点创建连接变量x[_x, _y]，如果两个点之间有连接，则x为1
    for i in range(m):
        _x = xx[i]
        _y = yy[i]
        # 为设施选择变量y[_y]创建一个二进制变量，表示点_y是否是设施
        if _y not in y:
            if _y in subset:
                y[_y] = model.addVar(
                    obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y)
                )  # 如果_y在子集subset中，y值为1
            else:
                y[_y] = model.addVar(obj=0, vtype="B", name="y_{}".format(_y))
        # 定义连接变量x[_x,_y]，表示点_x和点_y之间是否有连接
        x[_x, _y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x, _y))

    model.update()

    # 添加k-Center约束：设施数量不超过budget
    coef = [1 for j in range(n)]
    var = [y[j] for j in range(n)]
    model.addConstr(LinExpr(coef, var), "=", rhs=budget + len(subset), name="k_center")

    # 强约束：如果_x和_y之间有连接，y[_y]必须是设施
    for i in range(m):
        _x = xx[i]
        _y = yy[i]
        model.addConstr(x[_x, _y], "<", y[_y], name="Strong_{},{}".format(_x, _y))

    # 创建设施分配约束：每个点必须分配给一个设施
    yyy = {}
    for v in range(m):
        _x = xx[v]
        _y = yy[v]
        if _x not in yyy:
            yyy[_x] = []
        if _y not in yyy[_x]:
            yyy[_x].append(_y)

    for _x in yyy:
        coef = []
        var = []
        for _y in yyy[_x]:
            coef.append(1)
            var.append(x[_x, _y])
        coef.append(1)  # 辅助变量z[_x]的约束
        var.append(z[_x])
        model.addConstr(LinExpr(coef, var), "=", 1, name="Assign{}".format(_x))

    model.__data = x, y, z
    return model


# 加载数据并计算距离矩阵
data = pickle.load(open("feature_vectors_pickled"))  # 加载特征向量数据
budget = 10000  # 设置预算

start = time.clock()
num_images = data.shape[0]  # 数据点数目
dist_mat = numpy.matmul(data, data.transpose())  # 计算点之间的距离矩阵

# 通过距离矩阵进行转换，计算真实的距离
sq = numpy.array(dist_mat.diagonal()).reshape(num_images, 1)
dist_mat *= -2
dist_mat += sq
dist_mat += sq.transpose()

elapsed = time.clock() - start
print("Time spent in (distance computation) is: ", elapsed)

# 设置初始参数
num_images = 50000
subset = [i for i in range(1)]  # 初始化子集
ub = UB  # 上界 1 * 10-4 * n
lb = ub / 2.0  # 下界
max_dist = ub  # 最大距离

# 计算距离矩阵的索引和值
_x, _y = numpy.where(dist_mat <= max_dist)
_d = dist_mat[_x, _y]

# 创建并优化Gurobi模型
model = solve_fac_loc(_x, _y, subset, num_images, budget)
x, y, z = model.__data

# 设置精度
delta = 1e-7
while ub - lb > delta:  # 当上下界的差距小于delta时停止优化
    print("State", ub, lb)

    # 当前半径
    cur_r = (ub + lb) / 2.0
    viol = numpy.where(_d > cur_r)  # 查找违反当前半径约束的边
    new_max_d = numpy.min(_d[_d >= cur_r])  # 新的最大距离
    new_min_d = numpy.max(_d[_d <= cur_r])  # 新的最小距离
    print("If it succeeds, new max is:", new_max_d, new_min_d)

    # 将违反约束的边的上界设置为0
    for v in viol[0]:
        x[_x[v], _y[v]].UB = 0

    model.update()
    r = model.optimize()  # 执行优化

    # 检查模型状态
    if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        failed = True
        print("Infeasible")  # 如果不可行，表示当前半径不合适
    elif sum([z[i].X for i in range(len(z))]) > 0:
        failed = True
        print("Failed")  # 如果有损失点，表示当前解无效
    else:
        failed = False

    # 如果失败，则更新下界并将违反约束的边重新设置
    if failed:
        lb = max(cur_r, new_max_d)
        for v in viol[0]:
            x[_x[v], _y[v]].UB = 1
    else:
        print("sol founded", cur_r, lb, ub)  # 如果找到一个可行解，则更新上界
        ub = min(cur_r, new_min_d)
        model.write("s_{}_solution_{}.sol".format(budget, cur_r))  # 保存解
