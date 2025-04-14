import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from scipy.interpolate import UnivariateSpline
import numpy as np

DATA_PATH = "../data"
DATASET_NAME_list = ["glove"]
# DATASET_NAME_list = ["sift"]

def cal_t(dataset_name):
    # 预估每次距离计算花费的时间t
    # qps = 总查询次数 / 总时间(秒)
    # 设一次距离计算的时间是t，总时间=平均每次查询的距离比较次数*t*总查询次数
    # 平均每次查询的距离比较次数=avg_cmps
    # 所以，qps = 1 / (avg_cmps * t), t=1/(qps*avg_cmps)
    SCALE = 100000.0
    t_list = []
    path = os.path.join(DATA_PATH, dataset_name, "res")
    for file in os.listdir(path):
        if not file.endswith(".csv"):
            continue

        with open(os.path.join(path, file), 'r') as f:
            for _ in range(5):
                line = f.readline()
            while line and len(line) > 3:
                tmp = line.split(',')
                qps = float(tmp[3])
                avg_cmps = float(tmp[4])
                t_list.append(SCALE / (qps * avg_cmps))
                line = f.readline()

    avg_t = np.mean(t_list)
    # t = avg_t/SCALE, qps=SCALE/(avg_cmps * avg_t)
    return avg_t, SCALE

def get_max_recall(dataset_name):
    recall_list = []
    path = os.path.join(DATA_PATH, dataset_name, "res")
    for file in os.listdir(path):
        if not file.endswith(".csv"):
            continue
        recall = []
        with open(os.path.join(path, file), 'r') as f:
            for _ in range(5):
                line = f.readline()
            while line and len(line) > 3:
                tmp = line.split(',')
                r = float(tmp[2])
                recall.append(r)
                line = f.readline()
            recall_list.append(np.max(recall))
    return np.min(recall_list)

def main():
    for DATASET_NAME in DATASET_NAME_list:
        avg_t, SCALE = cal_t(DATASET_NAME)
        path = os.path.join(DATA_PATH, DATASET_NAME, "res")
        data = {
            "name": [],
            "qps": [],
            "R@10": []
        }
        max_recall = get_max_recall(DATASET_NAME)

        vs_res_dict = {}

        for file in os.listdir(path):
            if not file.endswith(".csv"):
                continue

            if True:
                recall = []
                qps = []
                with open(os.path.join(path, file), 'r') as f:
                    for _ in range(5):
                        line = f.readline()
                    while line and len(line) > 3:
                        tmp = line.split(',')
                        r = float(tmp[2])
                        avg_cmps = float(tmp[4])
                        q = SCALE / (avg_cmps * avg_t)
                        recall.append(r)
                        qps.append(q)
                        line = f.readline()
                x = np.array(recall)
                y = np.array(qps)

                # 创建平滑插值对象
                spline = UnivariateSpline(x, y, s=1)  # s参数控制平滑程度，值越大平滑程度越高
                # 需要插值的新数据点
                x_new = np.linspace(x.min(), x.max(), 256)
                # 计算平滑插值后的新数据点
                y_new = spline(x_new)

                if x.max() > 0.95:
                    vs_res_dict[file[:-4]] = spline(np.array([0.95]))[0] 

                # 仅看最大的20%
                ids = np.where(x_new > (max_recall-0.05))[0]
                x_new = x_new[ids]
                y_new = y_new[ids]

                print(file[:-4], np.max(y_new), np.min(x_new))
                print(" ".join(["(%.6f, %d)" % (x, int(y)) for x, y in zip(x_new, y_new)]))

                for x, y in zip(x_new, y_new):
                    data["R@10"].append(x)
                    data["qps"].append(y)
                    data["name"].append(file[:-4])

        # if "GID" in vs_res_dict and "vamana" in vs_res_dict:
            # print(DATASET_NAME, "GID vs vamana:", vs_res_dict["GID"] / vs_res_dict["vamana"])

        # if "GID_pe" in vs_res_dict and "vamana_low" in vs_res_dict:
            # print(DATASET_NAME, "GID_PE vs vamana:", vs_res_dict["GID_pe"] / vs_res_dict["vamana_low"])

        # if "vamana_pe" in vs_res_dict and "vamana_low" in vs_res_dict:
            # print(DATASET_NAME, "vamana_pe vs vamana:", vs_res_dict["vamana_pe"] / vs_res_dict["vamana_low"])

        df = pd.DataFrame(data)
        sns.set_style(style='white')
        ax = sns.lineplot(x="R@10", y="qps", hue="name", style="name", markers=True, dashes=False, data=df)
        #id += 1
        plt.show()



if __name__ == '__main__':
    main()
