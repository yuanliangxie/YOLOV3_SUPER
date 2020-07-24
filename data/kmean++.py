# coding=utf-8
# k-means ++ for YOLOv2 anchors
# 通过k-means ++ 算法获取YOLOv2需要的anchors的尺寸
import numpy as np
import os
from tqdm import tqdm
#TODO:运行效率太慢，通过查看cpu调用资源，发现只有一个核在100%跑，其余则在空线
# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes, n_anchors):
    print("初始化中心点！")
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)
    centroids.append(boxes[centroid_index[0]])

    #print(centroids[0].w, centroids[0].h)

    for centroid_index in tqdm(range(0, n_anchors-1)):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0
        #找距离
        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance#这里是只加上距离中心点最近的distance，如果连最近都很远的话就说明真的很远啦
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()
        #寻点
        for i in range(0, boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                #print(boxes[i].w, boxes[i].h)
                break

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in tqdm(boxes):
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


#画出matplotlib演化过程中的动态图
import matplotlib.pyplot as plt


def plot_dynamic_picture(centroids, groups, loss, image_shape, n_clusters):
    """

    :param centroids:
    :param groups:
    :param loss:
    :param image_shape: tuple:(h, w)
    :param n_clusters:
    :return:
    """
    image_h, image_w = image_shape
    plt.axis([0, image_w, 0, image_h])
    colors = ['aqua', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90']
    for i, group_boxes in enumerate(groups):
        for box in group_boxes:
            w = box.w * image_w
            h = box.h * image_h
            plt.scatter(w, h, color=colors[i], alpha=0.6)
    for i in tqdm(range(n_clusters)):
        centroid = centroids[i]
        w = centroid.w * image_w
        h = centroid.h * image_h
        # plt.text(w, h, str((centroid.w, centroid.h)), color=colors[i], \
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(w, h, marker='p', color='orange', linewidths=8)
    #plt.title("loss={:.2f}".format(loss))
    plt.title("k-means clustering for anchors")
    plt.xlabel('width')
    plt.ylabel('height')
    plt.show()


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(label_path, n_anchors, loss_convergence, grid_size, iterations_num, plus):
    """

    :param label_path: label_path应该是一个文件夹，里面装着文本文件，文件的每行的格式是：class_label, x_center, y_center, w , h
    :param n_anchors: 聚类几个anchor
    :param loss_convergence: 当聚类距离小于这个值时，循环跳出，得到结果
    :param grid_size: 如果label_path中加入的时绝对坐标，那么grid_size设置为1，如果加入的是相对坐标（即0-1，归一化），那么grid_size设置为所在图像的大小, format:(h,w)
    :param iterations_num: 限制迭代次数
    :param plus: 时候选择使用kmean+算法
    :return:
    """

    boxes = []
    for label_file in tqdm(os.listdir(label_path)):
        f = open(os.path.join(label_path, label_file))
        for line in f.readlines():
            temp = line.strip().split(" ")
            if len(temp) > 1:
                boxes.append(Box(0, 0, float(temp[3]), float(temp[4])))

    if plus:
        centroids = init_centroids(boxes, n_anchors)
        print('寻种子点完毕！')
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    #plot_dynamic_picture(centroids, groups, old_loss)
    iterations = 1
    while (True):
        print('iterations:%d'%iterations)
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        #plot_dynamic_picture(centroids, groups, loss)
        iterations = iterations + 1
        print("loss = %f" % loss)
        #if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
        if abs(old_loss - loss) < loss_convergence:
            break
        old_loss = loss

        # for centroid in centroids:
        #     print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    for centroid in centroids:
        print("k-means result：\n")
        print(centroid.w * grid_size[1], centroid.h * grid_size[0])

    #plot the finally_picture
    print("开始进行画图！")
    plot_dynamic_picture(centroids, groups, loss, image_shape=(306, 544), n_clusters=n_anchors)

if __name__ == '__main__':
    label_path = "./detrac/labels"
    n_anchors = 9
    loss_convergence = 1e-3
    featureMap_size = 1
    iterations_num = 100
    plus = 1
    compute_centroids(label_path, n_anchors, loss_convergence, (306, 544), iterations_num, plus)
