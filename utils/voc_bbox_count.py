bbox_per_image_count = {1: 6821, 2: 3696, 3: 1964, 4: 1238, 5: 820, 6: 575, 7: 360, 8: 292, 9: 183, 10: 173, 11: 85,
                        12: 78, 13: 71, 14: 36, 15: 35, 16: 26, 17: 16, 18: 18, 19: 16, 20: 12, 21: 6, 22: 4, 23: 6,
                        24: 3, 25: 2, 26: 2, 27: 0, 28: 2, 29: 0, 30: 1, 31: 0, 32: 1, 33: 0, 34: 1, 35: 1, 36: 0,
                        37: 2, 38: 1, 39: 1, 40: 0, 41: 0, 42: 2, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0,
                        50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 1}

import matplotlib.pyplot as plt

x, y = bbox_per_image_count.keys(), bbox_per_image_count.values()
coord = zip(x,y)
plt.bar(x, y, align='center', color='steelblue', alpha=0.8)
plt.ylabel('数量')
plt.title('带有多少个box的图像统计')
for x1, y1 in coord:
    plt.text(x1, y1 + 100, '%s' % round(y1, 1), ha='center')
plt.show()

def sum_y(y):
    cum_y=0
    y_cum=[]
    for i in y:
        cum_y += i
        y_cum.append(cum_y)
    return y_cum


coord_sum = list(zip(x, sum_y(y)))
x, y = zip(*coord_sum)
plt.bar(x, y, align='center', color='steelblue', alpha=0.8)
plt.ylabel('叠加数量')
plt.title('带有多少个box的图像统计')
for x1, y1 in coord_sum:
    plt.text(x1, y1 + 100, '%s' % round(y1, 1), ha='center')
plt.show()



