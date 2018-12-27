import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw_heatmap(data, row_labels, column_labels):
    # 描画する
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.show()
    plt.savefig('Img/heatmaptest.png')

    return heatmap


mapdata =[[1,2,3,4],
          [2,3,4,3],
          [1,3,0,2],
          [4,0,0,1]
      ]
mapdata=np.array(mapdata)
print(mapdata)
rl = ["hoge","hoge","hoge","hoge"]
cl = "column"
draw_heatmap(mapdata, rl, cl)
