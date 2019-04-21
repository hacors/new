import matplotlib.pyplot as plt


def scatter(x_digis, y_digis, color='steelblue'):
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x_digis, y_digis, s=20, c=color, marker='s', alpha=0.9, linewidths=0.3, edgecolors='red')
    plt.xlabel('tops')
    plt.ylabel('cmcs')
    plt.tick_params(top='off', right='off')
    plt.show()


def bars(datas, labels):
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.bar(range(len(datas)), datas, width=0.5, color='steelblue', tick_label=labels)
    plt.show()


if __name__ == '__main__':
    scatter([1, 2, 3], [4.12345234, 5, 6])
    bars([0.7357594936708861, 0.7199367088607594, 0.7389240506329114, 0.7531645569620253], ['head', 'body', 'leg', 'sum'])
