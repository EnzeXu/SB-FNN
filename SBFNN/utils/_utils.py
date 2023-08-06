from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time


def add_time(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "myprint":
            with open(args[1], "a") as f:
                f.write("{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
        print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))), end="")
        ret = func(*args, **kwargs)
        return ret
    return wrapper


@add_time
def myprint(string, filename):
    with open(filename, "a") as f:
        f.write("{}\n".format(string))
    print(string)


def draw_two_dimension(
    y_lists,
    x_list,
    color_list,
    line_style_list,
    legend_list=None,
    legend_location="auto",
    legend_bbox_to_anchor=(0.515, 1.11),
    legend_ncol=3,
    legend_fontsize=15,
    fig_title=None,
    legend_loc="best",
    fig_x_label="time",
    fig_y_label="val",
    show_flag=True,
    save_flag=False,
    save_path=None,
    save_dpi=300,
    fig_title_size=20,
    fig_grid=False,
    marker_size=0,
    line_width=2,
    x_label_size=15,
    y_label_size=15,
    number_label_size=15,
    fig_size=(8, 6),
    x_ticks_set_flag=False,
    x_ticks=None,
    y_ticks_set_flag=False,
    y_ticks=None,
    tight_layout_flag=True,
    x_ticks_dress=None,
    y_ticks_dress=None
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :param x_ticks: (list) list of x_ticks. e.g., range(2, 21, 1)
    :param x_ticks_set_flag: (boolean) whether to set x_ticks. e.g., False
    :param y_ticks: (list) list of y_ticks. e.g., range(2, 21, 1)
    :param y_ticks_set_flag: (boolean) whether to set y_ticks. e.g., False
    :return:
    """
    assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    if x_ticks_set_flag:
        if x_ticks_dress:
            plt.xticks(x_ticks, x_ticks_dress)
        else:
            plt.xticks(x_ticks)
    if y_ticks_set_flag:
        if y_ticks_dress:
            plt.xticks(y_ticks, y_ticks_dress)
        else:
            plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True, ncol=legend_ncol, loc=legend_loc)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize, loc=legend_loc)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def draw_two_dimension_different_x(
    y_lists,
    x_lists,
    color_list,
    line_style_list,
    legend_list=None,
    legend_location="auto",
    legend_bbox_to_anchor=(0.515, 1.11),
    legend_ncol=3,
    legend_fontsize=15,
    fig_title=None,
    fig_x_label="time",
    fig_y_label="val",
    show_flag=True,
    save_flag=False,
    save_path=None,
    save_dpi=300,
    fig_title_size=20,
    fig_grid=False,
    marker_size=0,
    line_width=2,
    x_label_size=15,
    y_label_size=15,
    number_label_size=15,
    fig_size=(8, 6),
    x_ticks_set_flag=False,
    x_ticks=None,
    y_ticks_set_flag=False,
    y_ticks=None,
    tight_layout_flag=True
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_lists: (list[list]) x value of lines. e.g., [[1,2,3,4], [5,6,7]]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :param x_ticks: (list) list of x_ticks. e.g., range(2, 21, 1)
    :param x_ticks_set_flag: (boolean) whether to set x_ticks. e.g., False
    :param y_ticks: (list) list of y_ticks. e.g., range(2, 21, 1)
    :param y_ticks_set_flag: (boolean) whether to set y_ticks. e.g., False
    :return:
    """
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    for i in range(y_count):
        assert len(y_lists[i]) == len(x_lists[i]), "Dimension of y should be same to that of x"
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_lists[i], y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    if x_ticks_set_flag:
        plt.xticks(x_ticks)
    if y_ticks_set_flag:
        plt.yticks(y_ticks)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        if legend_location == "fixed":
            plt.legend(legend_list, fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True, ncol=legend_ncol)
        else:
            plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def smooth_conv(data, kernel_size: int = 10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')


def draw_multiple_loss(
        loss_path_list,
        color_list,
        line_style_list,
        legend_list,
        fig_title,
        start_index,
        end_index,
        threshold=None,
        smooth_kernel_size=1,
        marker_size=0,
        line_width=1,
        fig_size=(8, 6),
        x_ticks_set_flag=False,
        x_ticks=None,
        y_ticks_set_flag=False,
        y_ticks=None,
        show_flag=True,
        save_flag=False,
        save_path=None,
        only_original_flag=False,
        fig_x_label="epoch",
        fig_y_label="loss",
        legend_location="auto",
        legend_bbox_to_anchor=(0.515, 1.11),
        legend_ncol=3,
        tight_layout_flag=True
):
    # line_n = len(loss_path_list)
    assert (len(loss_path_list) if only_original_flag else 2 * len(loss_path_list)) == len(color_list) == len(line_style_list) == len(legend_list), "Note that for each loss in loss_path_list, this function will generate an original version and a smoothed version. So please give the color_list, line_style_list, legend_list for all of them"
    x_list = range(start_index, end_index)
    y_lists = [np.load(one_path) for one_path in loss_path_list]
    print("length:", [len(item) for item in y_lists])
    y_lists_smooth = [smooth_conv(item, smooth_kernel_size) for item in y_lists]
    for i, item in enumerate(y_lists):
        print("{}: {}".format(legend_list[i], np.mean(item[start_index: end_index])))
        if threshold:
            match_index_list = np.where(y_lists_smooth[i] <= threshold)
            if len(match_index_list[0]) == 0:
                print("No index of epoch matches condition '< {}'!".format(threshold))
            else:
                print("Epoch {} is the first value matches condition '< {}'!".format(match_index_list[0][0], threshold))
    y_lists = [item[x_list] for item in y_lists]
    y_lists_smooth = [item[x_list] for item in y_lists_smooth]
    draw_two_dimension(
        y_lists=y_lists if only_original_flag else y_lists + y_lists_smooth,
        x_list=x_list,
        color_list=color_list,
        legend_list=legend_list,
        legend_location=legend_location,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_ncol=legend_ncol,
        line_style_list=line_style_list,
        fig_title=fig_title,
        fig_size=fig_size,
        fig_x_label=fig_x_label,
        fig_y_label=fig_y_label,
        x_ticks_set_flag=x_ticks_set_flag,
        y_ticks_set_flag=y_ticks_set_flag,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        marker_size=marker_size,
        line_width=line_width,
        show_flag=show_flag,
        save_flag=save_flag,
        save_path=save_path,
        tight_layout_flag=tight_layout_flag
    )


class MultiSubplotDraw:
    def __init__(self, row, col, fig_size=(8, 6), show_flag=True, save_flag=False, save_path=None, save_dpi=300, tight_layout_flag=False):
        self.row = row
        self.col = col
        self.subplot_index = 0
        self.show_flag = show_flag
        self.save_flag = save_flag
        self.save_path = save_path
        self.save_dpi = save_dpi
        self.tight_layout_flag = tight_layout_flag
        self.fig = plt.figure(figsize=fig_size)

    def draw(self, ):
        if self.tight_layout_flag:
            plt.tight_layout()
        if self.save_flag:
            plt.savefig(self.save_path, dpi=self.save_dpi)
        if self.show_flag:
            plt.show()
        plt.clf()
        plt.close()

    def add_subplot(
            self,
            y_lists,
            x_list,
            color_list,
            line_style_list,
            legend_list=None,
            legend_location="auto",
            legend_bbox_to_anchor=(0.515, 1.11),
            legend_ncol=3,
            legend_fontsize=15,
            fig_title=None,
            fig_x_label="time",
            fig_y_label="val",
            fig_title_size=20,
            fig_grid=False,
            marker_size=0,
            line_width=2,
            x_label_size=15,
            y_label_size=15,
            number_label_size=15,
            x_ticks_set_flag=False,
            x_ticks=None,
            y_ticks_set_flag=False,
            y_ticks=None,
            scatter_period=0,
            scatter_marker=None,
            scatter_marker_size=None,
            scatter_marker_color=None
    ):
        # assert len(list(y_lists[0])) == len(list(x_list)), "Dimension of y should be same to that of x"
        assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed. Got {} / {} / {}".format(len(y_lists), len(line_style_list), len(color_list))
        y_count = len(y_lists)
        self.subplot_index += 1
        ax = self.fig.add_subplot(self.row, self.col, self.subplot_index)
        for i in range(y_count):
            draw_length = min(len(x_list), len(y_lists[i]))
            # print("x_list[:draw_length]", x_list[:draw_length])
            # print("y_lists[i][:draw_length]", y_lists[i][:draw_length])
            # print("color_list[i]", color_list[i])
            ax.plot(x_list[:draw_length], y_lists[i][:draw_length], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i], label=legend_list[i] if legend_list else None)
            if scatter_period > 0:
                scatter_x = [x_list[:draw_length][idx] for idx in range(len(x_list[:draw_length])) if idx % scatter_period == 0]
                scatter_y = [y_lists[i][:draw_length][idx] for idx in range(len(y_lists[i][:draw_length])) if idx % scatter_period == 0]
                print(scatter_x)
                print(scatter_y)
                ax.scatter(x=scatter_x, y=scatter_y, s=scatter_marker_size, c=scatter_marker_color, marker=scatter_marker, linewidths=0, zorder=10)
        ax.set_xlabel(fig_x_label, fontsize=x_label_size)
        ax.set_ylabel(fig_y_label, fontsize=y_label_size)
        if x_ticks_set_flag:
            ax.set_xticks(x_ticks)
        if y_ticks_set_flag:
            ax.set_yticks(y_ticks)
        if legend_list:
            if legend_location == "fixed":
                ax.legend(fontsize=legend_fontsize, bbox_to_anchor=legend_bbox_to_anchor, fancybox=True,
                           ncol=legend_ncol)
            else:
                ax.legend(fontsize=legend_fontsize)
        if fig_title:
            ax.set_title(fig_title, fontsize=fig_title_size)
        if fig_grid:
            ax.grid(True)
        plt.tick_params(labelsize=number_label_size)
        return ax

    def add_subplot_turing(
            self,
            matrix,
            v_max,
            v_min,
            fig_title=None,
            fig_title_size=20,
            number_label_size=15,
            colorbar=True,
            x_ticks_set_flag=False,
            y_ticks_set_flag=False,
            x_ticks=None,
            y_ticks=None,
            x_ticklabels=None,
            y_ticklabels=None,
            y_label_rotate=None,
            invert=True,
            y_label="Y",
            x_label="X",
    ):
        self.subplot_index += 1
        ax = self.fig.add_subplot(self.row, self.col, self.subplot_index)
        im1 = ax.imshow(matrix, cmap=plt.cm.jet, vmax=v_max, vmin=v_min, aspect='auto')
        if x_ticks_set_flag:
            ax.set_xticks(x_ticks)
            if x_ticklabels is not None:
                ax.set_xticklabels(x_ticklabels)
        if y_ticks_set_flag:
            ax.set_yticks(y_ticks)
            if y_ticklabels is not None:
                ax.set_yticklabels(y_ticklabels)
        ax.set_title(fig_title, fontsize=fig_title_size)
        ax.set_xlabel(x_label, fontsize=number_label_size)
        if y_label_rotate is not None:
            print("y label has been rotated: {}".format(y_label_rotate))
            ax.set_ylabel(y_label, fontsize=number_label_size, rotation=y_label_rotate)
        else:
            ax.set_ylabel(y_label, fontsize=number_label_size)
        if invert:
            ax.invert_yaxis()
        if colorbar:
            plt.colorbar(im1, shrink=1)
        plt.tick_params(labelsize=number_label_size)
        return ax


"""
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(X, Y, Z)
plt.draw()
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(np.asarray([[x, y, z] for x, y, z in zip(X, Y, Z)]))
plt.show()
plt.clf()"""


def draw_three_dimension(
        lists,
        color_list,
        line_style_list,
        legend_list=None,
        fig_title=None,
        fig_x_label="X",
        fig_y_label="Y",
        fig_z_label="Z",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        lim_adaptive_flag=False,
        x_lim=(-25, 25),
        y_lim=(-25, 25),
        z_lim=(0, 50),
        line_width=1,
        alpha=1,
        x_label_size=15,
        y_label_size=15,
        z_label_size=15,
        number_label_size=15,
        fig_size=(8, 6),
        tight_layout_flag=True,
) -> None:
    for one_list in lists:
        assert len(one_list) == 3, "3D data please!"
        assert len(one_list[0]) == len(one_list[1]) == len(one_list[2]), "Dimension of X, Y, Z should be the same"
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    for i, one_list in enumerate(lists):
        ax.plot(one_list[0], one_list[1], one_list[2], linewidth=line_width, alpha=alpha, c=color_list[i], linestyle=line_style_list[i], label=legend_list[i] if legend_list else None)
    ax.legend(loc="lower left")
    ax.set_xlabel(fig_x_label, fontsize=x_label_size)
    ax.set_ylabel(fig_y_label, fontsize=y_label_size)
    ax.set_zlabel(fig_z_label, fontsize=z_label_size)
    if lim_adaptive_flag:
        x_lim = (min([min(item[0]) for item in lists]), max([max(item[0]) for item in lists]))
        y_lim = (min([min(item[1]) for item in lists]), max([max(item[1]) for item in lists]))
        z_lim = (min([min(item[2]) for item in lists]), max([max(item[2]) for item in lists]))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    if fig_title:
        ax.set_title(fig_title, fontsize=fig_title_size)
    plt.tick_params(labelsize=number_label_size)
    if tight_layout_flag:
        plt.tight_layout()
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class ColorCandidate:
    def __init__(self):
        self.raw_rgb = [
            (255, 0, 0),
            (0, 0, 255),
            (0, 128, 0),
            (255, 127, 0),
            (255, 0, 127),
            (0, 128, 127),
            (150, 10, 100),
            (150, 50, 20),
            (100, 75, 20),
            (127, 128, 0),
            (127, 0, 255),
            (0, 64, 255),
            (20, 75, 100),
            (20, 50, 150),
            (100, 10, 150),
        ]

    @staticmethod
    def lighter(color_pair, rate=0.5):
        return [int(color_pair[i] + (255 - color_pair[i]) * rate) for i in range(3)]

    def get_color_list(self, n, light_rate=0.5):
        assert n <= 15
        return [self.encode(item) for item in self.raw_rgb[:n]] + [self.encode(self.lighter(item, light_rate)) for item in self.raw_rgb[:n]]

    @staticmethod
    def decode(color_str):
        return [int("0x" + color_str[2 * i + 1: 2 * i + 3], 16) for i in range(3)]

    @staticmethod
    def encode(color_pair):
        return "#" + "".join([str(hex(item))[2:].zfill(2) for item in color_pair])


if __name__ == "__main__":
    pass
