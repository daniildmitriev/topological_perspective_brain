import itertools
import numpy as np
import matplotlib.pyplot as plt
import utils

def plot_confusion_matrices(diagrams, 
                            pairwise_distance, 
                            filenames_a, 
                            filenames_b, 
                            title, 
                            title_a, 
                            title_b, 
                            vmin=None, 
                            vmax=None):
    n = len(filenames_a)
    m = len(filenames_b)
    if n == 0 or m == 0:
        return
    lens_a = [f"{len(microglia_data[filename])}pts" for filename in filenames_a]
    lens_b = [f"{len(microglia_data[filename])}pts" for filename in filenames_b]
    plt.figure(figsize=(20, 5))
    plt.suptitle(title)
    plt.subplot(131)
    plt.imshow(build_confusion_matrix(diagrams, 
                                      pairwise_distance, 
                                      filenames_a, 
                                      filenames_a), vmin=0, vmax=200)
    plt.xticks(np.arange(n), lens_a)
    plt.yticks(np.arange(n), lens_a)
    plt.title(f"{title_a} vs. {title_a}")
    plt.subplot(133)
    plt.imshow(build_confusion_matrix(diagrams, 
                                      pairwise_distance,
                                      filenames_a, 
                                      filenames_b), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks(np.arange(m), lens_b)
    plt.yticks(np.arange(n), lens_a)
    
    plt.title(f"{title_a} vs. {title_b}")
    plt.subplot(132)
    plt.imshow(build_confusion_matrix(diagrams, 
                                      pairwise_distance,
                                      filenames_b, 
                                      filenames_b), vmin=0, vmax=200)
    plt.colorbar()
    plt.xticks(np.arange(m), lens_b)
    plt.yticks(np.arange(m), lens_b)
    plt.title(f"{title_b} vs. {title_b}")
    plt.show()
    
    
def plot_confusion_matrices_distinct(diagrams, 
                                     pairwise_distance,
                                     filenames_a, 
                                     filenames_b, 
                                     title_a, 
                                     title_b, 
                                     index=None,
                                     max_index=None,
                                     vmin=None, 
                                     vmax=None):
    n = len(filenames_a)
    m = len(filenames_b)
    if n == 0 or m == 0:
        return
    lens_a = [f"{len(microglia_data[filename])}pts" for filename in filenames_a]
    lens_b = [f"{len(microglia_data[filename])}pts" for filename in filenames_b]
    plt.subplot(1, max_index, index + 1)
    confusion_matrix = build_confusion_matrix(diagrams, 
                                              pairwise_distance, 
                                              filenames_a, 
                                              filenames_b)
    if np.min(confusion_matrix) == vmin:
        x_min, y_min = np.unravel_index(confusion_matrix.argmin(), confusion_matrix.shape)
        print(f"Min distance achieved between {filenames_a[x_min]} and {filenames_b[y_min]}")
        print(f"Number of points: {lens_a[x_min]} and {lens_b[y_min]}")
    plt.imshow(confusion_matrix, vmin=vmin, vmax=vmax)
    if index + 1 == max_index:
        plt.colorbar()
    plt.xticks(np.arange(m), lens_b)
    plt.yticks(np.arange(n), lens_a)
    plt.yticks(rotation=90)
    plt.title(f"{title_a} vs. {title_b}")
    
def plot_comparison(diagrams, 
                    pairwise_distance, 
                    fixed_params_a=None, 
                    fixed_params_b=None, 
                    vary_params=None):
    if fixed_params_a is None or fixed_params_b is None or vary_params is None:
        return
    for fixed_param_a in fixed_params_a:
        for fixed_param_b in fixed_params_b:
            vmin, vmax, total_n = calc_min_max_total_dists(diagrams,
                                                           pairwise_distance,
                                                           fixed_param_a=fixed_param_a,
                                                           fixed_param_b=fixed_param_b,
                                                           vary_params=vary_params)
            
            cur_index = 0
            plt.figure(figsize=(20, 5))
            plt.suptitle(f"Fixed=({fixed_param_a}, {fixed_param_b})")
            for i, vary_param_a in enumerate(vary_params):
                filenames_a = find_files(diagrams.keys(), vary_param_a, fixed_param_a, fixed_param_b)
                for j, vary_param_b in enumerate(vary_params):
                    if i >= j:
                        continue
                    filenames_b = find_files(diagrams.keys(), vary_param_b, fixed_param_a, fixed_param_b)
                    if len(filenames_a) == 0 or len(filenames_b) == 0:
                        continue
                    plot_confusion_matrices_distinct(diagrams,
                                                     pairwise_distance,
                                            filenames_a, 
                                            filenames_b, 
                                            vary_param_a,
                                            vary_param_b,
                                            index=cur_index,
                                            max_index=total_n,
                                            vmin=vmin,
                                            vmax=min(2000, vmax),)    
                    cur_index += 1
            plt.show()
    
def plot_3d(file_a, file_b):

    x_a, y_a, z_a = ((microglia_data[file_a] - microglia_data[file_a].mean(axis=0)).T)
    x_b, y_b, z_b = ((microglia_data[file_b] - microglia_data[file_b].mean(axis=0)).T)

    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    trace_a = go.Scatter3d(
        x=x_a,  # <-- Put your data instead
        y=y_a,  # <-- Put your data instead
        z=z_a,  # <-- Put your data instead
        mode='markers',
        marker={
            'color': 'blue',
            'size': 10,
            'opacity': 0.8,
        },
        name=file_a
    )

    # Configure the trace.
    trace_b = go.Scatter3d(
        x=x_b,  # <-- Put your data instead
        y=y_b,  # <-- Put your data instead
        z=z_b,  # <-- Put your data instead
        mode='markers',
        marker={
            'color': 'green',
            'size': 10,
            'opacity': 0.8,
        },
        name=file_b
    )


    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    data = [trace_a, trace_b]

    plot_figure = go.Figure(data=data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)
    
def compare_barcodes(selected_ind, n=2, mode='closest', distances=None):
    """ 
    mode: 'closest' to compare with min distance, 
    'furthest' to compare with max distance
    """
    filename_a = list(diagrams.keys())[selected_ind]
    indexes_a = diagrams[filename_a][:, 2] == 1
    x_a, y_a = diagrams[filename_a][indexes_a][:, :2].T
    sorted_indexes = np.argsort(distances[selected_ind])
    print(sorted_indexes)
    if mode == 'closest':
        indexes_to_compare = sorted_indexes[1:n+1]
    else:
        indexes_to_compare = sorted_indexes[-n:]
    for i in indexes_to_compare:
        filename_b = list(diagrams.keys())[i]
        indexes_b = diagrams[filename_b][:, 2] == 1
        x_b, y_b = diagrams[filename_b][indexes_b][:, :2].T
        plt.figure(figsize=(15, 7))
        plt.suptitle(f"Distance = {distances[selected_ind, i]:.2f}")
        plt.subplot(121)
        plt.scatter(x_a, y_a)
        plt.xlim(20, 150)
        plt.ylim(20, 150)
        plt.subplot(122)
        plt.scatter(x_b, y_b)
        plt.xlim(20, 150)
        plt.ylim(20, 150)
        plt.show()
    if mode == 'closest':
        return sorted_indexes[1]
    else:
        return sorted_indexes[-1]

def plot_diagram(filename, subplot, title):
    plt.subplot(subplot)
    plt.title(title)
    indexes = diagrams[filename][:, 2] == 1
    x, y = diagrams[filename][indexes][:, :2].T
    plt.scatter(x, y)
    plt.xlim(20, 150)
    plt.ylim(20, 150)
    

def plot_grouped_conf_matrix(diagrams, pairwise_distance, *order, subplot=None, show_colorbar=False, ):
    def to_print(args):
        result = ""
        for word in args:
            if word in ["OPL", "IPL"]:
                result += word + "\n"
            elif word in ["Sex_u", "Sex_f", "Sex_m"]:
                if word == "Sex_u":
                    result += "Unknown\n"
                elif word == "Sex_f":
                    result += "Female\n"
                else:
                    result += "Male\n"
            elif '0' <= word[1:] <= '9':
                result += word[1:] + " days\n"
            else:
                result += "Adult\n"
#             else:
#                 result += word[1:] + "\n"
        return result
    conf_matrix = build_grouped_conf_matrix(diagrams, pairwise_distance, *order)
    labels = list(itertools.product(*order))
    labels_str = [to_print(element) for element in labels]
    
    if subplot is None:
        plt.figure(figsize=(10, 10))
    elif type(subplot) is int:
        plt.subplot(subplot)
    else:
        plt.subplot(*subplot)
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.xticks(np.arange(len(labels)), labels_str)
    plt.yticks(np.arange(len(labels)), labels_str)
    plt.imshow(conf_matrix)
    if show_colorbar:
        if subplot is None:
            plt.colorbar(fraction=0.046, pad=0.04)
        else:
            plt.colorbar(fraction=0.046, pad=0.04)