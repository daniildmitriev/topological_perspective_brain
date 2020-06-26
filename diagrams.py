import numpy as np
import tmd

def calculate_distances(entities, dist_func):
    distances = np.zeros((len(entities), len(entities)))
    for i, entity_a in enumerate(entities):
        for j, entity_b in enumerate(entities):
            try:
                distances[i, j] = dist_func(entity_a, entity_b)
            except KeyboardInterrupt:
                raise
            except:
                distances[i, j] = np.nan
    return distances

def eucl_dist_func(neuron_a, neuron_b):
    center_a = neuron_a.soma.get_center()
    center_b = neuron_b.soma.get_center()
    return np.linalg.norm(np.array(center_a) - np.array(center_b)) 

def image_diff_func(phs_a, phs_b):
    img_a = tmd.analysis.get_average_persistence_image(phs_a)
    img_b = tmd.analysis.get_average_persistence_image(phs_b)
    diff_img = tmd.analysis.get_image_diff_data(img_a, img_b)
    dist = np.sum(np.abs(diff_img))
    return dist

def ph_diff_func(ph_a, ph_b):
    return tmd.analysis.distance_persistence_image(ph_a, ph_b)

def calculate_diagram_distances(filename, data, pairwise_distance):
    diag_dist_func = lambda bar_a, bar_b: get_distance(pairwise_distance, 
                                                       bar_a, 
                                                       bar_b)
    return calculate_distances(data[filename], diag_dist_func)

def calculate_euclidian_distances(filename, data):
    return calculate_distances(data[filename], eucl_dist_func)

def calculate_image_distances(filename, data):
    return calculate_distances(data[filename], ph_diff_func)

def calculate_ph_distances(phs):
    return calculate_distances(phs, ph_diff_func)

def build_weighted_diagram(filename, 
                           pairwise_distance, 
                           calculate_func_a, 
                           calculate_func_b, 
                           data_a,
                           data_b,
                           eucl_coef):
    distances_a = calculate_func_a(filename, data_a)
    distances_b = calculate_func_b(filename, data_b)
    
    if np.isnan(distances_a).any() or np.isnan(distances_b).any():
        return None
    if isinstance(eucl_coef, list):
        return [persistenceDiagram.fit_transform([distances_a * coef + 
                                                  distances_b * (1 - coef)]) 
                for coef in eucl_coef] 
    return persistenceDiagram.fit_transform([distances_a * eucl_coef + 
                                             distances_b * (1 - eucl_coef)])

def build_total_weighted_diagram(pairwise_distance, 
                                 calculate_func_a, 
                                 calculate_func_b,
                                 data_a,
                                 data_b,
                                 eucl_coefs):
    res_diagrams = {}
    for eucl_coef in eucl_coefs:
        res_diagrams[eucl_coef] = {}
    for filename in data_a.keys():
        print(filename)
        cur_diagrams = build_weighted_diagram(filename, 
                                              pairwise_distance, 
                                              calculate_func_a, 
                                              calculate_func_b,
                                              data_a,
                                              data_b,
                                              eucl_coefs)
        if cur_diagrams is not None:
            for i, eucl_coef in enumerate(eucl_coefs):
                res_diagrams[eucl_coef][filename] = cur_diagrams[i]
        else:
            for i, eucl_coef in enumerate(eucl_coefs):
                res_diagrams[eucl_coef][filename] = None
    return res_diagrams

def build_total_weighted_diagram_diag_eucl(pairwise_distance, eucl_data, diag_data, eucl_coefs):
    return build_total_weighted_diagram(pairwise_distance, 
                                        calculate_euclidian_distances,
                                        (lambda filename: 
                                         calculate_diagram_distances(filename, 
                                                                     pairwise_distance=pairwise_distance)),
                                        eucl_data,
                                        diag_data,
                                        eucl_coefs)
def build_total_weighted_diagram_image_eucl(pairwise_distance, eucl_data, diag_data, eucl_coefs):
    return build_total_weighted_diagram(pairwise_distance,
                                        calculate_euclidian_distances,
                                        calculate_image_distances,
                                        eucl_data,
                                        diag_data,
                                        eucl_coefs)