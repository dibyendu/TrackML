import networkx as nx
import numpy as np
from scipy import spatial
import copy, math, sys

def to_nx_graph(data, model=1):
    ''' If a particular point is hit by both the true particle and the noisy particle,
    we mark the point as a noisy point only if the number of layers got hit by the noisy particle
    is sterictly greater than that by the true particle
    
    =========== node_attrs =============
    (
        (0, {'layers': array([[1., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 0.], [0., 0.]]), 'noise': False, 'pos': array([0.4359949 , 0.02592623])}),
        (1, {'layers': array([[0., 0.], [0., 0.], [0., 1.], [0., 1.], [0., 0.], [0., 0.]]), 'noise': True, 'pos': array([0.54966248, 0.43532239])}),
        (2, {'layers': array([[0., 0.], [0., 0.], [0., 0.], [1., 0.], [0., 0.], [0., 0.]]), 'noise': False, 'pos': array([0.4203678 , 0.33033482])}),
        (3, {'layers': array([[0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 0.], [0., 0.]]), 'noise': False, 'pos': array([0.20464863, 0.61927097])})
    )
    =========== edge_attrs =============
    (
        (0, 0, {'distance': 0.0, 'noise': False}),
        (0, 1, {'distance': 0.42488296534900866, 'noise': False}),
        (0, 2, {'distance': 0.3048094411599255, 'noise': False}),
        ...
        (3, 3, {'distance': 0.0, 'noise': False})
    )
    '''

    dimension = data.hL[0].shape[0]

    pos = {}
    for layer, matrix in enumerate(data.hL):
        x, y = matrix.nonzero()
        for c in zip(x, y):
            if c not in pos:
                pos[c] = {
                    'layers': np.zeros((6, 2), dtype=float),
                    'noise': True
                }
            pos[c]['layers'][layer] = np.array([0.0, 1.0])
    for layer, matrix in enumerate(data.gthL):
        x, y = matrix.nonzero()
        for c in zip(x, y):
            if c not in pos:
                raise Exception(
                    'Ground Truth Hit Layer has extra point that doesn\'t exist in Hit Layer'
                )
            pos[c]['layers'][layer] = np.array([1.0, 0.0])
    for val in pos.values():
        val['noise'] = (np.argmax(np.sum(val['layers'], axis=0)) != 0)
        if not val['noise']:
            val['layers'][:, -1] = 0.0
    pos_tuple = tuple(pos.keys())
    node_attrs = tuple([(i, {
        'pos': np.array([x, y], dtype=float) / dimension,
        'layers': pos[(x, y)]['layers'],
        'noise': pos[(x, y)]['noise']
    }) for i, (x, y) in enumerate(pos_tuple)])
    if model == 1:
        distances = spatial.distance.squareform(spatial.distance.pdist(pos_tuple)) / dimension
        i_, j_ = np.meshgrid(range(len(pos_tuple)), range(len(pos_tuple)), indexing='ij')
        attrs = [{
            'distance': d,
            'noise': pos[pos_tuple[u]]['noise']
        } for (u, d) in zip(i_.ravel(), distances.ravel())]
        edge_attrs = tuple(zip(i_.ravel(), j_.ravel(), attrs))
    elif model == 2:
        distances = spatial.distance.squareform(spatial.distance.pdist(pos_tuple)) / dimension
        i_, j_ = np.meshgrid(range(len(pos_tuple)), range(len(pos_tuple)), indexing='ij')
        attrs = [{
            'distance': d,
            'noise': pos[pos_tuple[u]]['noise'] != pos[pos_tuple[v]]['noise']
        } for (u, v, d) in zip(i_.ravel(), j_.ravel(), distances.ravel())]
        edge_attrs = tuple(zip(i_.ravel(), j_.ravel(), attrs))
    elif model == 3:
        attrs = [{
            'distance': 0.0,
            'noise': pos[p]['noise']
        } for p in pos_tuple]
        edge_attrs = tuple(zip(range(len(pos_tuple)), range(len(pos_tuple)), attrs))

    graph = nx.DiGraph()
    graph.add_nodes_from(node_attrs)
    graph.add_edges_from(edge_attrs)

    return graph



def to_graph_dict_without_edges(data):
    ''' If a particular point is hit by both the true particle and the noisy particle,
    we mark the point as a noisy point only if the number of layers got hit by the noisy particle
    is sterictly greater than that by the true particle
    
    =========== input_graph_dict ===============
    {
        'globals': array([0.]),
        'nodes': array([
            [4.17022005e-01, 7.20324493e-01, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.14374817e-04, 3.02332573e-01, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.46755891e-01, 9.23385948e-02, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.86260211e-01, 3.45560727e-01, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]),
        'senders': array([], dtype=float64),
        'receivers': array([], dtype=float64),
        'edges': array([], shape=(1, 0), dtype=float64)
    }
    =========== target_graph_dict ===============
    {
        'globals': array([0.25]),
        'nodes': array([
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [1., 0.]
        ]),
        'senders': array([], dtype=float64),
        'receivers': array([], dtype=float64),
        'edges': array([], shape=(1, 0), dtype=float64)
    }
    '''

    dimension = 1024

    pos = {}
    for layer, pList in enumerate(data['hL']):
        for x, y in pList:
            if (x, y) not in pos:
                pos[(x,y)] = {
                    'layers': np.zeros((6, 2), dtype=float),
                    'noise': True
                }
            pos[(x,y)]['layers'][layer] = np.array([0.0, 1.0])
    for layer, pList in enumerate(data['gthL']):
        for x, y in pList:
            if (x, y) not in pos:
                raise Exception(
                    'Ground Truth Hit Layer has extra point that doesn\'t exist in Hit Layer'
                )
            pos[(x, y)]['layers'][layer] = np.array([1.0, 0.0])
    for val in pos.values():
        val['noise'] = (np.argmax(np.sum(val['layers'], axis=0)) != 0)
        if not val['noise']:
            val['layers'][:, -1] = 0.0
    pos_tuple = tuple(pos.keys())
    layers = np.array([pos[c]['layers'] for c in pos_tuple])
    noise = np.array([pos[c]['noise'] for c in pos_tuple])

    input_graph_node_features = np.concatenate(
        (np.array(pos_tuple).astype(float) / dimension,
         np.reshape(layers, (len(pos_tuple), 12))),
        axis=1)
    target_graph_node_features = np.eye(2)[noise.astype(int)]

    input_graph_dict = {
        'globals': np.array([0.0]),
        'nodes': input_graph_node_features,
        'edges': np.array([[]]),
        'senders': np.array([]),
        'receivers': np.array([])
    }
    target_graph_dict = {
        'globals': np.array([np.sum(noise).astype(float) / len(pos_tuple)]),
        'nodes': target_graph_node_features,
        'edges': np.array([[]]),
        'senders': np.array([]),
        'receivers': np.array([])
    }

    return input_graph_dict, target_graph_dict


def to_graph_dict_with_edges(data):

    epsilon, EPSILON = 0.0, sys.float_info.max
    dimension = 1024

    clustered_tracks = []
    for hits in data['tracks'].values():
        layers = [{'count': 0, 'x': 0, 'y': 0}, {'count': 0, 'x': 0, 'y': 0}, {'count': 0, 'x': 0, 'y': 0},
                  {'count': 0, 'x': 0, 'y': 0}, {'count': 0, 'x': 0, 'y': 0}, {'count': 0, 'x': 0, 'y': 0}]
        track = []
        for (layer, x, y) in hits:
            layers[layer]['count'] += 1
            layers[layer]['x'] += x
            layers[layer]['y'] += y
        for layer in layers:
            track.append((1.0 * layer['x'] / layer['count'], 1.0 * layer['y'] / layer['count']) if layer['count'] else None)
        clustered_tracks.append(track)

    node_dict = {}
    for track in clustered_tracks:
        for layer, node in enumerate(track):
            if node is not None:
                if node not in node_dict:
                    node_dict[node] = {'layer': -1, 'edges': []}
                node_dict[node]['layer'] = layer

    layers = np.array(clustered_tracks).T

    distances = []
    for layer_i, nodes_i in enumerate(layers):
        for layer_j, nodes_j in enumerate(layers):
            if layer_i != layer_j:
                for track_i, node_i in enumerate(nodes_i):
                    if node_i is not None:
                        for track_j, node_j in enumerate(nodes_j):
                            if node_j is not None:
                                u, v = node_i, node_j
                                distance = math.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)
                                distances.append(distance)
                                if distance >= epsilon and distance <= EPSILON:
                                    if track_i == track_j:
                                        min_layer, max_layer = min(layer_i, layer_j), max(layer_i, layer_j)
                                        partial_track = layers[min_layer + 1:max_layer, track_i]
                                    node_dict[u]['edges'].append({
                                        'node': v,
                                        'layer': layer_j,
                                        'distance': distance,
                                        'is_track_edge': (track_i == track_j) and not np.any(partial_track.astype(bool))
                                    })

    pos_array = sorted(node_dict.keys())
    
    input_graph_node_features, input_graph_edge_features, senders, receivers = [], [], [], []
    target_graph_edge_features = []
    
    for u in sorted(node_dict.keys()):
        src_one_hot_layer = [0.0] * 6
        src_one_hot_layer[node_dict[u]['layer']] = 1.0
        input_graph_node_features.append([u[0] * 1.0  / dimension, u[1] * 1.0  / dimension] + src_one_hot_layer)
        for v in node_dict[u]['edges']:
            dest_one_hot_layer = [0.0] * 6
            dest_one_hot_layer[v['layer']] = 1.0
            input_graph_edge_features.append([v['distance'] / dimension, u[0] * 1.0  / dimension, u[1] * 1.0  / dimension, v['node'][0] * 1.0  / dimension, v['node'][1] * 1.0  / dimension] + [node_dict[u]['layer'], v['layer']])
            target_graph_edge_features.append([0.0, 1.0] if v['is_track_edge'] else [1.0, 0.0])
            senders.append(pos_array.index(u))
            receivers.append(pos_array.index(v['node']))

    input_graph_dict = {
        'globals': np.array([0.0], dtype=float),
        'nodes': np.array(input_graph_node_features, dtype=float),
        'edges': np.array(input_graph_edge_features, dtype=float),
        'senders': np.array(senders, dtype=int),
        'receivers': np.array(receivers, dtype=int)
    }
    target_graph_dict = {
        'globals': np.array([len(data['tracks'])], dtype=float),
        'nodes': np.array([]),
        'edges': np.array(target_graph_edge_features, dtype=float),
        'senders': np.array(senders, dtype=int),
        'receivers': np.array(receivers, dtype=int)
    }

    return input_graph_dict, target_graph_dict, node_dict, min(distances) if distances else -1, max(distances) if distances else -1