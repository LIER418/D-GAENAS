import numpy as np
import random
import copy

node = 4
connect_selection = [
    [0, 1],
    [0, 1, 2],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4],
]
layer_type = [
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
type_number = len(layer_type)

def check_avail(net, node=5):
    is_avail = True
    base = 0
    for i in range(node):   
        if sum(net[base : base + (i+2) * type_number]) != 2:
            is_avail = False
        base += (i+2) * type_number
    return is_avail


def create_index_list(nums, shift=0):
    index_list = []
    for i in range(nums):
        index_list.append(i+shift)
    return index_list


def zero_supernet_generator(node, layer_type, is_int=False):
    vec_length = len(layer_type)
    masked_vec = np.zeros((1, vec_length))[0].tolist()
    disconnected_vec = np.zeros((1, vec_length))[0].tolist()
    supernet = [[] for v in range(node)]
    for i in range(node):
        for j in range(node + 2):
            if j < i + 2:
                supernet[i].append(masked_vec.copy())
            else:
                supernet[i].append(disconnected_vec.copy())
    if is_int:
        for i in range(len(supernet)):
            for j in range(len(supernet[i])):
                for n in range(len(supernet[i][j])):
                    supernet[i][j][n] = int(supernet[i][j][n])
    return supernet


def sampled_nets_generator(based_net, nums=1000):
    nets_dict = {}
    nets_list = []
    for n in range(nums):
        normal = random_supernet_generator(base_supernet=based_net)
        reduce = random_supernet_generator(base_supernet=based_net)
        sample = flatten_to_1D_vector(normal, reduce)
        assert check_avail(sample, node)
        if str(sample) not in nets_dict:
            nets_dict[str(sample)] = 1
            nets_list.append(sample)
    return nets_list


def progressive_sampled_nets_generator(based_net, nums=1, path_num=1):
    nets_dict = {}
    nets_list = []
    for n in range(nums):
        normal = progressive_supernet_generator(base_supernet=based_net, path_num=path_num)
        reduce = progressive_supernet_generator(base_supernet=based_net, path_num=path_num)
        sample = flatten_to_1D_vector(normal, reduce)
        if str(sample) not in nets_dict:
            nets_dict[str(sample)] = 1
            nets_list.append(sample)

    return nets_list


def progressive_supernet_generator(base_supernet, path_num):
    sample_net = copy.deepcopy(base_supernet)
    for i in range(0, len(base_supernet)):
        selected_connect = random.sample(connect_selection[i], 2)
        
        selected_operation = random.sample(range(0, type_number), k=path_num)
        for j in selected_operation:
            sample_net[i][selected_connect[0]][j] = 1.0
        
        selected_operation = random.sample(range(0, type_number), k=path_num)
        for j in selected_operation:
            sample_net[i][selected_connect[1]][j] = 1.0

    return sample_net


def flatten_to_1D_vector(normal, reduce):
    oneD_normal = []
    oneD_reduce =[]
    for i in range(len(normal)):
        for j in range(i+2):
            for n in range(len(normal[i][j])):
                oneD_normal.append(normal[i][j][n])
                oneD_reduce.append(reduce[i][j][n])
    assert len(oneD_normal) == (sum(create_index_list(len(normal), shift=2)) * type_number)
    assert len(oneD_reduce) == (sum(create_index_list(len(normal), shift=2)) * type_number)
    oneD_normal.extend(oneD_reduce)
    assert len(oneD_normal) == (sum(create_index_list(len(normal), shift=2)) * type_number * 2)

    return oneD_normal


def random_supernet_generator(base_supernet):
    index_list = []
    for i in range(len(base_supernet)):
        index_list.append(create_index_list((i+2) * type_number))

    sample_net = copy.deepcopy(base_supernet)
    for i in range(len(base_supernet)):
        selected_index = random.sample(index_list[i], 2)
        assert selected_index[0] < (i + 2) * type_number and selected_index[1] < (i + 2) * type_number
        sample_net[i][selected_index[0] // type_number][selected_index[0] % type_number] = 1.0
        sample_net[i][selected_index[1] // type_number][selected_index[1] % type_number] = 1.0

    return sample_net


def get_rand_vector(layer_type):
    vec_length = len(layer_type)
    masked_vec = []
    for i in range(0, vec_length):
        if random.random() > 0.5:
            masked_vec.append(1)
        else:
            masked_vec.append(0)
    return masked_vec


def mask_rand_generator():
    supernet = [[] for v in range(node)]
    for i in range(node):
        for j in range(node + 2):
            if j < i + 2:
                supernet[i].append(get_rand_vector(layer_type))
            else:
                supernet[i].append(0)
    return supernet


def supernet_generator(node, layer_type):
    vec_length = len(layer_type)
    masked_vec = np.ones((1, vec_length))[0].tolist()
    supernet = [[] for v in range(node)]
    for i in range(node):
        for j in range(node + 2):
            if j < i + 2:
                supernet[i].append(masked_vec.copy())
            else:
                supernet[i].append(0)
    return supernet


def mask_specific_value(supernet, node_id, input_id, operation_id):
    supernet[node_id][input_id][operation_id] = 0.0
    return supernet


def selected_specific_value(supernet, node_id, input_id, operation_id):
    for i in range(len(supernet[node_id][input_id])):
        if i != operation_id:
            supernet[node_id][input_id][i] = 0.0
    return supernet


def encoding_to_masks(encoding):
    encoding = np.array(encoding).reshape( -1, type_number )
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    supernet        = [supernet_normal, supernet_reduce]
    mask            = []
    counter         = 0
    for cell in supernet:
        mask_cell = []
        for row in cell:
            mask_row = []
            for col in row:
                if type(col) == type([]):
                    mask_row.append( encoding[counter].tolist() )
                    counter += 1
                else:
                    mask_row.append( 0 )
            mask_cell.append( mask_row )
        mask.append( mask_cell )
    
    normal_mask = mask[0]
    reduce_mask = mask[1]
    
    return normal_mask, reduce_mask


def supernet_mask():
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    return supernet_normal, supernet_reduce


def encode_supernet():
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    supernet        = [supernet_normal, supernet_reduce]
    
    layer_types_count = len( layer_type ) 
    count = 0
    assert type(supernet) == type([])
    for cell in supernet:
        for row in cell:
            for col in row:
                if type(col) == type([]):
                    count += layer_types_count 
    return np.ones( (count) ).tolist()


def define_search_space():
    search_space = encode_supernet()
    A = []
    b = []
    init_point = []
    param_pos = 0
    for i in range(0, len(search_space) ):
        tmp = np.zeros( len(search_space) )
        tmp[i] = 1
        A.append( np.copy(tmp) )
        b.append( 1.0000001 )
        A.append( -1*np.copy(tmp) )
        b.append( 0.0000001 )
    for i in range(0, len(search_space) ):
        if random.random() >= 0.5:
            init_point.append(0.0)
        else:
            init_point.append(1.0)
    return {"A":np.array(A), "b":np.array(b), "init_point":np.array(init_point) }


if __name__ == '__main__':
    supernet_normal = supernet_generator(node, layer_type)
    supernet_reduce = supernet_generator(node, layer_type)
    print('supernet_generator->supernet_normal: ', supernet_normal)
    print('supernet_generator->supernet_reduce: ', supernet_reduce)

    based_net = zero_supernet_generator(node, layer_type, is_int=True)
    print('zero_supernet_generator->based_net: ', based_net)
    net_list = sampled_nets_generator(based_net, nums=1)
    print('sampled_nets_generator: ', net_list)
    print('encoding_to_masks: ', encoding_to_masks(net_list[0]))
