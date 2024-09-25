# tree detection V3
import numpy as np
from skimage.morphology import skeletonize_3d
from skimage.measure import label as label_regions

# seg_slice_label_I, 
# connection_dict_of_seg_I, 
# number_of_branch_I,
# tree_length_I 
# = tree_detection(seg_processed, search_range=2) #上面四个都是tree_detection的output

def tree_detection(seg_onehot, search_range = 2, need_skeletonize_3d=True):
    center_map, center_dict, nearby_dict = get_the_skeleton_and_center_nearby_dict(seg_onehot, search_range=search_range, need_skeletonize_3d=need_skeletonize_3d)
    # center_map代表的是3D骨架化后的中心图像，其中像素值为整数，每个整数代表一个中心点。
    # center_dict是一个字典，用于存储中心点的索引和对应的坐标。
    # nearby_dict是一个字典，用于存储每个中心点附近的邻居中心点的索引

    connection_dict = get_connection_dict(center_dict, nearby_dict)
#     该函数将返回一个名为connection_dict的字典，该字典包含以下信息：
        # loc：中心点的坐标。
        # before：中心点之前的索引列表。
        # next：中心点之后的索引列表。
        # is_bifurcation：表示中心点是否为分叉点。
        # number_of_next：中心点之后的索引数。
        # generation：中心点的世代。
        # is_processed：表示该中心点是否已被处理过。

    number_of_branch = get_number_of_branch(connection_dict)#这个函数用于计算支气管的分支数
    tree_length = get_tree_length(connection_dict, is_3d_len=True)
    
    return center_map, connection_dict, number_of_branch, tree_length

#center_map_crop = get_crop(center_map, center_dict[i], search_range)
#这里的 get_crop 函数是用于在三维图像中获取以指定中心坐标为中心的正方体区域，其大小由 search_range 决定。
# 函数接受三个参数，input_3d_img 表示输入的三维图像，search_center 表示正方体区域的中心坐标，search_range 表示正方体区域的大小。
def get_crop(input_3d_img, search_center, search_range=1):
    shape_of_input_3d_img = input_3d_img.shape
    
    x = search_center[0]
    y = search_center[1]
    z = search_center[2]
    
# np.clip函数是将输入的数值限制在指定范围内。
# 在这个函数中，它的作用是将搜索中心周围的坐标限制在输入图像的范围内。
# 具体来说，将中心坐标x、y、z分别加上和减去搜索范围search_range，
# 并且保证这些坐标在[0, 图像相应维度大小)的范围内。
# 如果中心坐标加减搜索范围后超出图像边界，np.clip函数会将其限制在边界上。

#np.clip 是用来对输入数组中的值进行限制操作，将小于或大于指定范围的值强制变为边界值。
# 其使用方法为 np.clip(a, a_min, a_max, out=None)
    x_s = np.clip(x-search_range, 0, shape_of_input_3d_img[0])
    x_e = np.clip(x+search_range+1, 0, shape_of_input_3d_img[0])
    y_s = np.clip(y-search_range, 0, shape_of_input_3d_img[1])
    y_e = np.clip(y+search_range+1, 0, shape_of_input_3d_img[1])
    z_s = np.clip(z-search_range, 0, shape_of_input_3d_img[2])
    z_e = np.clip(z+search_range+1, 0, shape_of_input_3d_img[2])
    # 对于某个坐标轴（x、y、z），搜索范围的下限是该轴上的中心点坐标减去search_range，
    # 上限是中心点坐标加上search_range再加1（加1是因为Python中的切片索引是左闭右开区间，即不包含右端点）
    #如果中心点坐标减去search_range小于0，那么下限就会被限制在0处；
    # 如果中心点坐标加上search_range大于输入图像在该轴上的大小，那么上限就会被限制在输入图像大小处
    #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
    crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    
    return crop_3d_img

# step 1 get the skeleton
#center_map, center_dict, nearby_dict = 
#get_the_skeleton_and_center_nearby_dict(seg_onehot, 
#                                        search_range=search_range,
#                                        need_skeletonize_3d=need_skeletonize_3d)
def get_the_skeleton_and_center_nearby_dict(seg_input, search_range = 10, need_skeletonize_3d=True):
    #传入参数need_skeletonize_3d=True
    #使用3D骨架化函数骨架化分割后的输入图像，并将骨架化图像转换为中心图
    if need_skeletonize_3d:
        center_map = np.array(skeletonize_3d(seg_input)>0, dtype=np.int32)
    else:
        center_map = seg_input
    
    center_dict = {}
    nearby_dict = {}
    #对于中心图中的每个中心，获取中心的坐标以及给定搜索范围内附近中心的坐标。
    # print(center_map.shape,seg_input.shape)#(224, 512, 512) (224, 512, 512)
    #骨架化得到的中心图center_map和分割图像seg_input都是二值化的三维张量，其元素值为0或1，其中1表示该像素属于支气管的中心线
    
    center_locs = np.where(center_map>0)
        # np.where(center_map>0)返回一个元组，元组中包含了三个数组，
        # 分别代表了中心图center_map中值大于0的像素的三维坐标。
        # 具体来说，假设中心图的形状为(D, H, W)，
        # 则center_locs[0]表示中心图中所有值大于0的像素在第一个维度（深度方向）上的坐标，
        # 即一个长度为n的一维数组；
        # center_locs[1]表示所有像素在第二个维度（高度方向）上的坐标，
        # 也是一个长度为n的一维数组；
        # center_locs[2]表示所有像素在第三个维度（宽度方向）上的坐标，
        # 同样也是一个长度为n的一维数组。
        # 其中，n是中心图中值大于0的像素数量，即支气管的中心线像素数量

    #这部分代码是为每个中心点在字典center_dict中分配一个唯一的标识符，并将该标识符添加到中心图center_map中。    
    base_count = 1
    for i in range(len(center_locs[0])):
        #将该中心点在字典center_dict中分配一个唯一的标识符i+base_count，
        # 该标识符由当前循环次数i和base_count相加得到。
        # 然后，将该中心点的坐标(center_locs[0][i], center_locs[1][i], center_locs[2][i])作为值，将标识符作为键，添加到字典center_dict中。
        center_dict[i+base_count]=[center_locs[0][i],center_locs[1][i],center_locs[2][i]]#center_locs[0]、[1]、[2]是中心图的三维坐标
        #将中心图center_map中该中心点的像素值赋为其唯一标识符，即i+base_count。
        # 这里的目的是方便后面的操作中使用字典center_dict来查找每个中心点的坐标
        center_map[center_locs[0][i],center_locs[1][i],center_locs[2][i]] = i+base_count
    
    for i in center_dict.keys():
        center_map_crop = get_crop(center_map, center_dict[i], search_range)
        #get_crop这个函数用于裁剪出一个patch，裁剪面积如下
        # print(search_range)#2
        # print(center_map.shape,center_map_crop.shape)#(224, 512, 512) (5, 5, 5)
        #裁剪得到的张量的形状是 (2*search_range+1, 2*search_range+1, 2*search_range+1)

        #center_map_crop是一个三维张量，当中心图某个位置(如(x,y,z))为1时，center_map_crop对应位置的值为索引坐标
        #np.unique函数会返回一个已排序的唯一值数组，该数组包含给定数组中所有不同值的唯一值
        #这个函数可以把0值去除，仅返回有效位置(中心点)上的索引坐标，并存储到整型数组crop_img_vals中
        crop_img_vals = np.unique(center_map_crop).astype(np.int32)
        # print(center_map_crop,'*',crop_img_vals)

        #crop_img_vals的类型是numpy.ndarray，它是一个一维数组，包含子区域中所有像素的唯一值。
        # 这里的目的是从这些唯一值中去除0和中心点的值i，然后将剩余的唯一值存储到nearby_dict字典中。其中nearby_dict的键是中心点的索引，值是一个列表，包含中心点附近的所有像素的唯一值。
        # 这样做的目的是为了后续在生成支气管树时能够根据当前节点的索引，方便地找到其相邻节点的索引。
        crop_img_vals = crop_img_vals[crop_img_vals!=0]
        crop_img_vals = crop_img_vals[crop_img_vals!=i]
        nearby_dict[i] = crop_img_vals
    #这段代码通过遍历每一个切片，获取切片的中心坐标，以中心坐标为中心截取一个小的正方形区域。
    # 然后通过获取该区域内的像素值，去重，去掉该切片的像素值以及0值，就可以获取该区域内所有的临近点。
    # 将这些临近点存储在一个字典 nearby_dict 中，键为当前切片的索引，值为一个包含该切片所有临近点的列表。
    # 这样就能方便地查找每个切片的临近切片了
    return center_map, center_dict, nearby_dict

# connection_dict = get_connection_dict(center_dict, nearby_dict)
# step 2 get_connection_dict
def get_connection_dict(center_dict, nearby_dict):
    #这两行代码是用来获取所有中心点的索引，并将其按照从后往前的顺序排序的。
    # 在进行骨架化时，我们按照从中心点到末梢的方向遍历支气管树。
    # 因此，我们需要将中心点按照从远到近的顺序排序，以便在遍历树时能够正确连接中心点。
    slice_idxs = list(center_dict.keys())
    slice_idxs.reverse()
    
    # init connection dict
    global connection_dict
    connection_dict = {}
    #首先在这里初始化connection_dict字典，后续字典还会被find_connection处理
        #  "loc": 表示当前切片的中心坐标，对应center_dict中的相应条目；
        # "before": 表示该切片前面的切片，即该切片的父节点，初始为空；
        # "next": 表示该切片后面的切片，即该切片的子节点，初始为空；
        # "is_bifurcation": 表示该切片是否为分支点，初始值为False；
        # "number_of_next": 表示该切片的子节点数量，初始值为0；
        # "generation": 表示该切片在树形结构中的代数，初始值为0；
        # "is_processed": 表示该切片是否已被处理，初始值为False。
    # 该字典中的键和值可以随时被修改，用于在后续的函数中存储和更新节点的相关信息。
    for slice_idx in slice_idxs:#初始化
        connection_dict[slice_idx] = {}
        connection_dict[slice_idx]["loc"] = center_dict[slice_idx]
        connection_dict[slice_idx]["before"] = []
        connection_dict[slice_idx]["next"] = []
        connection_dict[slice_idx]["is_bifurcation"] = False
        connection_dict[slice_idx]["number_of_next"] = 0
        connection_dict[slice_idx]["generation"] = 0
        connection_dict[slice_idx]["is_processed"] = False
        # print(connection_dict[slice_idx])
   

    #这个函数是 get_connection_dict 函数的内部函数，被用于构建分割后的支气管的连接字典。
    # 这个函数的作用是找到每个支气管分割切片的相邻切片，并将它们连接起来。
        #find_connection(current_label=slice_idxs[0], before_label=0)
        
    def find_connection(current_label, before_label):#函数的内部定义函数
        # print(slice_idxs)#[2926, 2925, 2924, 2923, 2922, 2921, 2920,.....
        global connection_dict
        
        nearby_labels = nearby_dict[current_label]#导出current_label对应的相邻点
        #相邻点是指，在我们裁剪时裁剪出一个crop，任意一个current_label肯定对应一个crop，这个crop中所有中心点都在nearby_dict[current_label]中
        valid_next_labels = []
        dist_to_valid_labels = []
        processed_count = 0
        for nearby_label in nearby_labels:
            if connection_dict[nearby_label]["is_processed"]==False:
                valid_next_labels.append(nearby_label)
                # 这段代码是在计算两个标签点之间的距离，距离的计算方法是两个标签点的位置之间的欧氏距离的平方，
                # 也就是说，先计算这两个标签点在每个坐标轴上的距离，然后将每个坐标轴上的距离的平方相加，得到最终的距离的平方。
                # 这个距离的平方会被存储在一个名为"dist_to_valid_labels"的列表中
                dist_to_valid_labels.append(np.sum(((np.array(connection_dict[nearby_label]["loc"])-np.array(connection_dict[current_label]["loc"]))**2)))
            else:
                processed_count+=1
        
        #这段代码是对 valid_next_labels 和 dist_to_valid_labels 进行排序，以便找出离当前标签（current_label）最近的标签
        if len(valid_next_labels)>0:
            valid_next_labels = np.array(valid_next_labels)
            dist_to_valid_labels = np.array(dist_to_valid_labels)
            sort_locs = np.argsort(dist_to_valid_labels)#sort_locs保存的索引，排好序的索引
            valid_next_labels = valid_next_labels[sort_locs]
        
        connection_dict[current_label]["before"].append(before_label)
        connection_dict[current_label]["is_processed"] = True

        print("current_label is "+str(current_label), end="\r")
        print(connection_dict[current_label], end="\r")
        #使用DFS深度优先递归处理所有next结点
        j=0
        if len(valid_next_labels)==0 or len(nearby_labels)==processed_count:
            print('num',j)
            return connection_dict
        else:
            for valid_next_label in valid_next_labels:
                if connection_dict[valid_next_label]["is_processed"]==False:
                    j+=1
                    connection_dict[current_label]["next"].append(valid_next_label)
                    find_connection(valid_next_label, current_label)
        print(f"find_connection: current_label={current_label}, before_label={before_label}")
        

    
    find_connection(current_label=slice_idxs[0], before_label=0)
    
    for item in connection_dict.keys():
        assert len(connection_dict[item]["before"])<=1, (item, connection_dict[item])
        connection_dict[item]["number_of_next"] = len(connection_dict[item]["next"])
        connection_dict[item]["is_bifurcation"] = (connection_dict[item]["number_of_next"]>1)
        #if connection_dict[item]["is_bifurcation"]:
        #    print(connection_dict[item])

    #这段代码定义了一个名为 find_generation() 的函数，函数有两个参数： current_label 和 generation。
    # 函数的作用是为每个切片（slice）标记一代（generation），即从主支气管开始计算，每向下延伸一代，代数加1。
    # 函数的实现方式是递归，首先将当前切片的代数设为传入的 generation 参数，并遍历当前切片的所有后代（next）。
    # 如果当前切片是分叉点，则其所有后代的代数都加1，否则后代的代数与当前切片相同。
    # 递归过程中，如果遇到一个没有后代的切片，则返回整个 connection_dict 字典。
    # 最后，该函数被调用，并将 slice_idxs[0] 作为 current_label，0作为generation参数，开始进行处理。    
    def find_generation(current_label, generation):
        global connection_dict
        connection_dict[current_label]["generation"]=generation
        if connection_dict[current_label]["number_of_next"]>0:
            for next_label in connection_dict[current_label]["next"]:
                if connection_dict[current_label]["is_bifurcation"]:
                    find_generation(next_label, generation+1)
                else:
                    find_generation(next_label, generation)
        else:
            return connection_dict
        print(f"find_generation: current_label={current_label}, generation={generation}")
    find_generation(current_label=slice_idxs[0], generation=0)
    
    return connection_dict
#------------------------------------def get_connection_dict(center_dict, nearby_dict)结束--------------------------------------

# number_of_branch = get_number_of_branch(connection_dict)
def get_number_of_branch(connection_dict):
    number_of_branch = 1
    for label in connection_dict.keys():
        if connection_dict[label]["is_bifurcation"]:
            number_of_branch+=connection_dict[label]["number_of_next"]
    return number_of_branch

#tree_length = get_tree_length(connection_dict, is_3d_len=True)
def get_tree_length(connection_dict, 
                    is_3d_len=True):#is_3d_len参数表示是否计算3D长度，
                                    # 如果为True，则计算节点间的欧几里得距离，
                                    # 如果为False，则仅计算节点间的数量差（即在二维平面上的直线距离）
    global tree_length
    tree_length = 0
    for label in connection_dict.keys():
        if connection_dict[label]["before"][0]==0:
            start_label = label#首先找到树的起始节点
            break

    #递归函数get_tree_length_func()的输入是当前节点的标签current_label
    # 函数通过遍历当前节点的子节点来递归计算子分支的长度，并将其累加到当前分支的长度中。最终返回总长度tree_length。
    def get_tree_length_func(connection_dict, current_label):
        global tree_length
        if connection_dict[current_label]["number_of_next"]==0:
            return
        else:
            current_branch_length = 0
            for next_label in connection_dict[current_label]["next"]:
                if is_3d_len:#计算欧几里得距离
                    current_branch_length += np.sqrt(np.sum((np.array(connection_dict[current_label]["loc"])-np.array(connection_dict[next_label]["loc"]))**2))
                else:
                    current_branch_length += 1
            print("len of "+str(current_label)+" branch is "+str(current_branch_length),end="\r")
            tree_length += current_branch_length
            for next_label in connection_dict[current_label]["next"]:
                get_tree_length_func(connection_dict, next_label)
    get_tree_length_func(connection_dict, start_label)
    return tree_length




# # tree detection V2
# import edt
# #from sklearn.cluster import DBSCAN
# import numpy as np
# from skimage.measure import label as label_regions 
# from skimage.feature import peak_local_max

# def tree_detection(seg_onehot, axis=0):
#     seg_slice_label, center_dict, touching_dict = label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=axis)
#     connection_dict = get_connection_dict(seg_slice_label, center_dict, touching_dict)
#     number_of_branch = get_number_of_branch(connection_dict)
#     tree_length = get_tree_length(connection_dict, is_3d_len=True)
    
#     return seg_slice_label, connection_dict, number_of_branch, tree_length

# def get_distance_transform(img_oneshot):
#     return edt.edt(np.array(img_oneshot>0, dtype=np.uint32, order='F'), black_border=True, order='F', parallel=1)

# """
# def get_outlayer_of_a_3d_shape(a_3d_shape_onehot):
#     shape=a_3d_shape_onehot.shape
    
#     a_3d_crop_diff_x1 = a_3d_shape_onehot[0:shape[0]-1,:,:]-a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_x2 = -a_3d_shape_onehot[0:shape[0]-1,:,:]+a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_y1 = a_3d_shape_onehot[:,0:shape[1]-1,:]-a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_y2 = -a_3d_shape_onehot[:,0:shape[1]-1,:]+a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_z1 = a_3d_shape_onehot[:,:,0:shape[2]-1]-a_3d_shape_onehot[:,:,1:shape[2]]
#     a_3d_crop_diff_z2 = -a_3d_shape_onehot[:,:,0:shape[2]-1]+a_3d_shape_onehot[:,:,1:shape[2]]

#     outlayer = np.zeros(shape)
#     outlayer[1:shape[0],:,:] += np.array(a_3d_crop_diff_x1==1, dtype=np.int8)
#     outlayer[0:shape[0]-1,:,:] += np.array(a_3d_crop_diff_x2==1, dtype=np.int8)
#     outlayer[:,1:shape[1],:] += np.array(a_3d_crop_diff_y1==1, dtype=np.int8)
#     outlayer[:,0:shape[1]-1,:] += np.array(a_3d_crop_diff_y2==1, dtype=np.int8)
#     outlayer[:,:,1:shape[2]] += np.array(a_3d_crop_diff_z1==1, dtype=np.int8)
#     outlayer[:,:,0:shape[2]-1] += np.array(a_3d_crop_diff_z2==1, dtype=np.int8)
    
#     outlayer = np.array(outlayer>0, dtype=np.int8)
    
#     return outlayer
#     """

# def get_crop_by_pixel_val(input_3d_img, val, boundary_extend=1, axis=0):
#     locs = np.where(input_3d_img==val)
    
#     shape_of_input_3d_img = input_3d_img.shape
    
#     min_x = np.min(locs[0])
#     max_x =np.max(locs[0])
#     min_y = np.min(locs[1])
#     max_y =np.max(locs[1])
#     min_z = np.min(locs[2])
#     max_z =np.max(locs[2])
    
#     if axis==0:
#         x_s = np.clip(min_x-boundary_extend, 0, shape_of_input_3d_img[0])
#         x_e = np.clip(max_x+boundary_extend+1, 0, shape_of_input_3d_img[0])
#         y_s = np.clip(min_y, 0, shape_of_input_3d_img[1])
#         y_e = np.clip(max_y+1, 0, shape_of_input_3d_img[1])
#         z_s = np.clip(min_z, 0, shape_of_input_3d_img[2])
#         z_e = np.clip(max_z+1, 0, shape_of_input_3d_img[2])
    
#     #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
#     crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    
#     return crop_3d_img

# """
# def find_cluster_by_dbscan(img_slice, base_count, threshold=0, dbscan_min_samples=1, dbscan_eps=1):
#     locs=np.where(img_slice>0)
#     locs_x=locs[0]
#     locs_y=locs[1]
#     locs_len=locs[0].shape[0]
#     locs_reshape=np.concatenate((locs[0].reshape(locs_len,1),
#                                  locs[1].reshape(locs_len,1)),axis=1)
    
#     clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean').fit(locs_reshape)
#     clustering_labels=clustering.labels_
#     clustering_labels_unique,clustering_labels_counts=np.unique(clustering_labels, return_counts=True)
    
#     #clustering_labels_counts=clustering_labels_counts[clustering_labels_unique>-1]
#     #clustering_labels_unique=clustering_labels_unique[clustering_labels_unique>-1] # delete noise
#     clustering_labels_unique=clustering_labels_unique+1
    
#     clustering_labels_unique=clustering_labels_unique[np.where(clustering_labels_counts>threshold)]
#     clustering_labels_counts=clustering_labels_counts[np.where(clustering_labels_counts>threshold)]
    
#     seg_img_slice=np.zeros(img_slice.shape)
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         temp_label=clustering_labels_unique[i]
#         temp_label_locs=np.where(clustering_labels==temp_label-1)
#         seg_img_slice[locs_x[temp_label_locs],locs_y[temp_label_locs]]= \
#         clustering_labels_unique[i]+base_count
        
#         seg_img_slice_temp = np.zeros(seg_img_slice.shape)
#         seg_img_slice_temp[seg_img_slice==clustering_labels_unique[i]+base_count]=1        
#         seg_img_slice_edt_temp = get_distance_transform(seg_img_slice_temp)
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]+base_count] = center_loc
    
#     return seg_img_slice, center_dict, clustering_labels_unique, clustering_labels_counts
#     """
# def find_cluster(img_slice, base_count):
#     clustering_labels = label_regions(img_slice, connectivity=1)
#     clustering_labels[clustering_labels>0]+=base_count
    
#     img_slice_edt = get_distance_transform(img_slice)
    
#     clustering_labels_unique, clustering_labels_counts = np.unique(clustering_labels, return_counts=True)
#     clustering_labels_counts = clustering_labels_counts[clustering_labels_unique>0]
#     clustering_labels_unique = clustering_labels_unique[clustering_labels_unique>0]
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         seg_img_slice_edt_temp = np.zeros(img_slice_edt.shape)
#         seg_img_slice_edt_temp[clustering_labels==clustering_labels_unique[i]]=img_slice_edt[clustering_labels==clustering_labels_unique[i]]
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]] = center_loc
    
#     return clustering_labels, center_dict, clustering_labels_unique, clustering_labels_counts


# # step 1 label each slice and get centers of each slice
# def label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=0):
#     base_count = 1
#     center_dict = {}
#     seg_slice_label = np.zeros(seg_onehot.shape)
    
#     touching_dict = {}
    
#     if axis==0:
#         for i in np.arange(seg_onehot.shape[0]):
#             print("processing slice: "+str(i), end="\r")
#             current_slice = seg_onehot[i,:,:]
#             if np.sum(current_slice)>0:
#                 current_slice_labeled, center_dict_slice, clustering_labels_unique, clustering_labels_counts = \
#                 find_cluster(current_slice, base_count=base_count)
#                 for label in center_dict_slice.keys():
#                     center_dict_slice[label] = [i, center_dict_slice[label][0], center_dict_slice[label][1]]
#                 center_dict.update(center_dict_slice)
#                 seg_slice_label[i,:,:] = current_slice_labeled
#                 base_count += len(clustering_labels_unique)
                
#                 if i>0:
#                     for label in center_dict_slice.keys():
#                         crop_img = get_crop_by_pixel_val(seg_slice_label[i-1:i+1,:,:], label, boundary_extend=1)
#                         crop_img_vals = np.unique(crop_img).astype(np.int)
#                         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                         crop_img_vals = crop_img_vals[crop_img_vals!=label]
                        
#                         if label not in touching_dict.keys():
#                             touching_dict[label]=[]
#                         for crop_img_val in crop_img_vals:
#                             touching_dict[label].append(crop_img_val)
#                         for crop_img_val in crop_img_vals:
#                             if crop_img_val not in touching_dict.keys():
#                                 touching_dict[crop_img_val]=[]
#                             touching_dict[crop_img_val].append(label)
                            
#             else:
#                 seg_slice_label[i,:,:] = 0
        
#         for label in center_dict.keys():
#             if label not in touching_dict.keys():
#                 crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#                 crop_img_vals = np.unique(crop_img).astype(np.int)
#                 crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                 crop_img_vals = crop_img_vals[crop_img_vals!=label]
#                 if len(crop_img_vals)==0:
#                     touching_dict[label] = []
#                 else:
#                     touching_dict[label] = crop_img_vals
                
#             #print("clustering_labels_unique, clustering_labels_counts: "+str((clustering_labels_unique, clustering_labels_counts)))
#             #print("center_dict_slice: "+str(center_dict_slice))
#             #print("base count: "+str(base_count))
#             #print("----------")
    
#         for label in touching_dict.keys():
#             touching_dict[label] = np.array(touching_dict[label])
#             touching_dict[label] = np.unique(touching_dict[label])
#     """
#     for idx,label in enumerate(center_dict.keys()):
#         print("find touching relationship of each slice: "+str(idx/len(center_dict.keys())), end="\r")
#         crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#         crop_img_vals = np.unique(crop_img).astype(np.int)
#         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#         crop_img_vals = crop_img_vals[crop_img_vals!=label]
#         touching_dict[label] = crop_img_vals
#     """
#     return seg_slice_label, center_dict, touching_dict

# # step 2 get_connection_dict
# def get_connection_dict(seg_slice_label, center_dict, touching_dict=None):
#     slice_idxs = list(center_dict.keys())
#     slice_idxs.reverse()
    
#     # init connection dict
#     global connection_dict
#     connection_dict = {}
#     for slice_idx in slice_idxs:
#         connection_dict[slice_idx] = {}
#         connection_dict[slice_idx]["loc"] = center_dict[slice_idx]
#         connection_dict[slice_idx]["before"] = []
#         connection_dict[slice_idx]["next"] = []
#         connection_dict[slice_idx]["is_bifurcation"] = False
#         connection_dict[slice_idx]["number_of_next"] = 0
#         connection_dict[slice_idx]["generation"] = 0
#         connection_dict[slice_idx]["is_processed"] = False
    
#     def get_touching_labels(input_img, val):
#         if touching_dict is not None:
#             return touching_dict[val]
#         else:
#             crop_img = get_crop_by_pixel_val(input_img, val, boundary_extend=1)
#             crop_img_vals = np.unique(crop_img).astype(np.int)
#             crop_img_vals = crop_img_vals[crop_img_vals!=0]
#             crop_img_vals = crop_img_vals[crop_img_vals!=val]

#             return crop_img_vals
    
#     def find_connection(slice_label_img, current_label, before_label):
#         global connection_dict
        
#         touching_labels = get_touching_labels(slice_label_img, current_label)
#         valid_next_labels = []
#         dist_to_valid_labels = []
#         processed_count = 0
#         for touching_label in touching_labels:
#             if connection_dict[touching_label]["is_processed"]==False \
#             and connection_dict[touching_label]["loc"][0]!=connection_dict[current_label]["loc"][0]:
#                 valid_next_labels.append(touching_label)
#                 dist_to_valid_labels.append(np.sum(((np.array(connection_dict[touching_label]["loc"])-np.array(connection_dict[current_label]["loc"]))**2)))
#             if connection_dict[touching_label]["is_processed"]==True:
#                 processed_count+=1
        
#         if len(valid_next_labels)>0:
#             valid_next_labels = np.array(valid_next_labels)
#             dist_to_valid_labels = np.array(dist_to_valid_labels)
#             sort_locs = np.argsort(dist_to_valid_labels)
#             valid_next_labels = valid_next_labels[sort_locs]
        
#         connection_dict[current_label]["before"].append(before_label)
#         connection_dict[current_label]["is_processed"] = True
#         #connection_dict[current_label]["generation"] = generation

#         print("current_label is "+str(current_label), end="\r")
#         print(connection_dict[current_label], end="\r")

#         if len(valid_next_labels)==0 or len(touching_labels)==processed_count:
#             return connection_dict
#         else:
#             for valid_next_label in valid_next_labels:
#                 if connection_dict[valid_next_label]["is_processed"]==False:
#                     connection_dict[current_label]["next"].append(valid_next_label)
#                     find_connection(slice_label_img, valid_next_label, current_label)
    
#     find_connection(seg_slice_label, current_label=slice_idxs[0], before_label=0)
    
#     for item in connection_dict.keys():
#         assert len(connection_dict[item]["before"])<=1, (item, connection_dict[item])
#         connection_dict[item]["number_of_next"] = len(connection_dict[item]["next"])
#         connection_dict[item]["is_bifurcation"] = (connection_dict[item]["number_of_next"]>1)
#         #if connection_dict[item]["is_bifurcation"]:
#         #    print(connection_dict[item])
            
#     def find_generation(current_label, generation):
#         global connection_dict
#         connection_dict[current_label]["generation"]=generation
#         if connection_dict[current_label]["number_of_next"]>0:
#             for next_label in connection_dict[current_label]["next"]:
#                 if connection_dict[current_label]["is_bifurcation"]:
#                     find_generation(next_label, generation+1)
#                 else:
#                     find_generation(next_label, generation)
#         else:
#             return connection_dict
    
#     find_generation(current_label=slice_idxs[0], generation=0)
    
#     return connection_dict

# def get_number_of_branch(connection_dict):
#     number_of_branch = 1
#     for label in connection_dict.keys():
#         if connection_dict[label]["is_bifurcation"]:
#             number_of_branch+=connection_dict[label]["number_of_next"]
#     return number_of_branch

# def get_tree_length(connection_dict, is_3d_len=True):
#     global tree_length
#     tree_length = 0
#     for label in connection_dict.keys():
#         if connection_dict[label]["before"][0]==0:
#             start_label = label
#             break
#     def get_tree_length_func(connection_dict, current_label):
#         global tree_length
#         if connection_dict[current_label]["number_of_next"]==0:
#             return
#         else:
#             current_branch_length = 0
#             for next_label in connection_dict[current_label]["next"]:
#                 if is_3d_len:
#                     current_branch_length += np.sqrt(np.sum((np.array(connection_dict[current_label]["loc"])-np.array(connection_dict[next_label]["loc"]))**2))
#                 else:
#                     current_branch_length += 1
#             print("len of "+str(current_label)+" branch is "+str(current_branch_length),end="\r")
#             tree_length += current_branch_length
#             for next_label in connection_dict[current_label]["next"]:
#                 get_tree_length_func(connection_dict, next_label)
#     get_tree_length_func(connection_dict, start_label)
#     return tree_length

# # tree detection V1

# import edt
# import numpy as np
# from skimage.measure import label, regionprops
# from skimage.feature import peak_local_max
# #from sklearn.cluster import DBSCAN

# def tree_detection(seg_onehot, axis=0):
#     seg_slice_label, center_dict, touching_dict = label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=axis)
#     connection_dict = get_connection_dict(seg_slice_label, center_dict, touching_dict)
#     number_of_branch = get_number_of_branch(connection_dict)
#     tree_length = get_tree_length(connection_dict, is_3d_len=True)
    
#     return seg_slice_label, connection_dict, number_of_branch, tree_length

# def get_distance_transform(img_oneshot):
#     return edt.edt(np.array(img_oneshot>0, dtype=np.uint32, order='F'), black_border=True, order='F', parallel=1)

# """
# def get_outlayer_of_a_3d_shape(a_3d_shape_onehot):
#     shape=a_3d_shape_onehot.shape
    
#     a_3d_crop_diff_x1 = a_3d_shape_onehot[0:shape[0]-1,:,:]-a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_x2 = -a_3d_shape_onehot[0:shape[0]-1,:,:]+a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_y1 = a_3d_shape_onehot[:,0:shape[1]-1,:]-a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_y2 = -a_3d_shape_onehot[:,0:shape[1]-1,:]+a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_z1 = a_3d_shape_onehot[:,:,0:shape[2]-1]-a_3d_shape_onehot[:,:,1:shape[2]]
#     a_3d_crop_diff_z2 = -a_3d_shape_onehot[:,:,0:shape[2]-1]+a_3d_shape_onehot[:,:,1:shape[2]]

#     outlayer = np.zeros(shape)
#     outlayer[1:shape[0],:,:] += np.array(a_3d_crop_diff_x1==1, dtype=np.int8)
#     outlayer[0:shape[0]-1,:,:] += np.array(a_3d_crop_diff_x2==1, dtype=np.int8)
#     outlayer[:,1:shape[1],:] += np.array(a_3d_crop_diff_y1==1, dtype=np.int8)
#     outlayer[:,0:shape[1]-1,:] += np.array(a_3d_crop_diff_y2==1, dtype=np.int8)
#     outlayer[:,:,1:shape[2]] += np.array(a_3d_crop_diff_z1==1, dtype=np.int8)
#     outlayer[:,:,0:shape[2]-1] += np.array(a_3d_crop_diff_z2==1, dtype=np.int8)
    
#     outlayer = np.array(outlayer>0, dtype=np.int8)
    
#     return outlayer
#     """

# def get_crop_by_pixel_val(input_3d_img, val, boundary_extend=1, axis=0):
#     locs = np.where(input_3d_img==val)
    
#     shape_of_input_3d_img = input_3d_img.shape
    
#     min_x = np.min(locs[0])
#     max_x =np.max(locs[0])
#     min_y = np.min(locs[1])
#     max_y =np.max(locs[1])
#     min_z = np.min(locs[2])
#     max_z =np.max(locs[2])
    
#     if axis==0:
#         x_s = np.clip(min_x-boundary_extend, 0, shape_of_input_3d_img[0])
#         x_e = np.clip(max_x+boundary_extend+1, 0, shape_of_input_3d_img[0])
#         y_s = np.clip(min_y, 0, shape_of_input_3d_img[1])
#         y_e = np.clip(max_y+1, 0, shape_of_input_3d_img[1])
#         z_s = np.clip(min_z, 0, shape_of_input_3d_img[2])
#         z_e = np.clip(max_z+1, 0, shape_of_input_3d_img[2])
    
#     #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
#     crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    
#     return crop_3d_img

# """
# def find_cluster_by_dbscan(img_slice, base_count, threshold=0, dbscan_min_samples=1, dbscan_eps=1):
#     locs=np.where(img_slice>0)
#     locs_x=locs[0]
#     locs_y=locs[1]
#     locs_len=locs[0].shape[0]
#     locs_reshape=np.concatenate((locs[0].reshape(locs_len,1),
#                                  locs[1].reshape(locs_len,1)),axis=1)
    
#     clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean').fit(locs_reshape)
#     clustering_labels=clustering.labels_
#     clustering_labels_unique,clustering_labels_counts=np.unique(clustering_labels, return_counts=True)
    
#     #clustering_labels_counts=clustering_labels_counts[clustering_labels_unique>-1]
#     #clustering_labels_unique=clustering_labels_unique[clustering_labels_unique>-1] # delete noise
#     clustering_labels_unique=clustering_labels_unique+1
    
#     clustering_labels_unique=clustering_labels_unique[np.where(clustering_labels_counts>threshold)]
#     clustering_labels_counts=clustering_labels_counts[np.where(clustering_labels_counts>threshold)]
    
#     seg_img_slice=np.zeros(img_slice.shape)
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         temp_label=clustering_labels_unique[i]
#         temp_label_locs=np.where(clustering_labels==temp_label-1)
#         seg_img_slice[locs_x[temp_label_locs],locs_y[temp_label_locs]]= \
#         clustering_labels_unique[i]+base_count
        
#         seg_img_slice_temp = np.zeros(seg_img_slice.shape)
#         seg_img_slice_temp[seg_img_slice==clustering_labels_unique[i]+base_count]=1        
#         seg_img_slice_edt_temp = get_distance_transform(seg_img_slice_temp)
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]+base_count] = center_loc
    
#     return seg_img_slice, center_dict, clustering_labels_unique, clustering_labels_counts
#     """
# def find_cluster(img_slice, base_count):
#     clustering_labels = label(img_slice, connectivity=1)
#     clustering_labels[clustering_labels>0]+=base_count
    
#     img_slice_edt = get_distance_transform(img_slice)
    
#     clustering_labels_unique, clustering_labels_counts = np.unique(clustering_labels, return_counts=True)
#     clustering_labels_counts = clustering_labels_counts[clustering_labels_unique>0]
#     clustering_labels_unique = clustering_labels_unique[clustering_labels_unique>0]
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         seg_img_slice_edt_temp = np.zeros(img_slice_edt.shape)
#         seg_img_slice_edt_temp[clustering_labels==clustering_labels_unique[i]]=img_slice_edt[clustering_labels==clustering_labels_unique[i]]
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]] = center_loc
    
#     return clustering_labels, center_dict, clustering_labels_unique, clustering_labels_counts


# # step 1 label each slice and get centers of each slice
# def label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=0):
#     base_count = 1
#     center_dict = {}
#     seg_slice_label = np.zeros(seg_onehot.shape)
    
#     touching_dict = {}
    
#     if axis==0:
#         for i in np.arange(seg_onehot.shape[0]):
#             print("processing slice: "+str(i), end="\r")
#             current_slice = seg_onehot[i,:,:]
#             if np.sum(current_slice)>0:
#                 current_slice_labeled, center_dict_slice, clustering_labels_unique, clustering_labels_counts = \
#                 find_cluster(current_slice, base_count=base_count)
#                 for label in center_dict_slice.keys():
#                     center_dict_slice[label] = [i, center_dict_slice[label][0], center_dict_slice[label][1]]
#                 center_dict.update(center_dict_slice)
#                 seg_slice_label[i,:,:] = current_slice_labeled
#                 base_count += len(clustering_labels_unique)
                
#                 if i>0:
#                     for label in center_dict_slice.keys():
#                         crop_img = get_crop_by_pixel_val(seg_slice_label[i-1:i+1,:,:], label, boundary_extend=1)
#                         crop_img_vals = np.unique(crop_img).astype(np.int)
#                         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                         crop_img_vals = crop_img_vals[crop_img_vals!=label]
                        
#                         if label not in touching_dict.keys():
#                             touching_dict[label]=[]
#                         for crop_img_val in crop_img_vals:
#                             touching_dict[label].append(crop_img_val)
#                         for crop_img_val in crop_img_vals:
#                             if crop_img_val not in touching_dict.keys():
#                                 touching_dict[crop_img_val]=[]
#                             touching_dict[crop_img_val].append(label)
                            
#             else:
#                 seg_slice_label[i,:,:] = 0
        
#         for label in center_dict.keys():
#             if label not in touching_dict.keys():
#                 crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#                 crop_img_vals = np.unique(crop_img).astype(np.int)
#                 crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                 crop_img_vals = crop_img_vals[crop_img_vals!=label]
#                 if len(crop_img_vals)==0:
#                     touching_dict[label] = []
#                 else:
#                     touching_dict[label] = crop_img_vals
                
#             #print("clustering_labels_unique, clustering_labels_counts: "+str((clustering_labels_unique, clustering_labels_counts)))
#             #print("center_dict_slice: "+str(center_dict_slice))
#             #print("base count: "+str(base_count))
#             #print("----------")
    
#         for label in touching_dict.keys():
#             touching_dict[label] = np.array(touching_dict[label])
#             touching_dict[label] = np.unique(touching_dict[label])
#     """
#     for idx,label in enumerate(center_dict.keys()):
#         print("find touching relationship of each slice: "+str(idx/len(center_dict.keys())), end="\r")
#         crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#         crop_img_vals = np.unique(crop_img).astype(np.int)
#         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#         crop_img_vals = crop_img_vals[crop_img_vals!=label]
#         touching_dict[label] = crop_img_vals
#     """
#     return seg_slice_label, center_dict, touching_dict

# # step 2 get_connection_dict
# def get_connection_dict(seg_slice_label, center_dict, touching_dict=None):
#     slice_idxs = list(center_dict.keys())
#     slice_idxs.reverse()
    
#     # init connection dict
#     global connection_dict
#     connection_dict = {}
#     for slice_idx in slice_idxs:
#         connection_dict[slice_idx] = {}
#         connection_dict[slice_idx]["loc"] = center_dict[slice_idx]
#         connection_dict[slice_idx]["before"] = 0
#         connection_dict[slice_idx]["next"] = 0
#         connection_dict[slice_idx]["is_bifurcation"] = False
#         connection_dict[slice_idx]["number_of_next"] = 0
#         connection_dict[slice_idx]["generation"] = 0
#         connection_dict[slice_idx]["is_processed"] = False
    
#     def get_touching_labels(input_img, val):
#         if touching_dict is not None:
#             return touching_dict[val]
#         else:
#             crop_img = get_crop_by_pixel_val(input_img, val, boundary_extend=1)
#             crop_img_vals = np.unique(crop_img).astype(np.int)
#             crop_img_vals = crop_img_vals[crop_img_vals!=0]
#             crop_img_vals = crop_img_vals[crop_img_vals!=val]

#             return crop_img_vals
    
#     def find_connection(slice_label_img, current_label, before_label, generation):
#         global connection_dict
        
#         touching_labels = get_touching_labels(slice_label_img, current_label)
#         valid_next_labels = []
#         processed_count = 0
#         for touching_label in touching_labels:
#             if connection_dict[touching_label]["is_processed"]==False \
#             and connection_dict[touching_label]["loc"][0]!=connection_dict[current_label]["loc"][0]:
#                 valid_next_labels.append(touching_label)
#             if connection_dict[touching_label]["is_processed"]==True:
#                 processed_count+=1
        
#         connection_dict[current_label]["before"] = before_label
#         connection_dict[current_label]["generation"] = generation
#         connection_dict[current_label]["is_processed"] = True
#         connection_dict[current_label]["next"] = valid_next_labels
#         connection_dict[current_label]["is_bifurcation"] = (len(valid_next_labels)>=2)
#         connection_dict[current_label]["number_of_next"] = len(valid_next_labels)

#         print("current_label is "+str(current_label), end="\r")
#         print(connection_dict[current_label], end="\r")

#         if connection_dict[current_label]["number_of_next"]==0 or len(touching_labels)==processed_count:
#             return connection_dict
#         else:
#             for valid_next_label in valid_next_labels:
#                 if connection_dict[current_label]["is_bifurcation"]:
#                     find_connection(slice_label_img, valid_next_label, current_label, generation+1)
#                 else:
#                     find_connection(slice_label_img, valid_next_label, current_label, generation)
    
#     find_connection(seg_slice_label, current_label=slice_idxs[0], before_label=0, generation=0)
    
#     return connection_dict

# def get_number_of_branch(connection_dict):
#     number_of_branch = 1
#     for label in connection_dict.keys():
#         if connection_dict[label]["is_bifurcation"]:
#             number_of_branch+=connection_dict[label]["number_of_next"]
#     return number_of_branch

# def get_tree_length(connection_dict, is_3d_len=True):
#     global tree_length
#     tree_length = 0
#     for label in connection_dict.keys():
#         if connection_dict[label]["before"]==0:
#             start_label = label
#             break
#     def get_tree_length_func(connection_dict, current_label):
#         global tree_length
#         if connection_dict[current_label]["number_of_next"]==0:
#             return
#         else:
#             current_branch_length = 0
#             for next_label in connection_dict[current_label]["next"]:
#                 if is_3d_len:
#                     current_branch_length += np.sqrt(np.sum((np.array(connection_dict[current_label]["loc"])-np.array(connection_dict[next_label]["loc"]))**2))
#                 else:
#                     current_branch_length += 1
#             print("len of "+str(current_label)+" branch is "+str(current_branch_length),end="\r")
#             tree_length += current_branch_length
#             for next_label in connection_dict[current_label]["next"]:
#                 get_tree_length_func(connection_dict, next_label)
#     get_tree_length_func(connection_dict, start_label)
#     return tree_length