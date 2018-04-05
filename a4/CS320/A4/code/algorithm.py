# CSC320 Winter 2018
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################
    best_D = global_vars
    new_f = f_heap.copy()
    source_shape = source_patches.shape
    if best_D is None:
        new_D = np.zeros(source_patches.shape[0:2])
        new_D.fill(255)
    else:
        new_D = best_D
    if random_enabled:
        u_len = np.ceil(np.divide(np.log(np.divide(1, float(w))), np.log(alpha))).astype(np.int64)
        NNF_Y = np.zeros(u_len)
        NNF_X = np.zeros(u_len)
        ALPHA = np.zeros(u_len)
        ALPHA.fill(alpha)
        vector_w = np.zeros(u_len)
        vector_w.fill(w)
        r_c = np.arange(0, u_len, 1)
        const = np.power(ALPHA, r_c)
    if odd_iteration:
        range_y, range_x = range(shape_source[0])[::-1], range(shape_source[1])[::-1]
    else:
        range_y, range_x = range(shape_source[0]), range(shape_source[1])
    range_k = range(f_heap.shape[2])
    for i in range_y:
        for j in range_x:
            for k in range_k:
                min_val = nsmallest(1, new_f[i, j])
                new_D[i,j] = min_val[0][0]
                source = source_patches[i, j, ::]
                if propagation_enabled:
                    if odd_iteration:
                        direction = 1
                    else:
                        direction = -1
                    if 0 <= i + direction < source_shape[0] and 0 <= j + direction < source_shape[1]:
                        if (new_f[i + direction, j,k][2] + i >= target_patches.shape[0]) or (new_f[i + direction, j, k][3] + j >= target_patches.shape[1]):
                            v_temp = new_f[i + direction, j, k]
                            temp_y = (v_temp[2] + i).astype(np.int32)
                            temp_x = (v_temp[3] + j).astype(np.int32)
                            if new_f[i + direction, j, k][2] + i >= target_patches.shape[0]:
                                temp_y = target_patches.shape[0] - 1
                            if new_f[i + direction, j, k][3] + j >= target_patches.shape[1]:
                                temp_x = target_patches.shape[1] - 1
                            cur_target = target_patches[temp_y, temp_x, ...]
                            cur_target[np.isnan(cur_target)] = np.nan
                            cur_dist = np.subtract(cur_target, source)
                            cur_dist = np.abs(cur_dist)
                            if np.isnan(cur_dist).any():
                                current_distance = np.nanmean(cur_dist)
                            else:
                                current_distance = cur_dist.mean()
                        else:
                            v_temp = new_f[i + direction, j, k]
                            temp_y = (v_temp[2] + i).astype(np.int32)
                            temp_x = (v_temp[3] + j).astype(np.int32)
                            cur_target = target_patches[temp_y, temp_x, ...]
                            cur_target[np.isnan(cur_target)] = np.nan
                            cur_dist = np.subtract(cur_target, source)
                            cur_dist = np.abs(cur_dist)
                            if np.isnan(cur_dist).any():
                                current_distance = np.nanmean(cur_dist)
                            else:
                                current_distance = cur_dist.mean()
                        dist_1_x = temp_x
                        dist_1_y = temp_y
                        dist_1 = current_distance
                        if new_f[i, j + direction, k][2] + i >= target_patches.shape[0] or (new_f[i, j + direction, k][3] + j >= target_patches.shape[1]):
                            v_temp = new_f[i, j + direction, k]
                            temp_y = (v_temp[2] + i).astype(np.int32)
                            temp_x = (v_temp[3] + j).astype(np.int32)
                            if new_f[i, j + direction, k][2] + i >= target_patches.shape[0]:
                                temp_y = target_patches.shape[0] - 1
                            if new_f[i, j + direction, k][3] + j >= target_patches.shape[1]:
                                temp_x = target_patches.shape[1] - 1
                            cur_target = target_patches[temp_y, temp_x, ...]
                            cur_dist = np.abs(np.subtract(cur_target, source))
                            if np.isnan(cur_dist).any():
                                current_distance = np.nanmean(cur_dist)
                            else:
                                current_distance = cur_dist.mean()
                        else:
                            v_temp = new_f[i, j + direction, k]
                            temp_y = (v_temp[2] + i).astype(np.int32)
                            temp_x = (v_temp[3] + j).astype(np.int32)
                            cur_target = target_patches[temp_y, temp_x, ...]
                            cur_dist = np.abs(np.subtract(cur_target, source))
                            if np.isnan(cur_dist).any():
                                current_distance = np.nanmean(cur_dist)
                            else:
                                current_distance = cur_dist.mean()
                        dist_2_x = temp_x
                        dist_2_y = temp_y
                        dist_2 = current_distance
                        if new_D[i, j] < -dist_1 or new_D[i, j] < -dist_2:
                            if dist_1 < dist_2:
                                x = dist_1_x - j
                                y = dist_1_y - i
                                if not ((y, x) in f_coord_dictionary[i][j]):
                                    heappushpop([new_f[i, j]], [-dist_1, min_val[0][1], y, x])
                                    f_coord_dictionary.update({min_val[0][1]: (y, x)})
                            else:
                                x = dist_2_x - j
                                y = dist_2_y - i
                                if not ((y, x) in f_coord_dictionary[i][j]):
                                    print(i)
                                    print(j)
                                    print(k)
                                    print(x)
                                    print((dist_2, min_val[0][1], y, x))
                                    sys.stdout.flush()
                                    heappush([new_f[i, j]], (- dist_2 ,min_val[0][1], y, x))
                                    print(new_f[i,j])
                                    heappushpop([new_f[i, j]], (- dist_2 ,min_val[0][1], y, x))
                                    f_coord_dictionary.update({min_val[0][1]: (y, x)})
                v_0 = new_f[i, j, k]
                source = source_patches[i, j, ::]
                if random_enabled:
                    if v_0[2] < 0:
                        v_0[2] += target_patches.shape[0]
                    if v_0[3] < 0:
                        v_0[3] += target_patches.shape[1]
                    NNF_Y.fill(v_0[2])
                    NNF_X.fill(v_0[3])
                    random_y = np.random.randint(low=-v_0[2], high=target_patches.shape[0] - v_0[2], size=u_len)
                    random_x = np.random.randint(low=-v_0[3], high=target_patches.shape[1] - v_0[3], size=u_len)
                    u_y = np.round(np.add(NNF_Y, np.multiply(const, random_y)))
                    u_x = np.round(np.add(NNF_X, np.multiply(const, random_x)))
                    target = target_patches[u_y.astype(np.int64), u_x.astype(np.int64), ...]
                    target[np.isnan(target)] = np.nan
                    tile_source = np.tile(source, (10, 1, 1))
                    dist = np.subtract(target, tile_source)
                    dist = np.abs(dist)
                    dist_val = np.zeros(u_len)
                    for l in range(u_len):
                        if np.isnan(dist[l]).any():
                            dist_val[l] = np.nanmean(dist[l])
                        else:
                            dist_val[l] = dist[l].mean()
                    min_val = nsmallest(1, f_heap[i,j])
                    if min(dist_val) < - min_val[0][0]:
                        index = np.argmin(dist_val)
                        cur_dist = dist_val[index]
                        x = u_x[index]
                        y = u_y[index]
                        if not ((y,x) in f_coord_dictionary[i][j]):
                            if y >= target_patches.shape[1]:
                                print('rand')
                                print(i)
                                print(j)
                                print(k)
                                print(x)
                            heappushpop([new_f[i, j]], (-cur_dist, min_val[0][1], y, x))
                            f_coord_dictionary.update({min_val[0][1]: (y, x)})

    global_vars = new_D
    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    Count = 0
    source_shape = source_patches.shape[0:2]
    k = len(f_k)
    range_y = range(source_shape[0])
    range_x = range(source_shape[1])

    mat_i = np.zeros(k)
    mat_j = np.zeros(k)
    f_heap_shape = [source_patches.shape[0], source_patches.shape[1], k, 4]
    f_heap = np.empty(f_heap_shape)
    f_coord_dictionary = []
    for i in range_y:
        f_coord_dictionary_i = [{} for x in range(source_patches.shape[1])]
        for j in range_x:
            mat_i.fill(i)
            mat_j.fill(j)
            source = source_patches[i, j, ::]
            for m in range(k):
                v_0 = f_k[m, i, j, :]
                cur_y = v_0[0] + i
                cur_x = v_0[1] + j

                cur_target = target_patches[cur_y, cur_x, ...]
                cur_target[np.isnan(cur_target)] = np.nan
                cur_dist = np.abs(np.subtract(cur_target, source))
                if np.isnan(cur_dist).any():
                    current_distance = np.nanmean(cur_dist)
                else:
                    current_distance = cur_dist.mean()
                temp_f_heap = []
                f_coord_dictionary_i[j].update({Count:(cur_y, cur_x)})
                heappush(temp_f_heap, (-current_distance, Count, cur_y, cur_x))
                Count += 1
            f_heap[i][j] = temp_f_heap
            f_coord_dictionary.append(f_coord_dictionary_i)

    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    k = f_heap.shape[2]
    dimentions = [k, f_heap.shape[0], f_heap.shape[1]]
    f_k_x = np.empty(dimentions)
    f_k_y = np.empty(dimentions)
    D_k = np.empty(dimentions)
    for l in range(k):
        for i in range(f_heap.shape[0]):
            for j in range(f_heap.shape[1]):
                f_k_x[l][i][j] = nlargest(k, f_heap[i][j])[l][3]
                f_k_y[l][i][j] = nlargest(k, f_heap[i][j])[l][2]
                D_k[l][i][j] = nlargest(k, f_heap[i][j])[l][0]
    print(f_k_y.shape)
    print(f_k_x.shape)

    f_k = np.stack((f_k_y, f_k_x), 3)
    print('-----------------')
    print(f_k.shape)
    print('-----------------')
    sys.stdout.flush()
    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    f_k, D_k = NNF_heap_to_NNF_matrix(f_heap)
    f_k = f_k.swapaxes(0, 2).swapaxes(0, 1)
    D_k = D_k.swapaxes(0, 2).swapaxes(0, 1)
    k = f_k.shape[2]
    exp_D_arr = np.exp(-D_k / (h ** 2))
    z_arr = exp_D_arr.sum(axis=-1)
    w_arr = exp_D_arr / dim_extend(z_arr)
    trg_rearr_shape = tuple(list(D_k.shape) + [3])
    trg_rearr = np.empty(trg_rearr_shape)
    for i in range(k):
        trg_rearr[..., i, :] = reconstruct_source_from_target(target, f_k[..., i, :])
    weighted_pixels = trg_rearr * dim_extend(w_arr)
    summed_pixels = weighted_pixels.sum(axis=-2)
    denoised = summed_pixels
    #############################################

    return denoised




#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################


#############################################



# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    tar_shape = target.shape
    coor = make_coordinates_matrix(tar_shape, step=1)

    NNF_Y,NNF_X = np.dsplit(f, 2)
    coor_y, coor_x = np.dsplit(coor, 2)

    axis_y = coor_y + NNF_Y
    axis_x = coor_x + NNF_X

    temp = target[axis_y, axis_x, ]
    rec_source = np.squeeze(temp)

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
