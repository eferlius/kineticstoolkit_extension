import numpy as np
import scipy

def containsScalars(iterable):
    '''
    Checks if the iterable contains scalar or other iterables
    eg:
    iterable = [1,2,3] -> contains scalar
    iterable = [[1,2,3], [4,5,6]] -> contains list
    iterable = [[1,2,3]] -> contains list

    Parameters
    ----------
    iterable : list or array or other iterable
        you want to know if it contains scalar or iterables inside.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    # if the first element is a scalar
    if np.isscalar(iterable[0]):
        return True
    else: # could be a list of lists or list of other iterables
        return False

def mov_avg(y, samples_before, samples_after = -1):
    '''
    If nan are contained, calls mov_avg_loop()
    If not nans, then calls mov_avg_no_nan()
    
    Please refer to their documentation

    Parameters
    ----------
    y : array
        to be filtered.
    samples_before : int
        how many samples before the current one should be considered for the 
        sliding window.
    samples_after : int, optional
        how many samples after the current one should be considered for the 
        sliding window. The default is -1, which makes samples_after = samples_before.

    Returns
    -------
    y_filt : array, same type of y
        filtered array.

    '''
    if np.isnan(y).any: # slow but handles nan
        return mov_avg_loop(y, samples_before, samples_after = -1)
    else: # fast, it's possible to use it since no nans in the array
        return mov_avg_no_nan(y, samples_before, samples_after = -1)

def mov_avg_loop(y, samples_before, samples_after = -1):
    '''
    NB: slow but doesn't handles nan
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude of samples_before and samples_after

    if samples_before = samples_after = 0, returns the same array considering 
    only the current element
    if samples_before = samples_after = 1, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two adjacent
    if samples_before = samples_after = 2, computes the mean on a sliding window 
    of amplitude = 5 considering the current element and the two before and the two after
    if samples_before = 2 and samples_after = 0, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two before
    if samples_before = 0 and samples_after = 2, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two after

    The duration of execution is proportional to the length of y

    Parameters
    ----------
    y : array
        to be filtered.
    samples_before : int
        how many samples before the current one should be considered for the 
        sliding window.
    samples_after : int, optional
        how many samples after the current one should be considered for the 
        sliding window. The default is -1, which makes samples_after = samples_before

    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''

    # if samples_after it's not specified
    if samples_after == -1:
        samples_after = samples_before

    y_filt = y.copy()
    for i in range(len(y)):
        y_filt[i] = np.nanmean(y[np.maximum(i-samples_before,0):np.minimum(i+samples_after,len(y))+1])
    return y_filt

def mov_avg_no_nan(y, samples_before, samples_after = -1):
    '''
    NB: fast but doesn't handle nan
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude of samples_before and samples_after

    if samples_before = samples_after = 0, returns the same array considering 
    only the current element
    if samples_before = samples_after = 1, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two adjacent
    if samples_before = samples_after = 2, computes the mean on a sliding window 
    of amplitude = 5 considering the current element and the two before and the two after
    if samples_before = 2 and samples_after = 0, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two before
    if samples_before = 0 and samples_after = 2, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two after

    Equivalent to mov_avg_loop but much faster, the duration of execution is not
    heavily affected from the length of y

    Parameters
    ----------
    y : array
        to be filtered.
    samples_before : int
        how many samples before the current one should be considered for the 
        sliding window.
    samples_after : int, optional
        how many samples after the current one should be considered for the 
        sliding window. The default is -1, which makes samples_after = samples_before.

    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''

    # if samples_after it's not specified
    if samples_after == -1:
        samples_after = samples_before

    # amplitude of the window
    size = samples_before + samples_after + 1
    # execution of the filtering
    y_filt = scipy.ndimage.uniform_filter1d(y, size = size, origin = 0, mode = 'reflect')

    # since only size is specified, how to translate the array?
    shift = int(np.ceil((samples_after-samples_before)/2))

    if shift > 0:
        # translation backward
        y_filt[:-shift] = y_filt[shift:]
        y_filt[-shift:] = np.nan*shift
    if shift < 0:
        # translation foreward
        y_filt[-shift:] = y_filt[:shift]
        y_filt[:-shift] = np.nan*abs(shift)

    # fix the first samples
    for i in range(0, min(samples_before + abs(shift), len(y)), 1):
        y_filt[i] = np.nanmean(y[np.maximum(i-samples_before,0):np.minimum(i+samples_after,len(y))+1])
    # fix the last samples
    for i in range(len(y)-1, max(0, len(y) - samples_after - abs(shift) - 1), -1):
        y_filt[i] = np.nanmean(y[np.maximum(i-samples_before,0):np.minimum(i+samples_after,len(y))+1])

    return y_filt

def mov_avg_time(y, freq, time_units_before, time_units_after = -1):
    '''
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude depending on the corrispondent 
    moment, specified in x array. 

    NB: This function assumes constant sampling frequency.
    If the sampling freq is not constant, use mov_avg_time_variable_freq

    Parameters
    ----------
    y : array
        to be filtered.
    freq : float
        frequency of acquisition of the y array, assumed constant.
    time_units_before : float
        how large is the window on the left. In the same meas units of x array
    time_units_after : float, optional
        how large is the window on the right. In the same meas units of x array
        The default is -1, which makes time_units_after = time_units_before.


    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''
    samples_before = int(time_units_before * freq)
    if time_units_after != -1:
        samples_after = int(time_units_after * freq)
    else:
        samples_after = -1

    y_filt = mov_avg(y, samples_before, samples_after)

    return y_filt

def mov_avg_time_variable_freq(x, listYarrays, time_units_before, time_units_after = -1):
    '''
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude depending on the corrispondent 
    moment, specified in x array. 

    NB: Use this function when necessary, (data acquired at not constant freq), 
    use mov_avg_time instead which is much faster.

    Parameters
    ----------
    x : array
        contains the moment of acquisition of each frame of y.
    y : array
        to be filtered.
    time_units_before : float
        how large is the window on the left. In the same meas units of x array
    time_units_after : float, optional
        how large is the window on the right. In the same meas units of x array
        The default is -1, which makes time_units_after = time_units_before.

    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''
    oneArrayFlag = False
    # add a dimension
    if containsScalars(listYarrays):
        oneArrayFlag = True
        listYarrays = [listYarrays]

    # if time_units_after it's not specified
    if time_units_after == -1:
        time_units_after = time_units_before
    # conversion to float in order to be able to add nan
    x = np.array(x).astype("float")

    arrayYarrays = np.array(listYarrays).astype("float")
    arrayYarrays_filt = arrayYarrays.copy()
    for i in range(len(x)):
        this_moment = x[i]
        timing_window = x.copy()
        # give nan value to all the samples before
        timing_window[timing_window<this_moment-time_units_before] = np.nan
        # give nan values to all the samples after
        timing_window[timing_window>this_moment+time_units_after ] = np.nan
        # get indexes where timing_window is not nan
        indexes = np.argwhere(~np.isnan(timing_window))
        # consider y only where timing_window is not nan
        arrayYarrays_filt[:,i] = np.squeeze(np.nanmean(arrayYarrays[:,indexes], axis = 1))
    if oneArrayFlag: # bring back to single dimension
        return np.squeeze(arrayYarrays_filt)
    return arrayYarrays_filt
