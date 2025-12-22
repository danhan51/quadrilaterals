# Functions for extracting relevant data from behavioral files
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

EVENT_CODES_TO_EVENT_NAME = {
    9: 'trial_start',
    18: 'trial_end_ml2', #trial end by ml2 (after blue idle screen flashes)
    10: 'fix_cue',
    20: 'sample_on',
    21: 'sample_off',
    49: 'timeout',
    50: 'rew', 
    51: 'trial_end_blue' #trial ends, and blue idle screen flashes
}

def loadQuad(animal, date):
    """
    loads quad object for given animal and date
    NOTE: Here is where to change dir paths to match your system :)
    """
    if animal == 'Diego':
        subject = 'S1'
    elif animal == 'Pancho':
        subject = 'S2'


    basedir = '/home/danhan/code/data/quad_data'
    paths = {
    'ml2_dir': f'{basedir}/{animal}/behavior/{date}',
    # 'conditions_dir': f'{basedir}/{animal}/conditions_quadrilaterals_{subject}.txt', 
    'tdt_dir': f'{basedir}/{animal}/tdt/{date}',
    'spikes_dir': f'{basedir}/{animal}/spikes_postprocess/{date}/DATSTRUCT_CLEAN_MERGED.mat'
    }

    return Quads(paths)


def filter_df(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    """
    Filter a DataFrame based on a dictionary of allowed values.
    """
    
    mask = pd.Series(True, index=df.index)

    for col, allowed_vals in filter_dict.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        
        mask &= df[col].isin(allowed_vals)

    return df[mask]

def group_and_average(df, group_by, average_prefix):
    """
    basically SQL
    Flexuibly handles multiple columns to average over
    """

    #can change this if needed, right now will average all columns with given prefix
    cols_to_agg = [c for c in df.columns if c.startswith(average_prefix)]
    if not cols_to_agg:
        return None

    def agg(group):
        result = {}

        for col in cols_to_agg:
            stack = list(group[col])
            lengths = [len(x) for x in stack]

            # Ensure same-length arrays
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Arrays in column '{col}' have different lengths: {sorted(set(lengths))}"
                )

            arr = np.vstack(stack)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            sem  = std / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)

            result[f"{col}_mean"] = mean
            result[f"{col}_std"]  = std
            result[f"{col}_sem"]  = sem

        result["n"] = arr.shape[0]
        return pd.Series(result)

    grouped = df.groupby(group_by).apply(agg).reset_index()

    for col in grouped.columns:
        if col.endswith("_mean") or col.endswith("_std") or col.endswith("_sem"):
            grouped[col] = grouped[col].apply(np.asarray)

    return grouped


# def group_and_average(df, group_by, average_column):
#     """group and average on given coilumns"""

#     def avg_lists(list_of_lists):
#         arr = np.vstack(list_of_lists)
#         return arr.mean(axis=0)

#     return (
#         df.groupby(group_by)[average_column]
#           .apply(avg_lists)
#           .reset_index()
#     )


class Quads:
    def __init__(self, paths):
        self.paths = paths
        """
        format (if not use load function):
            paths = {
            'ml2_dir': f'{basedir}/{animal}/behavior/{date}',
            'conditions_dir': f'{basedir}/{animal}/conditions_quadrilaterals_{subject}.txt', 
            'tdt_dir': f'{basedir}/{animal}/tdt/{date}',
            'spikes_dir': f'{basedir}/{animal}/spikes_postprocess/{date}/DATSTRUCT_CLEAN_MERGED.mat'
            }
        """

        #mostly internal attrs
        self._session_rec_names = None
        self.ml2_dat_list = self.loadML2Data()              # dict with all trial info (convert bhv2 to mat)
        # self.conditions = self.loadCondtionsFile() # conditions file loaded from text as pd df
        self.tdt_dat_list = self.loadTdtData()
        self._session_durations = self.getSessionDurations()
        

        assert len(self.ml2_dat_list) > 0, 'no beh data'
        assert len(self.tdt_dat_list) > 0, 'no tdt data'

        #pretty data
        self.prettyBeh = self.generatePrettyBehDF()
        self.prettyTdt = self.generatePrettyTdtDF()
        self.spikeTimes = self.loadSpikeTimes()

        self.Dat = self.mergeBehTdt()

        #sanity check make sure isi correlated btwn beh and tdt times
        self.checkISI()

    
    
    ### LOAD DATA
    def loadML2Data(self):
        """
        Function to load ml2 data, returns list of beh dat structs. Pretty function will concat
        """
        from scipy.io import loadmat

        data_dir = self.paths['ml2_dir']
        bhv_list = [f'{data_dir}/{f}' for f in os.listdir(data_dir) if f.endswith('mat')]
        beh_dat_list = []
        #make sure list is ordered by session properly
        bhv_list_ordered = []
        for i in range(1,len(bhv_list)+1):
            for bhv in bhv_list:
                if bhv.split('.')[0].endswith(str(i)):
                    bhv_list_ordered.append(bhv)
        for bhv in bhv_list_ordered:
            beh_dat_list.append(loadmat(bhv,simplify_cells=True))
        return beh_dat_list

    def loadTdtData(self, load_eye_tracking = False, load_raw_ephys = False):
        """
        Function to load tdt data, pretty function concats
        """
        from tdt import read_block
        tdt_list = [entry.name for entry in os.scandir(self.paths['tdt_dir']) if entry.is_dir()]
        tdt_dat_list = []
        evs_load = ['streams', 'epocs']
        stores_load = ['SMa1', 'Rew/','PhD2']
        if load_eye_tracking:
            stores_load.append('Eyee')
        if load_raw_ephys:
            stores_load.extend(['RSn2,RSn3'])
        #probably overcomplicated but wnated ot make sure it does what I want
        #make sure tdt list sessions are ordered properly in time
        tdt_list_ordered = []
        sess_times = sorted(set([int(s.split('-')[-1]) for s in tdt_list]))
        for sess_time in sess_times:
            for sess in tdt_list:
                if sess_time == int(sess[-6:]):
                    tdt_list_ordered.append(sess)
        self._session_rec_names = tdt_list_ordered
        for tdt_session in tdt_list_ordered:
            fullpath = f"{self.paths['tdt_dir']}/{tdt_session}"
            session_data = read_block(fullpath,
                                    evtype=evs_load,
                                    store=stores_load)
            tdt_dat_list.append(session_data)
        return tdt_dat_list

    def loadCondtionsFile(self):
        """
        Function to load the condtions file as df for getting stim names
        """
        assert False, 'old method dont use'
        with open(self.paths['conditions_dir'], 'r') as f:
            conditions = pd.read_csv(f, delimiter = '\t')
        return conditions
    
    def loadSpikeTimes(self):
        """
        Function to load spike times
        """
        import mat73
        spikes_data = mat73.loadmat(self.paths['spikes_dir'])['DATSTRUCT']
        df = pd.DataFrame.from_dict(spikes_data)

        return df

    def loadTdtNeural(self, trange, nodata = 0):
        """
        Generally dont use, lots of data an dnot very useful except debugging
        Function to load tdt neural data
        """
        from tdt import read_block
        tdt_list = os.listdir(self.paths['tdt_dir'])
        tdt_dat_list = []
        evs_load = ['streams']
        stores_load = ['RSn2','RSn3']

        #probably overcomplicated but wnated ot make sure it does what I want
        #make sure tdt list sessions are ordered properly in time
        tdt_list_ordered = []
        sess_times = sorted(set([int(s.split('-')[-1]) for s in tdt_list]))
        for sess_time in sess_times:
            for sess in tdt_list:
                if sess_time == int(sess[-6:]):
                    tdt_list_ordered.append(sess)
        self._session_rec_names = tdt_list_ordered
        for tdt_session in tdt_list_ordered:
            fullpath = f"{self.paths['tdt_dir']}/{tdt_session}"
            session_data = read_block(fullpath,
                                    evtype=evs_load,
                                    store=stores_load,
                                    t1=trange[0],t2=trange[1],
                                    nodata=nodata)
            tdt_dat_list.append(session_data)
        return tdt_dat_list
    
    def loadTdtNeuralDup(self, trange):
        """
        Generally dont use, lots of data an dnot very useful except debugging
        Function to load tdt neural data, local dup to check timing
        """
        from tdt import read_block
        tdt_list = os.listdir(self.paths['tdt_dir'])
        tdt_dat_list = []
        tdt_sess_durations = []
        evs_load = ['streams']
        stores_load = ['dup1','dup2']

        #probably overcomplicated but wnated ot make sure it does what I want
        #make sure tdt list sessions are ordered properly in time
        tdt_list_ordered = []
        sess_times = sorted(set([int(s.split('-')[-1]) for s in tdt_list]))
        for sess_time in sess_times:
            for sess in tdt_list:
                if sess_time == int(sess[-6:]):
                    tdt_list_ordered.append(sess)
        
        for tdt_session in tdt_list_ordered:
            fullpath = f"{self.paths['tdt_dir']}/{tdt_session}"
            session_data = read_block(fullpath,
                                    evtype=evs_load,
                                    store=stores_load,
                                    t1=trange[0]-0.0001,t2=trange[1]+0.0001)
            tdt_dat_list.append(session_data)
        return tdt_dat_list
    
    def generatePrettyBehDF(self):
        """
        Flattens beh data into something workable
        """
        df_columns = ['trial_ml2','stim_index','stim_name','fixation_success_binary']
        df = pd.DataFrame(columns = df_columns)
        stim_index = 0 
        for session_index, dat in enumerate(self.ml2_dat_list):
            trial_nums = [int(t.split('Trial')[1]) for t in dat.keys() if (t.startswith('Trial') and t != 'TrialRecord')]
            for trial in trial_nums:
                stim_list,stim_code_times,success_fail_list = self.getWhatStimEachPresentation(session_index,trial)
                for stim,time,success in zip(stim_list,stim_code_times,success_fail_list):
                    new_entry = pd.DataFrame([
                        {
                            'beh_session': int(session_index),
                            'block_num': int(dat[f'Trial{trial}']['Block']),
                            'trial_ml2':int(trial), #relative to session
                            'stim_index': int(stim_index), #unique index for each stim presentation, counts over sessions
                            'condition': int(dat[f'Trial{trial}']['Condition']),
                            'stim_name':stim,
                            'fixation_success_binary':success,
                            'ml2_time': time/100
                        }
                    ])
                    df = pd.concat([df,new_entry], ignore_index=True)
                    stim_index += 1
        return df
    
    def generatePrettyTdtDF(self):
        """
        Same as beh
        """
        
        df_columns = ['trial_ml2','stim_index','code_type','on','off'] #trial calculated by num 9's
        full_df = pd.DataFrame(columns = df_columns)
        
        stim_counter = 0
        sess_offset = 0
        for session_ind, session_dat in enumerate(self.tdt_dat_list):
            session_df = pd.DataFrame(columns = df_columns)
            beh_codes = session_dat.epocs.SMa1.data
            ons = session_dat.epocs.SMa1.onset
            offs = session_dat.epocs.SMa1.offset
            trial_counter = 0
            for code,on,off in zip(beh_codes,ons,offs):
                if code in range(102,132):
                    code_type = f'stim_in_cond_{int(code)%100-2}'
                    stim_index = np.nan
                elif code == 9:
                    trial_counter += 1
                    stim_index = np.nan
                elif code == 20:
                    stim_index = stim_counter
                    stim_counter += 1
                else:
                    stim_index = np.nan
                if code not in range (102,132):
                    code_type = EVENT_CODES_TO_EVENT_NAME[code]
                new_entry = pd.DataFrame([
                    {
                        'trial_ml2': trial_counter,
                        'stim_index': stim_index,
                        'code_type': code_type,
                        'on': on + sess_offset,
                        'off': off
                    }
                ])
                session_df = pd.concat([session_df,new_entry], ignore_index=True)

            session_df = self.assignEventMarkerstoPDTimes(session_ind, session_df)
            full_df = pd.concat([full_df,session_df], ignore_index=True)
            sess_offset += self._session_durations[session_ind]
        return full_df
    
    def mergeBehTdt(self):
        """
        Merges beh and tdt data
        """
        beh = self.prettyBeh
        tdt = self.prettyTdt
        tdt_stim_inds = tdt['stim_index'].dropna().to_numpy()
        beh_stim_inds = beh['stim_index'].to_numpy()
        assert np.all(tdt_stim_inds == beh_stim_inds)
        merge = pd.merge(beh,tdt, on='stim_index')
        merge = merge.drop('trial_ml2_y', axis = 1)
        merge = merge.rename(columns={'trial_ml2_x':'trial_ml2'}) #better trial keeping

        
        return merge

    
    ### PLOTTING
    def plotPSTH(self, channel_list, params, align_to = 'sample_on', window = (0.4,1.0),
                    bin_size = 0.001, group_by = None, overlay = False, figsize = (7,10)):
        """
        Simple PSTH plot function
        Args:
            params (dict): flexible params for what to plot formatted as dict like:
            {
            'column_name': [desired_values]
            }
            window (int or tuple): if tuple, (left window, right window)
            bin_size(numeric, time, s): size of bin for spikes
            group_by (str): groups by this var and plots average over this var,
                            if none will 'group_by' stim_index (no grouping/avging)
            overlay (bool): overlays groups
        """
        from scipy.ndimage import gaussian_filter1d
        if not isinstance(window, tuple):
            window = (window, window)
        assert len(window) == 2, 'what you doin'

        channel_list = np.atleast_1d(channel_list) #in case int
        

        fwhm = 0.020
        kernel_sigma = (fwhm/2.355)/bin_size

        fig_dict = {}

        df = filter_df(self.Dat, params)
        df = df.reset_index(drop=True)
        if align_to != 'sample_on':
            #can code easily, just use the prettyTdt data to get photodiode times for other events
            #was hard to merge other events into the merged data since aligned on sample_on
            assert False, 'not coded yet'
        stims_in_order = sorted(set(df['stim_index']))
        times_list = []
        for i in stims_in_order:
            row = df.loc[df['stim_index'] == i]
            t0 = row['photodiode_time'].values[0]
            times_list.append(t0)
        dict_spike_times, index_to_channel = self.extractSpikeTimes(channel_list,times_list,window)
        ax_count = 0
        for unit_index, list_spike_times in dict_spike_times.items():
            df[f'spike_counts_{unit_index}'] = None
            df[f'bin_times_{unit_index}'] = None
            for t0, stim_index, spike_times in zip(times_list, stims_in_order, list_spike_times):
                t_start = t0 - window[0]
                t_end = t0 + window[1]
                counts, bin_edges, bin_centers = self.spikeTimesBinCounts(
                    spike_times, t_start, t_end, bin_size)
                bin_centers = bin_centers - t0
                
                if group_by is None:
                    fig, ax = plt.subplots(1,figsize=figsize)
                    smooth_counts = gaussian_filter1d(counts, 
                                                        sigma=kernel_sigma, mode='nearest')
                    ax.plot(bin_centers,smooth_counts/bin_size)
                    ax.axvline(0, color='red')
                    ax.set_title(f'Channel: {index_to_channel[unit_index]}; Unit index: {unit_index}')
                    ax.set_ylabel('counts per bin')

                else:

                    mask = df['stim_index'] == stim_index
                    dfidx = df.index[mask][0]
                    df.at[dfidx, f'spike_counts_{unit_index}'] = counts
                    df.at[dfidx, f'bin_times_{unit_index}'] = bin_centers
        if group_by is not None:
            df = group_and_average(df, group_by, 'spike_counts')
            if df is None:
                #No spikes here
                return {}
            for col, data in df.items():
                if 'spike_counts' in col and 'mean' in col:
                    unit_index = int(col.split('_')[-2])
                    if overlay:
                        fig,ax = plt.subplots(0,figsize)
                    else:
                        fig,ax = plt.subplots(len(df),figsize=figsize)
                    ax = np.atleast_1d(ax)
                    ax_plot = 0
                    for index,group in zip(df.index,data):
                        mean = gaussian_filter1d(
                            group,
                            sigma=kernel_sigma,
                            mode='nearest'
                        )
                        sem = gaussian_filter1d(
                            df.at[index,f'spike_counts_{unit_index}_sem'],
                            sigma=kernel_sigma,
                            mode='nearest'
                        )

                        line, = ax[ax_plot].plot(
                            bin_centers,
                            mean,
                            label=row[group_by]
                        )
                        color = line.get_color()

                        ax[ax_plot].fill_between(
                            bin_centers,
                            mean - sem,
                            mean + sem,
                            color=color,
                            alpha=0.3
                        )

                        ax[ax_count].axvline(0, color='red')
                        ax[ax_count].set_title(f'Channel: {index_to_channel[unit_index]}; Unit index: {unit_index}')
                        ax[ax_count].set_ylabel('counts per bin')
                        if not overlay:
                            ax_plot += 1
                        fig_dict[unit_index] = fig
        return fig_dict

    def plotRaster(self, channel, params, align_to = 'sample_on', window = (0.2,0.8), figsize=(6,6)):
        """
        Simple raster plot function
        Args:
            params (dict): flexible params for what to plot formatted as dict like:
            {
            'column_name': [desired_values]
            }

            window (int or tuple): if tuple, (left window, right window)
        """
        if not isinstance(window, tuple):
            window = (window, window)
        assert len(window)==2, 'what you doin'
        fig_dict = {}

        df = filter_df(self.Dat, params)
        if align_to != 'sample_on':
            #can code easily, just use the filterdf on prettyTdt data to get photodiode times for other events
            #was hard to merge other events into the merged data since aligned on sample_on
            assert False, 'not coded yet'
        stims_in_order = sorted(set(df['stim_index']))
        times_list = []
        for i in stims_in_order:
            row = df.loc[df['stim_index'] == i]
            t0 = row['photodiode_time'].values[0]
            times_list.append(t0)
        dict_spike_times, _ = self.extractSpikeTimes([channel],times_list,window)
        for unit_index, spike_times in dict_spike_times.items():
            raster_vind = 0
            fig, ax = plt.subplots(1,figsize=figsize)
            for t0_this, stim_spikes in zip(times_list,spike_times):
                self._plot_raster_line(ax, stim_spikes, raster_vind, alignto_time=t0_this)
                raster_vind += 1
            ax.axvline(0, color = 'red',alpha = 0.4)
            ax.set_title(f'Channel: {channel}; Unit index: {unit_index}')
            fig_dict[unit_index] = fig
        return fig_dict

    def _plot_raster_line(self, ax, times, yval, color='k', alignto_time=None,
        linelengths = 0.95, alpha = 0.8, linewidths=None):
        """ plot a single raster line at times at row yval
        PARAMS:
        - alignto_time, in sec, this becomes new 0
        """
        if alignto_time:
            t = times-alignto_time
        else:
            t = times
        #     ax.plot(times, yval*np.ones(time.shape), '.', color=color, alpha=0.55)

        ax.eventplot([t], lineoffsets=yval, color=color, alpha=alpha, linelengths=linelengths,
                     linewidths=linewidths)
        
    
        
    ### HELPERS

    def spikeTimesBinCounts(self, spike_times, t_start, t_end, bin_size, plot=False, 
                            assert_all_times_within_bounds=True):
        """
        Bin spike times into counts per bin.

        Parameters
        ----------
        spike_times : array-like
            Array of spike times (in seconds or ms, as long as consistent with t_start, t_end, bin_size).
        t_start : float
            Start time of the window.
        t_end : float
            End time of the window.
        bin_size : float
            Bin size.

        Returns
        -------
        counts : np.ndarray
            Array of spike counts for each bin.
        bin_edges : np.ndarray
            The edges of the bins (length = len(counts) + 1).
        bin_centers : np.ndarray
            The centers of the bins (length = len(counts)).
        """
        
        if assert_all_times_within_bounds and len(spike_times)>0:
            assert np.min(spike_times) >= t_start, "Spike times should be >= t_start"
            assert np.max(spike_times) <= t_end, "Spike times should be <= t_end"

        bin_edges = np.arange(t_start, t_end+0.0001, bin_size)
        counts, _ = np.histogram(spike_times, bins=bin_edges)
        bin_centers = bin_edges[:-1] + bin_size / 2 # Center of each bin
        assert len(bin_centers)==len(counts)
        if not len(spike_times)==np.sum(counts):
            print(spike_times)
            print(counts)
            print(bin_centers)
            assert False

        if plot:
            if False:
                print("Spike counts:", counts)
                print("Bin edges:", bin_edges)

            # Plot the spike counts histogram:
            plt.figure(figsize=(10, 5))
            plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', edgecolor='black')
            plt.xlabel('Time (s)')
            plt.ylabel('Spike counts')
            plt.title('Binned Spike Counts')

            # Overlay the spikes
            plt.eventplot(spike_times, orientation='horizontal', color='red')

            plt.show()
        return counts, bin_edges, bin_centers

    def extractSpikeTimes(self, channel_list, times_list, window):
        """
        Extract spike times for all units in a list of channels centered at t in times list around a certain window
        Returns dict with unit index and spike times

        Drops empty rows
        """
        dict_spike_times = {}
        index_to_channel_dict = {}

        df_sel = self.spikeTimes[self.spikeTimes['chan_global'].astype(int).isin(channel_list)]

        for index,row in df_sel.iterrows():
            dict_spike_times[index] = []
            spike_times = row['times_sec_all']
            for t in times_list:
                mask = (spike_times >= t - window[0]) & (spike_times <= t + window[1])
                spike_times_mask = spike_times[mask]
                dict_spike_times[index].append(spike_times_mask)
                index_to_channel_dict[index] = row['chan_global']
        inds_no_spikes = []
        for ind, times in dict_spike_times.items():
            lengths = [len(l) for l in times if len(l) > 0]
            if len(lengths) == 0:
                inds_no_spikes.append(ind)
        for ind in inds_no_spikes:
            del dict_spike_times[ind]
        return dict_spike_times, index_to_channel_dict
    

    def getChannelNumOrRegionName(self, key, return_as = 'readable'):
        """maps channel number to region name and region name to channel numbers

        Args:
            key (int or string): int channel return region name, str region name gives channel nums
        """

        if isinstance(key,str):
            all_channels = []
            for region, channels in MAP_CHANNEL_TO_REGION.items():
                if key in region:
                    all_channels.extend(list(channels))
            if return_as == 'range':
                return range(min(all_channels),max(all_channels)+1)
            elif return_as == 'readable':
                return (min(all_channels),max(all_channels))
            elif return_as == 'list':
                return all_channels
            else:
                assert False, 'meow'

        elif isinstance(key, int):
            for region, channels in MAP_CHANNEL_TO_REGION.items():
                if key in channels:
                    return region
        else:
            assert False, 'invalid type supplied'

    
    def getEphysStream(self, channels, t, window, dup_mode = False):
        """
        Probbaly dont need this, also janky
        Pull out ephys data for given time range
        RSn2 = ch 1-256
        RSn3 = ch 257-512
        trange = tuple (start,end) seconds
        """

        trange = (t-window[0], t+window[1])

        if dup_mode: #dup debug mode
            dup = 'dup1'
            channels = [1]
            n = self.loadTdtNeuralDup(trange)
            n = n[0]
            rs2_streams = np.array(n.streams.dup1.data)
            rs2_channels = np.array(n.streams.dup1.channel)
            rs2_times = n.time_ranges
            rs2_fs = n.streams.dup1.fs

            rs3_streams = np.array(n.streams.dup2.data)
            rs3_channels = np.array(n.streams.dup2.channel)
            rs3_times = n.time_ranges
            rs3_fs = n.streams.dup2.fs
            shape_ind = 0
        else:
            n = self.loadTdtNeural(trange[0]-0.0001,trange[1]+0.0001)
            n = n[0]
            rs2_streams = np.array(n.streams.RSn2.data)
            rs2_channels = np.array(n.streams.RSn2.channels)
            rs2_times = n.time_ranges
            rs2_fs = n.streams.RSn2.fs

            rs3_streams = np.array(n.streams.RSn3.data)
            rs3_channels = np.array(n.streams.RSn3.channels)
            rs3_times = n.time_ranges
            rs3_fs = n.streams.RSn3.fs
            shape_ind = 1

        
        assert rs2_fs == rs3_fs, 'why neural fs different?'
        assert np.array_equal(rs2_times,rs3_times)

        times = np.linspace(rs2_times[0],rs2_times[1],rs2_streams.shape[shape_ind]) #max bc dup mode


        data_by_channel_dict = {}
        data_by_channel_dict['time'] = times
        if dup_mode:
            if dup == 'dup1':
                data_by_channel_dict[1] = rs2_streams
            elif dup == 'dup2':
                data_by_channel_dict[257] = rs3_streams
        else:
            for channel in channels:
                if channel < 257:
                    stream = rs2_streams[np.where(rs2_channels.astype(int) == channel)[0][0]]
                    data_by_channel_dict[channel] = stream
                elif channel >= 257:
                    stream = rs3_streams[np.where(rs3_channels.astype(int) == channel)[0][0]]
                    data_by_channel_dict[channel] = stream
        return data_by_channel_dict

    def assignEventMarkerstoPDTimes(self, session, neural):
        """
        called internally
        Adds column to pretty neural to assign photodiode times to relevant events
        """

        def nearest_value(arr, target, max_dist=None):
            """
            Return (index, value) of the array element closest to target.
            If max_dist is given, return np.nan if no value is within that distance (i.e. no pd trigger for event).
            """

            d = np.abs(arr - target)

            idx = np.argmin(d)
            val = arr[idx]
            dist = d[idx]

            # Check distance constraint
            if max_dist is not None and dist > max_dist:
                return np.nan, np.nan

            return idx, val
        
        session_time_offset = 0
        if session > 0:
            for i in range(session):
                session_time_offset += self._session_durations[i]        
        pd_times = self.getPhotodiodeThresholdCrossings(session) + session_time_offset

        max_dist = 0.15 #thresh for how far pd time can be from 'onset' time
        neural['photodiode_time'] = np.nan

        rew_times = self.tdt_dat_list[session].epocs.Rew_.onset + session_time_offset
        

        rew_inds_taken = []
        inds_taken = []
        for i, row in neural.iterrows():
            if row['code_type'] in ['trial_start']:
                #if no pd use avg of on/off signal time
                pd_time = np.mean([row['on'],row['off']])
            elif row['code_type'] in ['trial_end_ml2']:
                #end time only has onset
                pd_time = row['on']
            elif row['code_type'] == 'rew':
                idx,pd_time = nearest_value(rew_times, row['on'], max_dist = max_dist)
                assert idx not in rew_inds_taken, 'on time two rew? no good'
                rew_inds_taken.append(idx)
            else:
                idx, pd_time = nearest_value(pd_times, row['on'], max_dist = max_dist)
                if np.isnan(pd_time):
                    assert row['code_type'] == 'fix_cue' and neural.loc[i-1,'code_type'] == 'trial_start'
                    #sometimes first fix at trial start cue no pd time
                    #since very infrequent I assume happens when esc and then restart
                    pd_time = row['on']
                        
                if idx in inds_taken:
                    if row['on'] - neural.loc[i-1,'on'] > 0.15:
                        #some events happen to fast for pd to respond to both?
                        #like sample off and fix cue on for success trials
                        print('current row', row['code_type'],row['on'])
                        print('prev row',neural.loc[i-1,'code_type'], neural.loc[i-1, 'on'])
                        assert False
                inds_taken.append(idx)

            neural.loc[i, 'photodiode_time'] = pd_time

        return neural

            
    
    def getPhotodiodeThresholdCrossings(self, session):
        """
        Gets photodiode trigger times to align with neural events later
        """
        from scipy.signal import butter,filtfilt,freqz

        pd_analog = self.tdt_dat_list[session].streams.PhD2.data

        fs = self.tdt_dat_list[session].streams.PhD2.fs
        cutoff_freq = 60
        nyquist_freq = 0.5*fs
        normal_cutoff = cutoff_freq/nyquist_freq
        order = 4

        b,a = butter(order,normal_cutoff,btype='low',analog=False)

        pd_filt = filtfilt(b,a,pd_analog)

        inds = np.array(list(range(0,len(pd_filt))))
        times = inds/fs

        assert len(pd_filt) == len(times) #idk just in case

        min_val = np.percentile(pd_filt,25)
        max_val = np.percentile(pd_filt,75)
        thresh = (min_val+max_val)/2


        inds_crossings = np.where(
            ((pd_filt[:-1] <= thresh) & (pd_filt[1:] > thresh)) |    
            ((pd_filt[:-1] > thresh) & (pd_filt[1:] <= thresh))     
        )[0]

        crossing_times = times[inds_crossings]
        return crossing_times


    def getListStimNames(self, session, trial_ml2):
        """
        get list of stim file names for given trial.

        inputs:
        trial (int): monkeylogic (1 indexed) trial num
        """
        stim_list = []
        task_objects = self.ml2_dat_list[session][f'Trial{trial_ml2}']['TaskObject']['Attribute']
        for obj in task_objects:
            if obj[0] == 'pic':
                stim_path = obj[1]
                stim_name = stim_path.rsplit('\\')[-1].split('.')[0]
                stim_list.append(stim_name)
        return stim_list
        
    
    def getWhatStimEachPresentation(self, session, trial_ml2):
        """
        Get list of stims on each presentation.
        in:
        trial (int): 1 index trial
        ret:
        stim_each_present (list): stim name on each presentation
        stim_success_fail (list): True is fixated, False otherwise
        """

        dat_trial = self.ml2_dat_list[session][f'Trial{trial_ml2}']
        stim_list = self.getListStimNames(session,trial_ml2)
        beh_codes = np.array(dat_trial['BehavioralCodes']['CodeNumbers'])
        beh_code_times = dat_trial['BehavioralCodes']['CodeTimes']
        mask = (beh_codes >= 102) & (beh_codes <= 131)
        stim_code_times = beh_code_times[mask]
        stim_codes = [c%100 for c in beh_codes if 102 <= c <= 131]
        assert len(stim_code_times) == len(stim_codes), 'why diff?'
        stim_success_fail = [c != stim_codes[i+1] for i,c in enumerate(stim_codes) if i < len(stim_codes)-1]
        stim_each_present = [stim_list[c-2] for c in stim_codes]
        if len(stim_each_present) > 0:
            if stim_codes[-1] == 31:
                stim_success_fail.append(True) #last fix true bc trial not end otherwise
            else:
                stim_success_fail.append(False) #user abort probably, rare
        assert len(stim_success_fail) == len(stim_each_present), 'why diff lens'

        #if user recorded data exists make sure they match (not all days have, feature added later in recording)
        if 'TrialData' in self.ml2_dat_list[session]['TrialRecord']['User'].keys():
                stim_full_user = self.ml2_dat_list[session]['TrialRecord']['User']['TrialData'][trial_ml2-1]['sample_filename']
                if isinstance(stim_full_user, str):
                    stim_full_user = [stim_full_user]
                stim_list_user = [s.rsplit('\\')[-1].split('.')[0] for s in stim_full_user]
                sample_error_codes_user = np.atleast_1d(self.ml2_dat_list[session]['TrialRecord']['User']['TrialData'][trial_ml2-1]['sample_error_code'])
                stim_code_user = [c for c in sample_error_codes_user]
                stim_list_user_drop_nofix = [stim for i,stim in enumerate(stim_list_user) if stim_code_user[i] not in [4,5]]
                stim_bin_list = []
                for code in stim_code_user:
                    if code == 0:
                        stim_bin_list.append(True)
                    elif code == 3:
                        stim_bin_list.append(False)

                # print(stim_list_user_drop_nofix)
                # print(stim_each_present)
                # print(trial_ml2)
                assert np.all(stim_list_user_drop_nofix == stim_each_present)
                assert len(stim_bin_list) == len(stim_success_fail)
                if len(stim_codes) > 0:
                    if stim_codes[-1] != 31:
                        assert np.all(stim_bin_list[:-1] == stim_success_fail[:-1]) #no way to tell with other method if manual abort whether success or fail
                        stim_success_fail = stim_bin_list #defer to user data for more accuarte reading
                    else:
                        assert np.all(stim_bin_list == stim_success_fail)


        return stim_each_present, stim_code_times, stim_success_fail
    
    def AlignBehWithNeuralData(self, trial_ml2):
        """
        Finds neural on/off times aligned to this beh trial
        """
        neural_beh_codes = self.tdt_dat.epocs.SMa1.data
        neural_beh_codes_times = self.tdt_dat.epocs.SMa1.onset
        assert len(neural_beh_codes) == len(neural_beh_codes_times), 'why diff lengths'
        start_counter = 0
        start_time = None
        end_time = None
        found_start = False
        for i,code in neural_beh_codes:
            if code == 9:
                start_counter += 1
            if start_counter == trial_ml2:
                start_time == neural_beh_codes_times[i]
                found_start = True
            if code == 18 and found_start:
                end_time = neural_beh_codes_times[i]

        assert start_time is not None and end_time is not None

        return (start_time,end_time)
    
    def getSessionDurations(self):
        """
        function to load session durations for accurate time keeping
        session duration in tdt.info is not accurate to the actual recording length
            - will not align with kilosort times
        so use duration time from the log files in tdt code ()
            - will align with kilosort times
        """
        # (3) Duration of lenght of each RS4 recordings, saved in raw logs
        # i.e., raw(RS4) [THIS, in logs] --> concated..
        fs = 24414.0625 #would normally load and check that fs in data matches, but all data here is good
        durs_rs = {}
        rs_missed = []
        for rs in [2, 3]:
            # Collect durations across all sessions.
            durations = [] # list, length sessions/
            for sessnum, sessrec in enumerate(self._session_rec_names):
                # - Collect duration for this session
                logfile = f"RSn{rs}_log"
                path = f"{self.paths['tdt_dir']}/{sessrec}/{logfile}.txt"
                with open(path) as f:
                    lines = f.readlines()

                if len(lines)>2:
                    # Then is something like this. Keep first and last.
                    # ['recording started at sample: 2\n', 'gap detected. last saved sample: 51833413, new saved sample: 51833425\n', 'recording stopped at sample: 332994022\n']
                    lines = [lines[0], lines[-1]]

                try:
                    assert lines[0][:27] == 'recording started at sample'
                    assert lines[1][:20] == 'recording stopped at'
                except AssertionError as err:
                    print("==========")
                    print(lines)
                    print(len(lines))
                    for l in lines:
                        print(l)
                    print(rs, sessnum, sessrec, path)
                    assert False, "investigate..."

                ind1 = lines[0].find(": ")
                ind2 = lines[0].find("\n")
                samp_on = int(lines[0][ind1+2:ind2])
                assert samp_on < 25, "why is RS4 signal offset from onset of trial. This probably means misalignment vs. Data tank..."

                ind1 = lines[1].find(": ")
                ind2 = lines[1].find("\n")
                samp_off = int(lines[1][ind1+2:ind2])
                nsamp = samp_off - samp_on + 1
                # if dur is None:
                #     dur = nsamp/FS
                # else:
                #     assert dur - nsamp/FS < 0.005

                durations.append(nsamp/fs)
            durs_rs[rs] = durations
        assert len(durs_rs[2]) == len(durs_rs[3])
        avg_durs = []
        for rs2_dur, rs3_dur in zip(durs_rs[2],durs_rs[3]):
            assert np.isclose(rs2_dur,rs3_dur,atol=0.0015)
            avg_durs.append(np.mean([rs2_dur,rs3_dur]))
        return avg_durs
    
    def checkISI(self, min_events=12):
        """
        For each trial in trial_ml2, compute the correlation between
        successive time differences in ml2_time and photodiode_time.

        Args:
            min_events (int): minimum number of events in a trial to evaluate
        Returns:
            dict: trial_id -> correlation coefficient (np.nan if not computable)
        """

        for trial, df_t in self.Dat.groupby(['trial_ml2','beh_session']):
            # drop rows with missing times
            df_t = df_t[['ml2_time', 'photodiode_time']]

            #short trials not great corr
            if len(df_t) < min_events:
                corr = np.nan
                continue

            df_t = df_t.sort_values('ml2_time')

            d_ml2 = np.diff(df_t['ml2_time'].values)
            d_pd  = np.diff(df_t['photodiode_time'].values)


            corr = np.corrcoef(d_ml2, d_pd)[0, 1]

            if np.abs(corr) < 0.99:
                print('Low isi correlation for this trial')
                print(trial,':',corr)
                print('ISIs')
                for x,y in zip(d_ml2,d_pd):
                    print(x,y)
                assert False


        


    







MAP_CHANNEL_TO_REGION = {
    #Same for both animals
    # ---- RSn2 (1–256) ----
    "M1 (med)":       range(1, 33),
    "M1 (lat)":       range(33, 65),

    "PMv (lat)":      range(65, 97),
    "PMv (med)":      range(97, 129),

    "PMd (post)":     range(129, 161),
    "PMd (ant)":      range(161, 193),

    "dlPFC (post)":   range(193, 225),
    "dlPFC (ant)":    range(225, 257),

    # ---- RSn3 (shifted by +256 → 257–512) ----
    "vlPFC (post)":   range(257, 289),     # 1–32   → 257–288
    "vlPFC (ant)":    range(289, 321),     # 33–64  → 289–320

    "FP (post)":      range(321, 353),     # 65–96  → 321–352
    "FP (ant)":       range(353, 385),     # 97–128 → 353–384

    "SMA (post)":     range(385, 417),     # 129–160 → 385–416
    "SMA (ant)":      range(417, 449),     # 161–192 → 417–448

    "preSMA (post)":  range(449, 481),     # 193–224 → 449–480
    "preSMA (ant)":   range(481, 513),     # 225–256 → 481–512
}

REGIONS = ['M1','PMv','PMd','dlPFC','vlPFC','FP','SMA','preSMA']

    


        







