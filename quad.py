# Functions for extracting relevant data from behavioral files
import pandas as pd

EVENT_CODES_TO_EVENT_NAME = {
    9: 'trial_start',
    18: 'trial_end',
    10: 'fix_cue',
    20: 'sample',
    21: 'sample_off',
    49: 'timeout',
    50: 'rew', 
    51: 'trialend'
}

class Quads:
    def __init__(self, dat, conditions, neural):
        self.dat = dat              # dict with all trial info (convert bhv2 to mat)
        self.conditions = conditions # conditions file loaded from text as pd df
        self.neural = neural


        self.prettyBeh = self.generatePrettyBehDF()
        self.prettyNeural = self.generatePrettyNeuralDF()

    
    def generatePrettyBehDF(self):
        """
        Flattens beh data into something workable
        """
        trial_nums = [int(t.split('Trial')[1]) for t in self.dat.keys() if (t.startswith('Trial') and t != 'TrialRecord')]
        df_columns = ['trial_ml2','stim_index','stim_presented','fixation_success_binary']
        df = pd.DataFrame(columns = df_columns)
        stim_index = 0 #unique index for each stim in this session
        for trial in trial_nums:
            stim,success_fail = self.getWhatStimEachPresentation(trial)
            for stim,success in zip(stim,success_fail):
                new_entry = pd.DataFrame([
                    {
                        'trial_ml2':trial,
                        'stim_index': stim_index,
                        'stim_presented':stim,
                        'fixation_success_binary':success
                    }
                ])
                df = pd.concat([df,new_entry], ignore_index=True)
                stim_index += 1
        return df
    def generatePrettyNeuralDF(self):
        """
        Same as beh
        """
        beh_codes = self.neural.epocs.SMa1.data
        ons = self.neural.epocs.SMa1.onset
        offs = self.neural.epocs.SMa1.offset
        df_columns = ['stim_index','on','off']
        for code,on,off in zip(beh_codes,ons,offs):
            print('meow')

    def getListStimNames(self, trial_ml2):
        """
        get list of stim file names for given trial.

        inputs:
        trial (int): monkeylogic (1 indexed) trial num
        """

        dat_trial = self.dat[f'Trial{trial_ml2}']
        condition_num = dat_trial['Condition']
        conds = self.conditions
        stim_list = []
        for i in range (2,32):
            stim_full = conds.loc[conds['Condition'] == condition_num, f'TaskObject#{i}'].iloc[0]
            stim = stim_full.split('/')[1].split(')')[0]
            stim_list.append(stim)
        return stim_list
    
    def getWhatStimEachPresentation(self, trial_ml2):
        """
        Get list of stims on each presentation.
        in:
        trial (int): 1 index trial
        ret:
        stim_each_present (list): stim name on each presentation
        stim_success_fail (list): True is fixated, False otherwise
        """

        dat_trial = self.dat[f'Trial{trial_ml2}']
        stim_list = self.getListStimNames(trial_ml2)
        beh_codes = dat_trial['BehavioralCodes']['CodeNumbers']
        stim_codes = [c%100 for c in beh_codes if 102 <= c <= 131]
        stim_success_fail = [c != stim_codes[i+1] for i,c in enumerate(stim_codes) if i < len(stim_codes)-1]
        stim_success_fail.append(True) #last fix true bc trial not end otherwise
        stim_each_present = [stim_list[c-2] for c in stim_codes]
        assert len(stim_success_fail) == len(stim_each_present), 'why diff lens'

        return stim_each_present, stim_success_fail
    def AlignBehWithNeuralData(self, trial_ml2):
        """
        Finds neural on/off times aligned to this beh trial
        """
        neural_beh_codes = self.neural.epocs.SMa1.data
        neural_beh_codes_times = self.neural.epocs.SMa1.onset
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
    


        







