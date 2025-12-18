from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
from quad import loadQuad, Quads #custom class for quad data
import tdt
import matplotlib.pyplot as plt

animal_date_dict = {
    'Pancho': ['251118','251119','251120'],
    'Diego': ['251113','251114','251118']
}

for animal, dates in animal_date_dict.items():
    for date in dates:
        print(f'Doing {animal} {date}...')
        basedir = f'/home/danhan/code/data/quad_data/plots/{animal}/{date}'
        quad = loadQuad(animal, date)

        from quad import REGIONS
        for r in REGIONS:
            channels = quad.getChannelNumOrRegionName(r, return_as = 'list')

            params = {
            'fixation_success_binary': [True],
            }

            savedir = f'{basedir}/rasters_each_site/{r}'

            if not os.path.exists(savedir):
                os.makedirs(savedir)
            print('Plotting rasters for each unit...')
            for channel in channels:
                fig_dict = quad.plotRaster(channel,params, window = (0.4,1.0))
                for index, fig in fig_dict.items():
                    fig.savefig(f'{savedir}/{r}_{channel}_{index}.png')
                    plt.close(fig)
                plt.close('all')
            print('... Done')
            print('Plotting PSTH for each site accross whole day...')
            for channel in channels:                                                    
                fig_dict = quad.plotPSTH([channel], params, group_by='fixation_success_binary')

                savedir = f'{basedir}/psth_whole_day/{r}'
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                for index,fig in fig_dict.items():
                    fig.savefig(f'{savedir}/{r}_{channel}_{index}.png')
                    plt.close(fig)
                plt.close('all')
            print('... Done')
