import numpy as np
import sys
sys.path.append(r'C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroStatistics\ML\GammaRayBursts\Utilities')
import utils

url = 'https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt'
raw, names = utils.read_data(url)

results = {
    'factor': [],
    'n_clusters': [],
    'score': []
}

T90, fluence = zip(*[(float(a), float(c)) for a, c in zip(raw[6], raw[9]) 
                                           if a != '-999' and float(a)!=0 
                                           and c != '-999' and float(c) != 0
                                           ])

DD = np.vstack([np.log10(np.array(fluence)), np.log10(np.array(T90))])

for factor in np.arange(10, 1, -2, dtype=int):
    
    print('Processing factor number...', factor)
    
    data = DD[:, ::factor].T
    print(data[0].shape)
    
    n_clusters, score = utils.Gauss_peaks(data, 2, 10, False)
    labels, model = utils.GM_fit(data, n_clusters, False)

    results['factor'].append(factor)
    results['n_clusters'].append(n_clusters)
    results['score'].append(score)
    
with open("results.txt","w") as f:
    for key, value in results.items():
        f.write(f"{key}: {value}\n")