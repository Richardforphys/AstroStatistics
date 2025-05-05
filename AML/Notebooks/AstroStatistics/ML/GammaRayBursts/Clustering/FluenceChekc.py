import numpy as np
import sys
sys.path.append(r'C:\Users\ricca\Documents\Unimib-Code\AstroStatistics\AML\Notebooks\AstroStatistics\ML\GammaRayBursts\Utilities')
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
scaler = StandardScaler()

for factor in np.arange(10, 1, -2, dtype=int):
    
    print('Processing factor number...', factor)
    
    DD_scaled = scaler.fit_transform(DD)
    data = DD_scaled[:, ::factor].T
    X_train , X_test = train_test_split(data, test_size=0.1, random_state=42)
    
    n_clusters, score = utils.Gauss_peaks(X_train, 2, 10, False)
    _ , model = utils.GM_fit(X_train, n_clusters, False)
    labels = model.predict(X_test)

    results['factor'].append(factor)
    results['n_clusters'].append(n_clusters)
    results['score'].append(score)
    
with open("results.txt","w") as f:
    for key, value in results.items():
        f.write(f"{key}: {value}\n")