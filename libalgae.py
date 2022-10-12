"""functions library for algae"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def load_algae():
    df = pd.read_csv("./algae-1.csv", sep=";").drop(columns=["Unnamed: 0"])

    def str2float(string):
        return float(string.replace(",", "."))

    df.wind = df.wind.apply(str2float)
    df.discharge = df.discharge.apply(str2float)
    df.rainfall = df.rainfall.apply(str2float)
    df.temperature = df.temperature.apply(str2float)
    return df


class Distrib:
    def __init__(self, dist, params=None):
        self.dist = dist
        self.params = params


@dataclass
class AlgaeClassifier:

    distrib_dict: dict

    def fit(self, X_train, y_train):
        for name, dists in self.distrib_dict.items():
            for label, dist in enumerate(dists):
                dist.params = dist.dist.fit(
                    X_train.loc[y_train == label, name])

    def plot_fitted_distrib(self, ax, name, label):
        distp = self.distrib_dict[name][label]
        xmin, xmax = ax.get_xlim()
        xax = np.linspace(xmin, xmax, 100)
        ax.plot(xax, distp.dist.pdf(xax, *distp.params), label=distp.dist.name)

    def predict(self, X):
        llh = [np.zeros(X.shape[0]), np.zeros(X.shape[0])]
        for name, distps in self.distrib_dict.items():
            for label, distp in enumerate(distps):
                llh[label] = llh[label] + np.log(
                    distp.dist.pdf(X[name], *distp.params)+1e-20)
        return llh[1] > llh[0]

    def get_scores(self, X, y_true):
        y_pred = self.predict(X)
        return (accuracy_score(y_true, y_pred),
                f1_score(y_true, y_pred))


if __name__ == "__main__":

    df_algae = load_algae()

    features = df_algae.drop(columns=["rainfall", "risk.label"])
    target = df_algae["risk.label"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.15)

    distrib_dict = {
        "wind": (Distrib(stats.norm), Distrib(stats.norm)),
        #"temperature": (Distrib(stats.norm), Distrib(stats.norm)),
        "discharge": (Distrib(stats.norm), Distrib(stats.weibull_min)),
    }

    clf = AlgaeClassifier(distrib_dict)
    clf.fit(X_train, y_train)

    print(clf.get_scores(X_train, y_train))
    results = pd.DataFrame(
        data=[clf.get_scores(X_train, y_train),
              clf.get_scores(X_test, y_test)],
        index=["train", "test"],
        columns=["accuracy", "f1"]
    )

    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print(results)    
 
