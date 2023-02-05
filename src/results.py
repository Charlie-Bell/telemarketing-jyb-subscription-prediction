import matplotlib.pyplot as plt
import pandas as pd

def feature_importance(data, model):
    feature_importances = pd.DataFrame({
        'Feature': data.columns,
        'Importance': model.feature_importances_
    })
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)[:20]

    plt.bar(x=feature_importances['Feature'], height=feature_importances['Importance'], color='#08AE8B')
    plt.title('Feature importances obtained from impurity', size=20)
    plt.xticks(rotation='vertical')
    plt.show()

def plot_kde(feature, data, subscribed):
    for i in [False, True]:
        if i:
            data[subscribed['subscribed']==i][feature].rename(str('Subscribed')).plot.kde(legend=True, title=feature+' probability density')
        else:
            data[subscribed['subscribed']==i][feature].rename(str('Not subscribed')).plot.kde(legend=True, title=feature+' probability density')


def plot_roi(gains, counts):
    roi = pd.DataFrame({
        'Result': ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
        'Gain per Customer': gains,
        'Count': counts,
        'Gain ($)':[gains[0]*counts[0], gains[1]*counts[1], gains[2]*counts[2], gains[3]*counts[3]],
        },
    ).set_index('Result')
    print(pd.pivot_table(roi, values=['Gain per Customer', 'Count', 'Gain ($)'], index='Result', aggfunc='sum', margins=True, margins_name='Total'))
