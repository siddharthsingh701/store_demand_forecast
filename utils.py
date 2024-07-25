import matplotlib.pyplot as plt 

def plot_ts(df,store_id,item_id):
    df1= df.loc[(df['store']==store_id)&(df['item']==item_id)]
    plt.figure(figsize=(12,4))
    df1.set_index('date')['sales'].plot()
    plt.show()

def plot_ts(df,unique_id):
    df1= df.loc[df['unique_id']==unique_id]
    plt.figure(figsize=(12,4))
    df1.set_index('ds')['y'].plot()
    plt.show()