import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx

def get_data(data_path):
    df = pd.read_excel(data_path)
    return df

def get_lat_lon():
    address = 'Hirakata, Osaka, Japan'
    geolocator = Nominatim(user_agent="foursquare_agent")
    location = geolocator.geocode(address)
    lat = location.latitude
    lon = location.longitude
    print('The geograpical coordinate of {} are {}, {}.'.format(address, lat, lon))
    return lat, lon

def cluster_map(df, X, Y, n_clusters):
    cust_array = np.array([df[X].tolist(), df[Y].tolist()]).T
    pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(cust_array)
    return pred

# Create each graph based on the map after clustering
def make_graph(df, cluster_id) -> nx.Graph:
    df_cluster = df[df['cluster_id'] == cluster_id]
    G = nx.Graph()
    for i in range(len(df_cluster)):
        G.add_node(df_cluster.iloc[i]['住所'])
    for i in range(len(df_cluster)):
        for j in range(i + 1, len(df_cluster)):
            G.add_edge(df_cluster.iloc[i]['住所'], df_cluster.iloc[j]['住所'])
    return G
        

def main():
    df = get_data('data/garbage_place.xlsx')
    # print(data.head())
    cluster = cluster_map(df, 'X', 'Y', 44)
    df['cluster_id'] = cluster
    # print(df.head())
    for i in range(44):
        G = make_graph(df, i)
        nx.draw_networkx(G, with_labels=True)
    
if __name__ == '__main__':
    main()
    