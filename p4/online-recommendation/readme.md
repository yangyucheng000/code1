# Dynamic Heterogeneous Attributed Network Embedding

Information networks generally exhibit three characteristics, namely dynamicity, heterogeneity, and node attribute diversity. However, most existing network embedding approaches only consider two of the three when embedding each node into low-dimensional space. Adding to such an existing approach a technique of processing the remaining characteristic can easily cause incompatibility. One solution to process the three characteristics together is to treat the dynamic heterogeneous attributed network (DHAN) as a temporal sequence of heterogeneous attributed network (HAN) snapshots. For example, existing graph convolutional networks (GCNs)-based DHAN embedding approaches embed the HAN snapshots to get static representations offline, and then dynamically capture temporal dependencies between adjacent snapshots online to maintain fresh representations of the DHAN. However, those approaches encounter the convergence problem when stacking multiple convolutional layers to capture more topological information. Some other existing approaches dynamically update the representations of HAN snapshots online, neglecting the efficiency requirement of online scenarios and the temporal dependencies between snapshots. To address the two issues, we propose a new framework called Dynamic Heterogeneous Attributed Network Embedding (DHANE), consisting of a static model MGAT and a dynamic model NICE. MGAT captures more topological information while maintaining GCN convergence by performing metagraph-based attention in each convolutional layer. NICE preserves network freshness while reducing the computational load of the update by only examining network changes and updating their embedding representations. 

### Environment:

In **`requirements.txt`**

Deprecation notice
This is an example experiment not the full one because file size exceeds capacity limit. To run the code

### Usage:

```
python main.py
```

Arguments can be modified in main.py.



