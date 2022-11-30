1. Clustering additional
    1. <a href="https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
">beginners-guide-hierarchical-clustering</a>
    2. <a href="https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/
">4-types-of-distance-metrics-in-machine-learning</a>
2. Guassian Clustering
    1. <a href="https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/
">gaussian-mixture-models-clustering</a>
    2. <a href="https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
">how-to-determine-the-optimal-k-for-k-means-708505d204eb</a>
    3. <a href="https://towardsdatascience.com/10-tips-for-choosing-the-optimal-number-of-clusters-277e93d72d92
">10-tips-for-choosing-the-optimal-number-of-clusters-277e93d72d92</a>
    4. <a href="https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right/
">getting-clustering-right</a>
3. Gettting Clustering Right
    1. <a href="https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right/
">getting-clustering-right</a>
    2. <a href="https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right-part-ii/
">getting-clustering-right-part-ii</a>
6. hierarchical Clustering
    1. <a href="https://www.saedsayad.com/clustering_hierarchical.htm
">clustering_hierarchical</a>
9. Clusters not making sense
    1. <a href="https://towardsdatascience.com/when-clustering-doesnt-make-sense-c6ed9a89e9e6	
">LINK</a>
    2. <a href="https://stats.stackexchange.com/questions/143699/clustering-not-producing-even-clusters	
">LINK</a>
11. Clustering messy data Alternatives to Kmeans
    1. <a href="https://www.datascience.com/blog/k-means-alternatives
">k-means-alternatives</a>
13. expectation maximization Distribution models clustering
    1. <a href="https://medium.com/technology-nineleaps/expectation-maximization-4bb203841757
">expectation-maximization-4bb203841757</a>
14. Difference between K Means and heirarchical Clustering
    1. <a href="https://towardsdatascience.com/unsupervised-learning-k-means-vs-hierarchical-clustering-5fe2da7c9554
">unsupervised-learning-k-means-vs-hierarchical-clustering-5fe2da7c9554</a>
15. Combining hierarchical clustering and k means
 
KEY POINTS
1. Why Hierarchial Clustering takes more time than Kmeans Clustering:The only problem with the technique is that it is able to only handle small number of data-points and is very time consuming. This is because it tries to calculate the distance between all possible combination and then takes one decision to combine two groups/individual data-point.
2. When we compare the two techniques, we find that the Hierarchical Clustering starts with individual data-points and sequentially club them to find the final cluster whereas  k-means Clustering  starts from some initial cluster and then tries to reassign data-points to k clusters to minimize the total penalty term. Hence for large number of data-points,   k-means uses far lesser iterations then Hierarchical Clustering.
3. Decide Optimal number of K:Overall, you can choose number of clusters in in two different paths.
4. knowledge driven: you should have some ideas how many cluster do you need from business point of view. For example, you are clustering customers, you should ask yourself, after getting these customers, what should I do next? May be you will have different treatment for different clusters? (e.g., advertising by email or phone). Then how many possible treatments are you planning? In this example, you select say 100 clusters will not make too much sense.
5. Data driven: more number of clusters is over-fitting and less number of clusters is under-fitting. You can always split data in half and run cross validation to see how many number of clusters are good. Note, in clustering you still have the loss function, similar to supervised setting.
6. Finally, you should always combine knowledge driven and data driven together in real world.
7. Answer1:
8. Whereas k-means tries to optimize a global goal (variance of the clusters) and achieves a local optimum, agglomerative hierarchical clustering aims at finding the best step at each cluster fusion (greedy algorithm) which is done exactly but resulting in a potentially suboptimal solution.
9. One should use hierarchical clustering when underlying data has a hierarchical structure (like the correlations in financial markets) and you want to recover the hierarchy. You can still apply k-means to do that, but you may end up with partitions (from the coarsest one (all data points in a cluster) to the finest one (each data point is a cluster)) that are not nested and thus not a proper hierarchy.
10. If you want to dig into finer properties of clustering, you may not want to oppose flat clustering such as k-means to hierarchical clustering such as the Single, Average, Complete Linkages. For instance, all these clustering are space-conserving, i.e. when you are building clusters you do not distort the space, whereas a hierarchical clustering such as Ward is not space-conserving, i.e. at each merging step it will distort the metric space.
11. To conclude, the drawbacks of the hierarchical clustering algorithms can be very different from one to another. Some may share similar properties to k-means: Ward aims at optimizing variance, but Single Linkage not. But they can also have different properties: Ward is space-dilating, whereas Single Linkage is space-conserving like k-means.
12. Answer 2:
13. Scalability
14. k means is the clear winner here. O(n·k·d·i) is much better than the O(n3d) (in a few cases O(n2d)) scalability of hierarchical clustering because usually both k and i and d are small (unfortunately, i tends to grow with n, so O(n) does not usually hold). Also, memory consumption is linear, as opposed to quadratic (usually, linear special cases exist).
15. Flexibility
16. k-means is extremely limited in applicability. It is essentially limited to Euclidean distances (including Euclidean in kernel spaces, and Bregman divergences, but these are quite exotic and nobody actually uses them with k-means). Even worse, k-means only works on numerical data (which should actually be continuous and dense to be a good fit for k-means).
17. Hierarchical clustering is the clear winner here. It does not even require a distance - any measure can be used, including similarity functions simply by preferring high values to low values. Categorial data? sure just use e.g. Jaccard. Strings? Try Levenshtein distance. Time series? sure. Mixed type data? Gower distance. There are millions of data sets where you can use hierarchical clustering, but where you cannot use k-means.
18. Model
19. No winner here. k-means scores high because it yields a great data reduction. Centroids are easy to understand and use. Hierarchical clustering, on the other hand, produces a dendrogram. A dendrogram can also be very very useful in understanding your data set.
20. Hierarchical clustering may give locally optimise clusters as it is based on greedy approach but K means gives globally optimised clusters. I have also experienced that explanation of hierarchical clustering is relatively easy to business people compare to K means. – Arpit Sisodia Sep 9 '17 at 10:13
21. EDIT thanks to ttnphns: One feature that hierarchical clustering shares with many other algorithms is the need to choose a distance measure. This is often highly dependent on the particular application and goals. This might be seen as an additional complication (another parameter to select...), but also as an asset - more possibilities. On the contrary, classical K-means algorithm specifically uses Euclidean distance.
22. http://www.sthda.com/english/wiki/print.php?id=244
