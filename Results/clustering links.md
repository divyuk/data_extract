1. Customer Segmentation linked in Very good article
    1. <a href="https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
">comprehensive-guide-k-means-clustering</a>
    2. <a href="https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
">beginners-guide-hierarchical-clustering</a>
    3. <a href="https://www.analyticsvidhya.com/blog/2020/10/a-simple-explanation-of-k-means-clustering/
">a-simple-explanation-of-k-means-clustering</a>
    4. <a href="https://www.insanegrowth.com/customer-segmentation/
">customer-segmentation</a>
2. Clustering Evalaution Metrics
    1. <a href="https://gdcoder.com/silhouette-analysis-vs-elbow-method-vs-davies-bouldin-index-selecting-the-optimal-number-of-clusters-for-kmeans-clustering/
">silhouette-analysis-vs-elbow-method-vs-davies-bouldin-index-selecting-the-optimal-number-of-clusters-for-kmeans-clustering</a>
    2. <a href="https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/
">dunn-index-and-db-index-cluster-validity-indices-set-1</a>
    3. <a href="https://scikit-learn.org/stable/modules/clustering.html
">clustering</a>
    4. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
">sklearn</a>
    5. <a href="https://medium.com/@ODSC/assessment-metrics-for-clustering-algorithms-4a902e00d92d
">assessment-metrics-for-clustering-algorithms-4a902e00d92d</a>
    6. <a href="https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
">how-to-determine-the-optimal-k-for-k-means-708505d204eb</a>
    7. <a href="https://towardsdatascience.com/10-tips-for-choosing-the-optimal-number-of-clusters-277e93d72d92
">10-tips-for-choosing-the-optimal-number-of-clusters-277e93d72d92</a>
    8. <a href="https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right/
">getting-clustering-right</a>
3. Gettting Clustering Right
    1. <a href="https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right/
">getting-clustering-right</a>
    2. <a href="https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right-part-ii/
">getting-clustering-right-part-ii</a>
5. DBSCAN
    1. <a href="https://www.geeksforgeeks.org/difference-between-k-means-and-dbscan-clustering/
">difference-between-k-means-and-dbscan-clustering</a>
    2. <a href="https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
">how-dbscan-clustering-works</a>
    3. <a href="https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/
">dbscan-clustering-in-ml-density-based-clustering</a>
8. Clusters not making sense
    1. <a href="https://towardsdatascience.com/when-clustering-doesnt-make-sense-c6ed9a89e9e6	
">LINK</a>
10. Clustering messy data Alternatives to Kmeans
    1. <a href="https://www.datascience.com/blog/k-means-alternatives
">k-means-alternatives</a>
12. expectation maximization Distribution models clustering
    1. <a href="https://medium.com/technology-nineleaps/expectation-maximization-4bb203841757
">expectation-maximization-4bb203841757</a>
 
KEY POINTS
1. Why Hierarchial Clustering takes more time than Kmeans Clustering:The only problem with the technique is that it is able to only handle small number of data-points and is very time consuming. This is because it tries to calculate the distance between all possible combination and then takes one decision to combine two groups/individual data-point.
2. When we compare the two techniques, we find that the Hierarchical Clustering starts with individual data-points and sequentially club them to find the final cluster whereas  k-means Clustering  starts from some initial cluster and then tries to reassign data-points to k clusters to minimize the total penalty term. Hence for large number of data-points,k-means uses far lesser iterations then Hierarchical Clustering.
3. Decide Optimal number of K:Overall, you can choose number of clusters in in two different paths.
4. Knowledge driven: you should have some ideas how many cluster do you need from business point of view. For example, you are clustering customers, you should ask yourself, after getting these customers, what should I do next? May be you will have different treatment for different clusters? (e.g., advertising by email or phone). Then how many possible treatments are you planning? In this example, you select say 100 clusters will not make too much sense.
5. Data driven: more number of clusters is over-fitting and less number of clusters is under-fitting. You can always split data in half and run cross validation to see how many number of clusters are good. Note, in clustering you still have the loss function, similar to supervised setting.
6. Finally, you should always combine knowledge driven and data driven together in real world.
7. Difference between K Means and heirarchical Clustering
8. Whereas k-means tries to optimize a global goal (variance of the clusters) and achieves a local optimum, agglomerative hierarchical clustering aims at finding the best step at each cluster fusion (greedy algorithm) which is done exactly but resulting in a potentially suboptimal solution. One should use hierarchical clustering when underlying data has a hierarchical structure (like the correlations in financial markets) and you want to recover the hierarchy. You can still apply k-means to do that, but you may end up with partitions (from the coarsest one (all data points in a cluster) to the finest one (each data point is a cluster)) that are not nested and thus not a proper hierarchy. If you want to dig into finer properties of clustering, you may not want to oppose flat clustering such as k-means to hierarchical clustering such as the Single, Average, Complete Linkages. For instance, all these clustering are space-conserving, i.e. when you are building clusters you do not distort the space, whereas a hierarchical clustering such as Ward is not space-conserving, i.e. at each merging step it will distort the metric space.
9. To conclude, the drawbacks of the hierarchical clustering algorithms can be very different from one to another. Some may share similar properties to k-means: Ward aims at optimizing variance, but Single Linkage not. But they can also have different properties: Ward is space-dilating, whereas Single Linkage is space-conserving like k-means.
10. customer.care@lalpathlabs.com
