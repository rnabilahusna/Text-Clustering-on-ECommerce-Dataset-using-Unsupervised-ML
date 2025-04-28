# Text Clustering on E-Commerce Dataset using Unsupervised Machine Learning

  The project will show how to perform clustering on Product Description and Review clustering using Machine Learning algorithms, specifically K-Means, Hierarchical Clustering, and DBSCAN to find the similarities in the clusters in which the  product descriptions and reviews have been grouped. 

  A retail industry generates a large amount of textual data from various sources including product description reviews. The dataset used will initially consist of only two columns which are the “category” which is the product category, and “description” that is the product description and its reviews by users. Originally, there are four types of product categories from the dataset that were described and reviewed, of which the products are from the “Household”, “Books”, “Electronics”, and “Clothing & Accessories” category.

  The purpose of this project is to categorize after identifying distinct product categories within the product descriptions and reviews text data. Mainly, this data consists of one column that is named as “label” which consists of four main categories; Household, Books, Electronics, Clothing & Accessories. However, after we discover the text data that has been clustered by three different algorithms, there might be new product categories that are not in the existing label. So, we want to discover if there are any new product categories.

# 🔎 Finding and Discussion

## 1.1 Comparison of Clustering Results

This section compares the results of three different clustering algorithms used throughout this project. By comparing the results, we can identify the most compelling technique among them.

## 1.2 K-Means

### 1.2.1 Default Number of Clusters (k = 5)

Initially, we used **k = 5** as the default number of clusters for K-Means clustering.  
Table 1.1 below shows the top keywords extracted from the five clusters.

**Table 1.1: Top Keywords for Five Clusters**

| Cluster | Top Keywords |
|:-------:|:------------:|
| 0 | 'cotton', 'woman', 'men', 'wear', 'fabric', 'fit', 'boy', 'girl', 'made', 'pack' |
| 1 | 'product', 'cm', 'size', 'set', 'home', 'inch', 'black', 'color', 'steel', 'easy' |
| 2 | 'book', 'author', 'time', 'life', 'new', 'review', 'university', 'world', 'one', 'story' |
| 3 | 'steel', 'set', 'stainless', 'cm', 'home', 'table', 'product', 'color', 'kitchen', 'size' |
| 4 | 'usb', 'bluetooth', 'cable', 'speaker', 'camera', 'device', 'audio', 'mm', 'power', 'wireless' |

From the table above, the top keywords from each cluster help us identify each cluster's characteristics:

- **Cluster 0**: Related to fashion.
- **Cluster 2**: Related to books and literature.
- **Cluster 4**: Dedicated to electronic devices and components.

Upon closer examination, **Cluster 1** and **Cluster 3** both describe household products, indicating some overlap. Both clusters feature words like "steel", "home", and "product."  
Given the similarity, it makes sense to combine them into a single cluster, reducing the number of clusters to **4**.  
This consolidation will provide a more accurate categorization of the products.

---

The **silhouette score** is a measure of how similar an object is to its own cluster compared to other clusters.  
The score ranges from -1 to 1, where a higher score indicates better-defined clusters.

Below are the silhouette scores using different distance metrics for five clusters:

- **a)** Silhouette Score (Euclidean, K-Means): **0.46**
- **b)** Silhouette Score (Cosine, K-Means): **0.64**
- **c)** Silhouette Score (Manhattan, K-Means): **0.45**

The highest silhouette score (**0.64**) was achieved using the **cosine distance metric**, suggesting it is the most suitable for the given text data.

**Figure 1.1: Visualization of Product Category Clusters (k = 5)**

The default cluster visualization (k = 5) shows scattered points with unclear boundaries, especially between Cluster 1 and Cluster 3, indicating poor clustering.

---

To identify the best number of clusters, two analyses were conducted:

1. **Elbow Method Analysis**
2. **Silhouette Scores Analysis**

The following table shows the **Within-Cluster Sum of Squares (WCSS)** and **Silhouette Scores** obtained:

**Table 1.2: WCSS and Silhouette Scores for Each Cluster**

| Cluster (k) | WCSS     | Silhouette Score |
|:-----------:|:--------:|:----------------:|
| 2           | 10307.53 | 0.533             |
| 3           | 6315.92  | 0.632             |
| 4           | 3383.66  | 0.738             |
| 5           | 2920.79  | 0.641             |
| 6           | 2488.46  | 0.642             |
| 7           | 2161.08  | 0.625             |
| 8           | 1945.82  | 0.561             |
| 9           | 1815.91  | 0.547             |

---

**Figure 1.2: The Elbow Method for Product Categories**

Based on the elbow method, the rate of WCSS decrease slows noticeably after **k = 4**, indicating diminishing returns in clustering effectiveness beyond this point.

**Figure 1.3: Number of Clusters vs Silhouette Score**

The silhouette score peaks at **k = 4**, suggesting that this configuration yields the most distinct and well-defined clusters of product categories.  
Choosing **k = 4** provides a balance between granularity and clustering quality.

Thus, for the final K-Means model, **k = 4** is selected to obtain the final cluster labels and centers.

# 1.2.2 Optimal Number of Cluster (k = 4)

After identifying the best number of clusters during hyperparameter tuning, the optimal number of clusters identified will be 4. The optimal cluster will be used to re-run K-Means, evaluate clustering quality by calculating the silhouette score, and lastly visualize the final clusters and cluster centers.

To interpret the clusters, below shows the identification of the top keywords for each cluster for interpretability.

**Table 1.3: Top Keywords for Four Clusters**

| Cluster | Top Keywords |
|:-------:|:-------------|
| 0 | 'book', 'author', 'time', 'life', 'new', 'review', 'world', 'university', 'one', 'story' |
| 1 | 'steel', 'set', 'cm', 'stainless', 'home', 'product', 'table', 'size', 'color', 'easy' |
| 2 | 'usb', 'bluetooth', 'cable', 'speaker', 'camera', 'device', 'mm', 'audio', 'power', 'black' |
| 3 | 'cotton', 'woman', 'men', 'wear', 'fabric', 'fit', 'boy', 'girl', 'made', 'pack' |

Given the analysis, 4 clusters provide a clearer and more interpretable categorization of the product descriptions and reviews:
- (i) Books and Literature
- (ii) Housewares
- (iii) Electronics
- (iv) Fashion and Apparel

These categories are distinct enough to offer meaningful insights while maintaining high clustering quality as indicated by the silhouette scores as explained below.

The silhouette scores using the cosine metric for 4 clusters provide a better silhouette score, which suggests better-defined clusters. The silhouette scores for 4 clusters are as follows:

- a) **Silhouette Score Euclidean (K-Means):** 0.54
- b) **Silhouette Score Cosine (K-Means):** 0.74
- c) **Silhouette Score Manhattan (K-Means):** 0.53

To easily understand the scatter plot below, each of the dots represents a single product description incorporated with its review, while the cluster centers (marked with a small 'x') are centrally located within their respective clusters, which are the product categories:
- 0 - Books and Literature
- 1 - Housewares
- 2 - Electronics
- 3 - Fashion and Apparel

**Figure 1.4: Visualization of Product Category Clusters of Four Clusters**

The figure above shows the visualization of the final clusters and cluster centers, with the points grouped more tightly within clusters and clearer boundaries between clusters with minimal overlap.

The clearer cluster separation can be seen where the clusters are more distinct and well-separated compared to the five clusters in the earlier visualization. Each cluster represents a unique group of data points (e.g., 0 - Books and Literature, 1 - Housewares, 2 - Electronics, 3 - Fashion and Apparel).

Furthermore, the new number of clusters has a higher silhouette score, indicating better cohesion within clusters and better separation between clusters.

In conclusion, the optimal clustering provided clearer and more interpretable results by having more distinct categories, which aids in better insights and decision-making. Additionally, higher silhouette scores validate the effectiveness of the optimal clustering, which improves the cluster quality.

# 1.3 Hierarchical Clustering

The standardized vectorization in this project is done using TF-IDF vectorizer. After the vectorization, the dimensionality of the dataframe is reduced due to the kernel crashing caused by insufficient storage. Thus, Principal Component Analysis (PCA) with a dimensionality of 2 is applied on the dataset before implementing the Agglomerative Hierarchical model.

To implement the Agglomerative Hierarchical model, the number of clusters must be identified. The clusters could be easily distinguished using a dendrogram. There are several types of linkages used to create the dendrogram, including Single Linkage, Complete Linkage, Average Linkage, Centroid Linkage, and Ward’s Linkage.

Each linkage method will produce different results. In this project, the chosen linkages are **Complete Linkage** and **Ward Linkage** as these are commonly used. 
- **Complete Linkage** produces a compact dendrogram with clusters emerging at higher distances, as shown in **Figure 1.5**.
- **Ward Linkage** produces a tidier structure because it emphasizes minimizing the variance within each cluster, as shown in **Figure 1.6**.

**Figure 1.5: Dendrogram with Complete Linkage**

**Figure 1.6: Dendrogram with Ward Linkage**

When comparing **Figure 1.5** with **Figure 1.6**, the dendrogram using Ward Linkage produces a more concise and interpretable visualization compared to Complete Linkage.

That being said, the optimal number of clusters is yet to be determined. Thus, evaluation using the Silhouette Score is used to determine the appropriate number of clusters. **Figure 1.7** shows the silhouette scores from cluster numbers 4 to 8. The scores are also visualized using a bar plot in **Figure 1.8**.

**Figure 1.7: Silhouette Score for Hierarchical Agglomerative**

**Figure 1.8: Visualization for Silhouette Score of Hierarchical Agglomerative**

It is shown that the optimum number of clusters in Hierarchical Agglomerative is **5**, with a silhouette score of **0.49**. Thus, the optimum number of clusters is picked, and the data is applied into the Hierarchical Agglomerative model. The clusters are visualized using a scatter plot as shown in **Figure 1.9**.

**Figure 1.9: Clusters in Hierarchical Agglomerative**

In conclusion, the Hierarchical Agglomerative model considers there are five clusters within the dataset. This finding is supported by the silhouette score evaluation to determine the optimal number of clusters.


# 1.4 DBSCAN

The analysis revealed that **DBSCAN** is effective in identifying clusters with irregular shapes, which is a significant advantage over other clustering algorithms such as K-Means that assume spherical cluster shapes. The parameter selection played a crucial role in the performance of the algorithm. An appropriate choice of ε and MinPts ensured that meaningful clusters were identified without excessive noise classification.

### Figure 1.10: Top 20 TF-IDF Features

The X-axis (tf-idf score) in Figure 1.10 represents the Term Frequency-Inverse Document Frequency value, which indicates the importance of a word in a document relative to a corpus.  
The Y-axis (feature) lists the top 20 terms with the highest TF-IDF scores.

Words with higher TF-IDF scores are more unique and significant in the context of the documents they appear in. In this chart, **"cotton"** has the highest TF-IDF score, indicating it's a particularly important term in the analyzed documents.

Terms such as **"set," "book," "size," "product,"** and **"black"** are also significant, suggesting frequent and unique use in the dataset. This information is useful for understanding the main themes and distinguishing terms in the dataset. For instance, in an e-commerce context, "cotton" might be prevalent in clothing product descriptions, while "book" might be important in a subset of literature-related products.

### Figure 1.11: 2D Visualization of Reduced Data

The scatter plot in Figure 1.11 shows the distribution of data points in a two-dimensional space. Each point represents a reduced-dimension representation of an original data point from the dataset.  

- Areas with higher densities of points suggest regions where data points have similar characteristics.
- The plot shows a dense region towards the bottom left, indicating a large number of data points with similar values.
- Distinct groupings and outliers are visible, particularly in the upper right portion of the plot, possibly indicating clusters with significantly different characteristics.

Such visualizations are useful for understanding the dataset structure before applying clustering algorithms like DBSCAN. The visible clusters and outliers can guide the choice of parameters and interpretation of results.

## 1.4.1 Parameter Selection in DBSCAN Clustering

DBSCAN is highly sensitive to the values of **eps** and **min_samples**, making it crucial to understand their selection:

- **min_samples** should be at least one greater than the number of dimensions: `min_samples >= Dimensions + 1`.
- It does not make sense to set `min_samples = 1` because it would result in each point being its own cluster. It must be at least 3, and generally about twice the number of dimensions.  
- Domain knowledge can also guide the value selection.

### Figure 1.12: DBSCAN Cluster (eps=0.5, min_samples=5)

### Figure 1.13: DBSCAN Cluster (eps=best_eps, min_samples=best_min_sample)

- Figure 1.12 visualizes clusters identified by DBSCAN on reduced data, showing irregular and not well-defined clusters.
- Figure 1.13 shows clusters after parameter tuning, resulting in more regular and meaningful groupings.
- This comparison highlights the importance of parameter optimization to achieve accurate and actionable clustering results.

## 1.4.2 Silhouette Analysis

### a) Silhouette Plot

The silhouette plot visually evaluates the quality of clustering:

- **X-axis:** Silhouette coefficient values (ranging from -0.4 to 1.0).
- **Y-axis:** Cluster labels (0 to 25000).

### b) Silhouette Score

- The average silhouette score is approximately **0.77**.
- A higher score indicates better separation between clusters.
- Points closer to 1 are well-clustered; points near 0 or negative may be misclassified.
- The red dashed vertical line represents the average silhouette score.

### Figure 1.14: The Silhouette Plot for Various Clusters

# 1.5 Analysis of Clusters

From the scatter plot below:

- Each dot represents a single product description incorporating its review.
- Small red crosses indicate the product categories and their labels.

### Figure 1.15: Visualization of Product Category Clusters of Four Clusters

Using the best cluster results from K-Means clustering, we explored further to understand the product descriptions within each cluster. The text clustering analysis of product descriptions and reviews revealed distinct themes providing insights into product categories.  

Each cluster serves as a **categorical snapshot** of key aspects and features discussed in reviews, helping better understand product characteristics.

### Cluster 0: Books and Literature

**Top Keywords:** "book", "author", "time", "life", "new", "review", "world", "university", "one", "story"

**Description:**  
- Predominantly contains product descriptions and reviews related to books and literature.
- Themes include discussions around authors, life experiences, storytelling, and new releases.

### Figure 1.16: WordCloud for Cluster 0 - Books and Literature

---

### Cluster 1: Housewares

**Top Keywords:** "steel", "set", "cm", "stainless", "home", "product", "table", "size", "color", "easy"

**Description:**  
- Descriptions of household items focusing on materials like stainless steel and product sets.
- Highlights practical aspects such as size, color, and ease of use.

### Figure 1.17: WordCloud for Cluster 1 - Housewares

---

### Cluster 2: Electronics

**Top Keywords:** "usb", "bluetooth", "cable", "speaker", "camera", "device", "mm", "audio", "power", "black"

**Description:**  
- Centered around electronic devices and accessories.
- Focuses on connectivity, audio equipment, and technical specifications.

### Figure 1.18: WordCloud for Cluster 2 - Electronics

---

### Cluster 3: Fashion and Apparel

**Top Keywords:** "cotton", "woman", "men", "wear", "fabric", "fit", "boy", "girl", "made", "pack"

**Description:**  
- Captures fashion and apparel product descriptions.
- Emphasizes materials (like cotton), target demographics, and fit.

### Figure 1.19: WordCloud for Cluster 3 - Fashion and Apparel

The analysis of product descriptions and reviews using **K-Means clustering** with **TF-IDF vectorization** effectively categorized text data into meaningful clusters.  
Each cluster reveals specific themes related to different product categories, providing valuable insights into product features and customer focus.

# 2. Conclusion

This comprehensive report analyzing clustering algorithms—**K-Means, Hierarchical Agglomerative, and DBSCAN**—provides insights into which algorithms perform best for clustering words in an E-commerce dataset:

- **K-Means** achieved the highest silhouette score of **0.73** using the **Cosine** metric.
- Using **Manhattan** and **Euclidean** metrics, K-Means achieved lower scores (**0.52** and **0.54** respectively).
- **Hierarchical Agglomerative** achieved a highest silhouette score of **0.49** using Euclidean metric (Ward linkage).

Although **DBSCAN** achieved the highest silhouette score (**0.77**), indicating very well-defined clusters:

- **K-Means** might still be preferred for practical use due to faster computation, scalability, and easier interpretability.

