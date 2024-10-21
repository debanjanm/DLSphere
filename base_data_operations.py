from sklearn.datasets import make_classification
import pandas as pd

def create_classification_dataset(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, 
                                  n_classes=2, n_clusters_per_class=2, class_sep=1.0, random_state=42):
    """
    Create a classification dataset using sklearn's make_classification function.
    
    Parameters:
    - n_samples: The number of samples
    - n_features: The total number of features
    - n_informative: The number of informative features
    - n_redundant: The number of redundant features
    - n_classes: The number of classes (or labels)
    - n_clusters_per_class: The number of clusters per class
    - class_sep: The factor multiplying the hypercube size
    - random_state: Random state for reproducibility
    
    Returns:
    - X: pandas DataFrame of features
    - y: pandas Series of labels
    """
    
    # Generate the dataset
    X, y = make_classification(n_samples=n_samples, 
                               n_features=n_features, 
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               class_sep=class_sep,
                               random_state=random_state)
    
    # Convert to pandas DataFrame and Series
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    
    return X, y

# Example usage
X, y = create_classification_dataset()
print(X.shape, y.shape)
print(X.head())
print(y.head())