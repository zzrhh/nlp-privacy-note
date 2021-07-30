## Information Leakage in Embedding Models

### Embedding inversion attacks

the adversary’s goal is to invert a target embedding Φ(x ∗ ) and recover words in x ∗ . We consider attacks that involve both black-box and white-box access to Φ. The adversary is also allowed access to an unlabeled Daux and in both scenarios is able to evaluate Φ(x) for x ∈ Daux.

The goal of inverting text embeddings is to recover the target input texts x ∗ from the embedding vectors Φ(x ∗ ) and access to Φ. We focus on inverting embeddings of short texts and for practical reasons.

#### white - box inversion

In a white-box scenario, we assume that the adversary has access to the embedding model Φ’s parameters and architecture.

#### black - box inversion

In the black-box scenario, we assume that the adversary only has query access to the embedding model Φ, i.e., adversary observes the output embedding Φ(x) for a query x. Gradient-based optimization is not applicable as the adversary does not have access to the model parameters and thus the gradients

### Sensitive attribute inference attacks

the adversary’s goal is to infer sensitive attribute s ∗ of a secret input x ∗ from a target embedding Φ(x ∗ ).

### Membership inference attacks.

we expand the definition of training data membership to consider this data with their contexts, which is used in training.

 For unsupervised embedding models trained on units of data in context, we thus wish to infer the membership of a context of data.