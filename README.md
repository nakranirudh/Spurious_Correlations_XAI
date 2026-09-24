## Evaluating the efficacy of explanation methods in uncovering spurious correlations

Modern day machine learning tools often exploit non-interpretable features from the train dataset.
In addition to achieving state-of-the-art accuracy on benchmarks, there is also an underlying need
to understand which concepts are being learned by the designed model. A natural consequence to
designing without such robustness in mind is the fact that a model might learn from unwanted microsignals
that might be unexplainable through a human lens. In the context of Natural Language based
tasks, spurious correlations often come from unexpected sources. Adversaries that flip decisions
can be generated at a character level without changing the semantic content. [Ebrahimi et al.
2018a],[Ebrahimi et al. 2018b] Spurious correlations can also arise due to biased datasets. Models
trained on such corpora are prone to learning unwanted correlations even though the model itself
might be robust in theory.[Zhang et al. 2018], [Bolukbasi et al. 2016]. A great practical example of a
spurious correlation arisen through biases in datasets is given by [Wang and Culotta 2020a] who
find that words like Spielberg and New York Subway have high co-occurences with the sentiment of
movie reviews.

We adopt the generic framework suggested in [Wang et al. 2021] for identifying and mitigating
spurious correlations.

- Important Token Identification: Using the underlying basic causal mapping between the
input tokens and the output class in a text classification problem, we evaluate different
explainability techniques (such as LIME, IG, etc.) and regimes (such as local vs global explanations,
post-hoc vs learned explanations) to identify how effective they are in recovering
spurious correlations.

- Identifying spurious correlations: We adopt the approach outlined by [Wang et al. 2021]
and use a cross-dataset consistency based approach with the goal being to identify globally
meaningful tokens (out of domain importance) and locally meaningful tokens (in domain
importance).

This is of particular importance to NLP tasks where we can have a lot of unintended bias built
into the dataset due to broad web crawling based collection procedures. We are explaining the
model’s underlying working by uncovering data that might be noise such that researchers and
broadly humans think these data samples are not useful for the task its being used for. In this study,
we gear the explanations towards Machine Learning researchers.
