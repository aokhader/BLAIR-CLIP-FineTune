# Video Script: BLaIR-CLIP Fine-Tuning for Product Recommendation

**Total Duration:** ~20 Minutes
**Speaker:** [Your Name]
**Date:** December 2025

---

## 0:00 - Introduction & Predictive Task (Topic 1)

**[Visual: Title Slide - "BLaIR-CLIP Fine-Tuning: Multimodal Product Recommendation"]**

**Speaker:**
"Hello everyone. Welcome to our presentation on 'BLaIR-CLIP Fine-Tuning'. In this project, we explore the intersection of Natural Language Processing and Computer Vision to improve product recommendation systems. Specifically, we are looking at how we can leverage the rich multimodal data available in e-commerce—text descriptions and product images—to build better retrieval models."

**[Visual: Slide - Predictive Task Definition]**
*   **Task:** Product Recommendation / Retrieval
*   **Input:** User History / Query (Text + Image)
*   **Output:** Ranked List of Items
*   **Goal:** Maximize relevance of top-k items

**Speaker:**
"First, let's define our **Predictive Task**. We are focusing on **Product Recommendation and Retrieval**. The core problem is: given a user's context—which could be their interaction history or a specific search query—we want to rank a candidate set of items such that the most relevant items appear at the very top.
This is a fundamental problem in web mining and e-commerce. Traditional methods often rely heavily on just text or just user-item interaction matrices. Our goal is to assess if adding visual signals—product images—via a multimodal approach actually improves performance compared to these unimodal baselines."

**[Visual: Slide - Evaluation Strategy]**
*   **Metrics:** Recall@K (Hit Rate), AUC (Ranking Quality)
*   **Baselines:** TF-IDF (Text), Matrix Factorization (CF), BLaIR (Text-Only)
*   **Validity:** Leave-One-Out Split (Temporal)

**Speaker:**
"To evaluate our model, we use a standard retrieval setup.
For **metrics**, we focus on **Recall@K**, which measures if the ground-truth item appears in the top K recommendations—crucial for user experience where people rarely scroll past the first page. We also use **AUC** to measure the overall quality of the ranking.
We compare our proposed BLaIR-CLIP model against three distinct **baselines**:
1.  **TF-IDF**: A classic text-based baseline from our Text Mining curriculum.
2.  **Matrix Factorization (BPR)**: A standard collaborative filtering approach from Recommender Systems.
3.  **BLaIR**: The state-of-the-art text-only model we are building upon.
To ensure the **validity** of our predictions, we use a **Leave-One-Out** temporal split. We train on past interactions and test on the very next interaction, strictly avoiding data leakage from the future."

---

## 4:00 - Exploratory Analysis (Topic 2)

**[Visual: Slide - Dataset Overview]**
*   **Source:** Amazon Reviews 2023 (Appliances Category)
*   **Features:** Title, Description, Features, Images, User Reviews
*   **Stats:** [Insert N] Users, [Insert M] Items

**Speaker:**
"Moving on to our **Exploratory Analysis**.
We utilized the **Amazon Reviews 2023** dataset, specifically the **Appliances** category. This dataset, curated by the McAuley Lab, is excellent for this task because it provides rich metadata—not just ratings, but full text descriptions, feature lists, and crucially, links to product images."

**[Visual: Code Snippet - Data Preprocessing]**
```python
# Conceptual Preprocessing
text_input = title + " " + description + " " + features
# User Filtering
if user_interactions < 2: drop_user()
# Split
train = interactions[:-1]
test = interactions[-1]
```

**Speaker:**
"For **preprocessing**, we had to clean and format the data significantly.
First, for the text representation, we concatenated the product `title`, `description`, and `features` into a single string. This gives the model a comprehensive semantic view of the item.
Second, we performed **User Filtering**. We removed 'cold-start' users who had fewer than 2 interactions, as we need at least one interaction for training and one for testing.
Finally, as mentioned, we applied a temporal split: the last interaction for each user is reserved for the test set, while all prior interactions form the training set."

**[Visual: Plot - Data Distribution (Bar Chart from Notebook)]**
*   *Show the bar chart comparing Train vs. Test interaction counts.*

**Speaker:**
"Here you can see the distribution of our data. The training set is naturally much larger, capturing the user's history, while the test set contains exactly one target item per user. This imbalance is expected and mirrors the real-world scenario of predicting the 'next' single action."

---

## 8:00 - Modeling (Topic 3)

**[Visual: Slide - Modeling Approach]**
*   **Formulation:** Contrastive Learning (Dual Encoder)
*   **Architecture:** RoBERTa (Text) + CLIP (Vision)
*   **Objective:** Maximize similarity between (User, Positive Item) pairs

**Speaker:**
"Now, let's dive into **Modeling**.
We formulate the recommendation task as a **Contrastive Learning** problem. The intuition is simple: we want to learn a shared vector space where the representation of a user (or their query) is geometrically close to the representation of the item they actually bought, and far away from random 'negative' items."

**[Visual: Code Walkthrough - BlairCLIPDualEncoder]**
*   *Show `blair/multimodal/blair_clip.py` snippet*

**Speaker:**
"Let's look at the actual implementation in `blair/multimodal/blair_clip.py`.
We defined a class called `BlairCLIPDualEncoder`. This is a **Dual Encoder** architecture.
On one side, we have the **Text Encoder**, initialized from the pre-trained **BLaIR** model (which is based on RoBERTa). It processes the item descriptions using the `encode_text` method.
On the other side, we added a **Vision Encoder**, initialized from **OpenAI's CLIP**. This processes the product images using the `encode_image` method.
Crucially, we project both of these representations into a shared dimension using linear projection layers (`self.text_projection` and `self.image_projection`)."

**[Visual: Code Snippet - Forward Pass]**
```python
# From blair/multimodal/blair_clip.py
def forward(self, ...):
    # ... (encoding steps) ...
    logit_scale = self.logit_scale.exp().clamp(max=100)
    logits_per_text = logit_scale * gathered_text @ gathered_images.t()
    
    # Symmetric Cross Entropy Loss
    clip_loss = (
        self.cross_entropy(logits_per_text, labels) + 
        self.cross_entropy(logits_per_image, labels)
    ) / 2.0
```

**Speaker:**
"In the `forward` pass, you can see exactly how we compute the loss.
We calculate the dot product similarity between the gathered text embeddings and image embeddings: `logits_per_text = logit_scale * gathered_text @ gathered_images.t()`.
We then use a symmetric **Cross Entropy Loss** to maximize this similarity for matching pairs. This forces the model to align the visual and textual understanding of a product, which is key for our multimodal recommendation task."

**[Visual: Slide - Trade-offs]**
*   **TF-IDF:** Fast, Simple, but no semantic meaning.
*   **Matrix Factorization:** Good for personalization, but fails on cold-start items.
*   **BLaIR-CLIP:** Semantic + Visual understanding, handles cold-start, but computationally expensive (requires GPUs).

**Speaker:**
"We chose this approach because it addresses the limitations of our baselines. TF-IDF is fast but lacks semantic understanding. Matrix Factorization is great for personalization but fails completely on new items with no interaction history. BLaIR-CLIP is computationally heavier, but it can 'read' and 'see' the product, allowing it to recommend even new items effectively."

---

## 13:00 - Evaluation (Topic 4)

**[Visual: Slide - Evaluation Protocol]**
*   **Method:** Ranking-based
*   **Test Set:** 1 Positive vs. All Negatives (or Sampled Negatives)
*   **Metrics:** Recall@10, AUC

**Speaker:**
"For **Evaluation**, we implemented a rigorous ranking protocol. For every user in our test set, we take their true next item and mix it with a set of negative items (other products they didn't interact with). We then ask our models to rank this list."

**[Visual: Table/Plot - Results Comparison]**
*   *Show the table/plot from the notebook comparing TF-IDF, MF, and BLaIR-CLIP.*

**Speaker:**
"Here are our results.
Interestingly, **TF-IDF** performed quite strongly with an AUC of around **0.71**. This suggests that simple text matching is a very strong baseline in this specific category—people search for appliances using very specific keywords.
**Matrix Factorization**, on the other hand, struggled with an AUC of roughly **0.48**, performing slightly worse than random. This highlights the 'sparsity' problem in this dataset—many users have very few interactions, making it hard for collaborative filtering to learn good embeddings without content features.
We are currently finalizing the training for **BLaIR-CLIP**, but our hypothesis is that by adding the visual signal, we will see a lift over the text-only baselines, particularly for items where visual aesthetics matter."

**[Visual: Code Snippet - Evaluation Loop]**
```python
# baseline_utils.py or run_baselines.py
model.run() # Computes metrics
print("TF-IDF Results:", results)
```

**Speaker:**
"Our evaluation code, shown here, standardizes this process across all models, ensuring a fair 'apples-to-apples' comparison using the exact same data splits and metric calculations."

---

## 17:00 - Related Work & Conclusion (Topic 5)

**[Visual: Slide - Related Work]**
*   **BLaIR (Hou et al., 2024):** Pre-training on Amazon Reviews.
*   **CLIP (Radford et al., 2021):** Learning transferrable visual models.
*   **SimCSE (Gao et al., 2021):** Contrastive sentence embeddings.

**Speaker:**
"Finally, let's discuss **Related Work**. Our project doesn't exist in a vacuum.
We heavily leverage **BLaIR**, a recent paper from UCSD that showed pre-training language models on Amazon reviews significantly boosts recommendation performance. We use their checkpoints as our starting point.
We also build on **CLIP**, which revolutionized computer vision by showing that you can learn powerful image representations from natural language supervision.
And methodologically, our loss function is inspired by **SimCSE**, which applied contrastive learning to sentence embeddings.
Our contribution is combining these three ideas: taking the domain-specific text power of BLaIR and fusing it with the visual power of CLIP using a SimCSE-style objective."

**[Visual: Slide - Conclusion]**

**Speaker:**
"In conclusion, we've built a multimodal recommendation system that goes beyond simple keywords or user IDs. By 'looking' at the products, we aim to provide more accurate and relevant recommendations. Thank you for listening!"

---
