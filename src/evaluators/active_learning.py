"""
Active Learning Priority Screening Ranker for sTRCview.
Implements ASReview-style continuous active learning priority queueing
using TF-IDF text representations and balanced statistical classifiers
(Logistic Regression / SGDClassifier).
"""

from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity

from src.models.article import Article


class ActiveLearningRanker:
    """
    ASReview-style Active Learning Priority Ranker for Title & Abstract screening.
    Continuously trains on researcher decisions ('Include' vs 'Exclude')
    and prioritizes the most probable relevant articles for subsequent screening.
    """

    def __init__(
        self,
        classifier: str = "logistic_regression",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        """
        Parameters:
            classifier: 'logistic_regression' or 'sgd'
            max_features: Maximum vocabulary size for TF-IDF
            ngram_range: N-gram range for word tokens
            random_state: Random seed for deterministic reproducibility
        """
        self.classifier_name = classifier.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,
        )

        if self.classifier_name in ("sgd", "sgdclassifier"):
            self.model = SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=1000,
            )
        else:
            self.model = LogisticRegression(
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=1000,
                C=1.0,
            )

        self.is_fitted: bool = False
        self.training_samples: int = 0
        self.positive_count: int = 0
        self.negative_count: int = 0

    @staticmethod
    def _get_text(article: Any) -> str:
        """Extracts combined Title, Abstract, and Keywords for feature extraction."""
        if isinstance(article, dict):
            title = str(article.get("title") or "")
            abstract = str(article.get("abstract") or "")
            keywords = str(article.get("keywords") or "")
        else:
            title = getattr(article, "title", "") or ""
            abstract = getattr(article, "abstract", "") or ""
            keywords = getattr(article, "keywords", "") or ""

        parts = [p.strip() for p in [title, abstract, keywords] if p and p.strip()]
        return " . ".join(parts) if parts else "empty"

    @staticmethod
    def _get_decision(article: Any) -> Optional[int]:
        """
        Extracts binary label:
          1 for Include / Included / True
          0 for Exclude / Excluded / False
          None for Unscreened / Borderline / unknown
        """
        if isinstance(article, dict):
            raw = (
                article.get("decision")
                or article.get("human_decision")
                or article.get("label")
                or article.get("final_decision")
            )
        else:
            raw = (
                getattr(article, "decision", None)
                or getattr(article, "human_decision", None)
                or getattr(article, "label", None)
                or getattr(article, "final_decision", None)
            )

        if raw is None:
            return None

        if isinstance(raw, (int, bool)):
            return 1 if raw else 0

        str_raw = str(raw).strip().lower()
        if str_raw in ("include", "included", "yes", "1", "true", "positive"):
            return 1
        elif str_raw in ("exclude", "excluded", "no", "0", "false", "negative"):
            return 0
        return None

    def fit_and_rerank(
        self,
        labeled_articles: List[Any],
        unlabeled_articles: List[Any],
        *,
        preserve_model_scores: bool = False,
    ) -> List[Any]:
        """
        Fits or updates the active learning model on labeled data and re-ranks
        unlabeled articles in descending order of predicted relevance score.

        Parameters:
            labeled_articles: Articles that have received 'Include' or 'Exclude' decisions
            unlabeled_articles: Candidate articles pending screening

        Returns:
            List[Any]: The re-ranked list of unlabeled articles (with updated relevance_score)
        """
        if not unlabeled_articles:
            return []

        # Extract labeled samples
        labeled_data: List[Tuple[str, int]] = []
        for art in labeled_articles:
            y = self._get_decision(art)
            if y is not None:
                text = self._get_text(art)
                labeled_data.append((text, y))

        unlabeled_texts = [self._get_text(art) for art in unlabeled_articles]
        n_unlabeled = len(unlabeled_articles)

        # Case 1: No valid labeled data -> return unlabeled as-is
        if not labeled_data:
            return unlabeled_articles

        texts, labels = zip(*labeled_data)
        labels = np.array(labels)
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        self.training_samples = len(labels)
        self.positive_count = len(pos_indices)
        self.negative_count = len(neg_indices)

        # Fit vectorizer on all available texts (labeled + unlabeled) for rich vocabulary
        all_corpus = list(texts) + unlabeled_texts
        if not any(self.vectorizer.build_analyzer()(text) for text in all_corpus):
            return unlabeled_articles
        self.vectorizer.fit(all_corpus)

        X_train = self.vectorizer.transform(texts)
        X_unlabeled = self.vectorizer.transform(unlabeled_texts)

        # Case 2: Only 1 class available (Cold-start phase)
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            if self.positive_count > 0:
                # Rank by cosine similarity to known positive articles
                pos_vectors = X_train[pos_indices]
                # Average centroid of positive embeddings
                centroid = np.asarray(pos_vectors.mean(axis=0))
                similarities = cosine_similarity(X_unlabeled, centroid).flatten()
                scores = np.clip(similarities, 0.0, 1.0)
            else:
                # Only negatives known: assign default low scores
                scores = np.full(n_unlabeled, 0.1)

            self.is_fitted = False
        else:
            # Case 3: Both positive and negative classes available -> full model training
            self.model.fit(X_train, labels)
            self.is_fitted = True

            # Predict probability of class 1 (Include)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_unlabeled)
                # Find column index for class 1
                class_1_idx = np.where(self.model.classes_ == 1)[0]
                if len(class_1_idx) > 0:
                    scores = probs[:, class_1_idx[0]]
                else:
                    scores = probs[:, -1]
            elif hasattr(self.model, "decision_function"):
                df = self.model.decision_function(X_unlabeled)
                # Sigmoid transform to [0, 1]
                scores = 1.0 / (1.0 + np.exp(-df))
            else:
                scores = np.full(n_unlabeled, 0.5)

        # Update scores on articles and sort
        scored_pairs: List[Tuple[float, Any]] = []
        for i, art in enumerate(unlabeled_articles):
            score_val = round(float(scores[i]), 4)
            if preserve_model_scores:
                art.priority_score = score_val
            elif isinstance(art, dict):
                art["relevance_score"] = score_val
                art["model_used"] = f"ActiveLearning-{self.classifier_name}"
            else:
                art.relevance_score = score_val
                art.model_used = f"ActiveLearning-{self.classifier_name}"
            scored_pairs.append((score_val, art))

        # Sort descending by relevance score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        return [art for _, art in scored_pairs]

    @staticmethod
    def check_stopping_rule(
        consecutive_irrelevant: int,
        threshold: int = 50,
        total_screened: Optional[int] = None,
        total_included: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tracks a reviewer-defined consecutive exclusion threshold.

        Parameters:
            consecutive_irrelevant: Number of consecutive exclusions currently encountered
            threshold: Number of consecutive non-relevant articles required to trigger stop (default: 50)
            total_screened: Optional count of all articles screened so far
            total_included: Optional count of all articles included so far

        Returns:
            Dict containing:
              - should_stop: bool
              - consecutive_irrelevant: int
              - threshold: int
              - progress_pct: float (0.0 to 100.0)
              - message: str
              - recommendation: str ('STOP' or 'CONTINUE')
        """
        consecutive = max(0, int(consecutive_irrelevant))
        thresh = max(1, int(threshold))
        should_stop = consecutive >= thresh
        progress_pct = round(min(100.0, (consecutive / thresh) * 100.0), 1)

        if should_stop:
            msg = (
                f"Durma Kriteri Sağlandı: Ardışık {consecutive} adet ilgisiz makale elendi "
                f"(Eşik: {thresh}). Bu eşik kalan kayıtların duyarlılığını ölçmez."
            )
            rec = "STOP"
        else:
            remaining = thresh - consecutive
            msg = (
                f"Tarama Devam Ediyor: Son {consecutive} makale ilgisiz elendi. "
                f"Durma eşiğine kalan: {remaining} makale (İlerleme: %{progress_pct})."
            )
            rec = "CONTINUE"

        result = {
            "should_stop": should_stop,
            "consecutive_irrelevant": consecutive,
            "threshold": thresh,
            "remaining_to_stop": max(0, thresh - consecutive),
            "progress_pct": progress_pct,
            "message": msg,
            "recommendation": rec,
        }

        if total_screened is not None:
            result["total_screened"] = total_screened
        if total_included is not None:
            result["total_included"] = total_included

        return result
