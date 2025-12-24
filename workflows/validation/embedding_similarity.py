"""
Embedding Similarity Validation Layer (Layer 3).

Uses embeddings to detect semantic drift between original PRP and implementation.
A similarity score ≥0.9 is recommended to indicate high fidelity (faithful implementation, not just similar).
The threshold parameter defaults to 0.9 but can be overridden by the caller for other use cases.
"""

from typing import Dict, Any, List
import os
import numpy as np
from openai import OpenAI

from workflows.schemas.prp_schemas import ValidationResult


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score between 0.0 and 1.0
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def get_embedding(text: str, embedding_config: Dict[str, Any]) -> np.ndarray:
    """
    Get embedding vector for text using OpenAI API.

    Args:
        text: Text to embed
        embedding_config: Configuration (provider, model, dimensions)

    Returns:
        Embedding vector as numpy array

    Raises:
        ValueError: If provider not supported or API error
    """
    if embedding_config.get("provider") != "openai":
        raise ValueError(f"Unsupported embedding provider: {embedding_config.get('provider')}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)

    try:
        response = client.embeddings.create(
            model=embedding_config.get("model", "text-embedding-3-small"),
            input=text,
            dimensions=embedding_config.get("dimensions", 1536)
        )

        return np.array(response.data[0].embedding)

    except Exception as e:
        raise ValueError(f"Failed to get embedding: {str(e)}")


def embedding_similarity_validation(
    original_text: str,
    implementation_text: str,
    embedding_config: Dict[str, Any],
    threshold: float = 0.9
) -> ValidationResult:
    """
    Layer 3: Embedding Similarity Check

    Compares original PRP text with implementation description using embeddings.
    Detects semantic drift between what was requested and what was implemented.

    Args:
        original_text: Original PRP requirements
        implementation_text: Description of what was implemented
        embedding_config: Embedding configuration
        threshold: Minimum similarity required (default 0.9 for high fidelity)

    Returns:
        ValidationResult with similarity score and assessment

    Example:
        original = "Implement Docker GHCR integration with secure credentials"
        implementation = "Added Docker GHCR with encrypted secret management"
        result = embedding_similarity_validation(original, implementation, config, 0.9)
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        # Get embeddings for both texts
        original_embedding = get_embedding(original_text, embedding_config)
        implementation_embedding = get_embedding(implementation_text, embedding_config)

        # Calculate similarity
        similarity = cosine_similarity(original_embedding, implementation_embedding)

        # Check against threshold
        passed = similarity >= threshold

        if not passed:
            errors.append(
                f"Semantic drift detected: similarity={similarity:.3f} < threshold={threshold}"
            )
            errors.append(
                "Implementation may not faithfully match original requirements"
            )

        if 0.85 <= similarity < threshold:
            warnings.append(
                f"Similarity close to threshold: {similarity:.3f} (threshold={threshold})"
            )

        return ValidationResult(
            layer_name="embedding_similarity",
            passed=passed,
            errors=errors,
            warnings=warnings,
            confidence=float(similarity),
            details={
                "similarity_score": float(similarity),
                "threshold": threshold,
                "original_length": len(original_text),
                "implementation_length": len(implementation_text),
                "embedding_dimensions": len(original_embedding)
            }
        )

    except Exception as e:
        return ValidationResult(
            layer_name="embedding_similarity",
            passed=False,
            errors=[f"Embedding similarity check failed: {str(e)}"],
            warnings=[],
            confidence=0.0,
            details={"error_type": type(e).__name__}
        )


def batch_embedding_similarity(
    text_pairs: List[tuple[str, str]],
    embedding_config: Dict[str, Any],
    threshold: float = 0.9
) -> List[ValidationResult]:
    """
    Perform embedding similarity checks on multiple text pairs.

    Args:
        text_pairs: List of (original, implementation) text pairs
        embedding_config: Embedding configuration
        threshold: Minimum similarity threshold

    Returns:
        List of ValidationResult objects, one per pair
    """
    results = []

    for original, implementation in text_pairs:
        result = embedding_similarity_validation(
            original,
            implementation,
            embedding_config,
            threshold
        )
        results.append(result)

    return results


def extract_key_requirements(prp_text: str) -> List[str]:
    """
    Extract key requirements from PRP text for granular similarity checking.

    This is a placeholder - actual implementation would use LLM to extract
    individual requirements from PRP.

    Args:
        prp_text: Full PRP markdown text

    Returns:
        List of individual requirement statements
    """
    # TODO: Use Claude API to extract requirements
    # For now, split by sections or paragraphs
    requirements = []

    # Simple paragraph splitting
    paragraphs = prp_text.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        if len(para) > 50 and not para.startswith('#'):
            requirements.append(para)

    return requirements
