"""Hidden Markov Model implementation."""

from hmm.types import FloatArray


class HMM:
    """Hidden Markov Model."""

    def __init__(self, transition: FloatArray, emission: FloatArray) -> None:
        """Initialise model.

        Args
            transition (FloatArray): Transition probability matrix for model states.
            emission (FloatArray): Emission probability matrix for observations.
        """
        self.transition = transition  # Denoted uppercase gamma (Γ).
        self.emission = emission  # P(X = x | Z = z) = Poisson - with mean λ_z > 0.
